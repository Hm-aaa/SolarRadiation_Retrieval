#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接请求 Open-Meteo Satellite Radiation API，解析 JSON 并保存 CSV。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import requests

API = "https://satellite-api.open-meteo.com/v1/archive"
MODEL = "jma_jaxa_himawari"
UA = "himawari-direct-api/1.0 (research; python-requests)"
ROWS_PER_DAY = 144


@dataclass(frozen=True)
class Point:
    point_id: str
    lat: float
    lon: float
    lat_text: str
    lon_text: str


class Task(NamedTuple):
    coord_path: Path
    point_index: int
    point: Point
    final_path: Path


class TaskResult(NamedTuple):
    task: Task
    rows: int
    status: str
    error: str = ""


class RateLimiter:
    def __init__(self, interval: float):
        self.interval = max(0.0, interval)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self.interval

        delay = start_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_thread_local = threading.local()


def get_session(args) -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is not None:
        return session

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": UA,
            "Accept": "application/json",
        }
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.workers,
        pool_maxsize=args.workers,
        max_retries=0,
    )
    session.mount("https://", adapter)
    _thread_local.session = session
    return session


def read_points(path: Path) -> list[Point]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path} 没有表头")

        columns = {name.strip().lower(): name for name in reader.fieldnames}
        lat_col = columns.get("lat") or columns.get("latitude")
        lon_col = columns.get("lon") or columns.get("longitude")
        name_col = columns.get("name") or columns.get("id")

        if not lat_col or not lon_col:
            raise ValueError(f"{path} 必须包含 lat,lon 或 latitude,longitude")

        points: list[Point] = []
        for row_no, row in enumerate(reader, start=1):
            lat_text = (row.get(lat_col) or "").strip()
            lon_text = (row.get(lon_col) or "").strip()
            if not lat_text or not lon_text:
                continue

            point_id = (row.get(name_col) or "").strip() if name_col else ""
            if not point_id:
                point_id = f"{path.stem}_{row_no:06d}"

            points.append(
                Point(
                    point_id=point_id,
                    lat=float(lat_text),
                    lon=float(lon_text),
                    lat_text=lat_text,
                    lon_text=lon_text,
                )
            )

    if not points:
        raise ValueError(f"{path} 中没有有效坐标")
    return points


def safe_filename(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return name or "point"


def make_output_path(
    draw_dir: Path,
    point_index: int,
    point: Point,
    use_gzip: bool,
) -> Path:
    suffix = ".csv.gz" if use_gzip else ".csv"
    lat_name = safe_filename(point.lat_text)
    lon_name = safe_filename(point.lon_text)
    return draw_dir / (
        f"coords_{point_index:02d}_{lat_name}_{lon_name}{suffix}"
    )


def fetch_payload(
    session: requests.Session,
    api: str,
    params: dict,
    args,
    limiter: RateLimiter,
):
    last_error: Exception | None = None

    for attempt in range(args.retries):
        limiter.wait()

        try:
            response = session.get(
                api,
                params=params,
                timeout=(20, args.timeout),
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = min(60.0 * (2**attempt), args.max_retry_wait)

                if args.max_retry_wait > 0 and wait > args.max_retry_wait:
                    raise RuntimeError(
                        f"HTTP 429，要求等待 {wait:.0f}s，"
                        f"超过 --max-retry-wait {args.max_retry_wait}s"
                    )

                print(
                    f"    [429] 限流，等待 {wait:.0f}s 后重试 "
                    f"({attempt + 1}/{args.retries})",
                    flush=True,
                )
                time.sleep(wait)
                last_error = RuntimeError("HTTP 429 Too Many Requests")
                continue

            if response.status_code in (500, 502, 503, 504):
                wait = min(5.0 * (2**attempt), args.max_retry_wait)
                print(
                    f"    [retry] HTTP {response.status_code}，"
                    f"{wait:.0f}s 后重试 "
                    f"({attempt + 1}/{args.retries})",
                    flush=True,
                )
                time.sleep(wait)
                last_error = RuntimeError(f"HTTP {response.status_code}")
                continue

            response.raise_for_status()
            return response.json()

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc
            wait = min(5.0 * (2**attempt), args.max_retry_wait)
            print(
                f"    [retry] {exc}，{wait:.0f}s 后重试 "
                f"({attempt + 1}/{args.retries})",
                flush=True,
            )
            time.sleep(wait)

    raise last_error or RuntimeError("请求失败")


def payload_to_dataframe(payload: dict, point: Point) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError("API 返回中没有 hourly 数据")

    times = hourly.get("time") or []
    ghi_instant = hourly.get("shortwave_radiation_instant") or []
    ghi = hourly.get("shortwave_radiation") or []

    if not times:
        raise RuntimeError("API 返回中没有 time 数据")

    if not (len(times) == len(ghi_instant) == len(ghi)):
        raise RuntimeError(
            "API 返回长度不一致: "
            f"time={len(times)}, "
            f"instant={len(ghi_instant)}, "
            f"ghi={len(ghi)}"
        )

    time_utc = pd.to_datetime(
        times,
        format="%Y-%m-%dT%H:%M",
        errors="coerce",
    )

    return pd.DataFrame(
        {
            "time_utc": time_utc,
            "lat": point.lat_text,
            "lon": point.lon_text,
            "ghi_wm2": pd.to_numeric(
                pd.Series(ghi), errors="coerce"
            ),
            "ghi_instant_wm2": pd.to_numeric(
                pd.Series(ghi_instant), errors="coerce"
            ),
            "point_id": point.point_id,
        }
    )


def download_one(
    task: Task,
    start: date,
    end: date,
    args,
    limiter: RateLimiter,
    stop_event: threading.Event,
) -> TaskResult:
    if stop_event.is_set():
        return TaskResult(task=task, rows=0, status="cancelled")

    part_path = task.final_path.with_suffix(
        task.final_path.suffix + ".part"
    )

    try:
        params = {
            "latitude": f"{task.point.lat:.4f}",
            "longitude": f"{task.point.lon:.4f}",
            "hourly": (
                "shortwave_radiation_instant,"
                "shortwave_radiation"
            ),
            "models": MODEL,
            "temporal_resolution": "native",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }

        session = get_session(args)
        payload = fetch_payload(
            session=session,
            api=args.api,
            params=params,
            args=args,
            limiter=limiter,
        )
        dataframe = payload_to_dataframe(payload, task.point)

        part_path.unlink(missing_ok=True)
        dataframe.to_csv(
            part_path,
            index=False,
            date_format="%Y-%m-%d %H:%M",
            na_rep="",
            compression="gzip" if args.gzip else None,
        )
        os.replace(part_path, task.final_path)

        return TaskResult(
            task=task,
            rows=len(dataframe),
            status="ok",
        )

    except Exception as exc:
        part_path.unlink(missing_ok=True)
        return TaskResult(
            task=task,
            rows=0,
            status="failed",
            error=str(exc),
        )


def append_failure(
    failure_path: Path,
    task: Task,
    error: str,
) -> None:
    new_file = not failure_path.exists() or failure_path.stat().st_size == 0
    with failure_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if new_file:
            writer.writerow(
                ["file", "point_index", "point_id", "lat", "lon", "error"]
            )
        writer.writerow(
            [
                task.coord_path.name,
                task.point_index,
                task.point.point_id,
                task.point.lat_text,
                task.point.lon_text,
                error,
            ]
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="直接请求 Open-Meteo API 并逐坐标保存 CSV"
    )
    parser.add_argument("--coords-dir", required=True, help="坐标 CSV 目录")
    parser.add_argument(
        "--pattern", default="draw_*.csv", help="坐标文件匹配模式"
    )
    parser.add_argument("--start", default="2015-07-01", help="开始日期 UTC")
    parser.add_argument("--end", default="2026-10-01", help="结束日期 UTC")
    parser.add_argument("--out", default="./ghi", help="输出根目录")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数，默认 1；建议先用 1 避免限流",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="读取超时秒数"
    )
    parser.add_argument("--retries", type=int, default=5, help="重试次数")
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.5,
        help="所有线程合计的请求最小间隔秒数",
    )
    parser.add_argument(
        "--max-retry-wait",
        type=float,
        default=600,
        help="单次退避最长等待秒数；0 表示不限制",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="本次最多请求数，0 表示不限制",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=20,
        help="连续失败达到该数量时停止；0 表示不限制",
    )
    parser.add_argument("--gzip", action="store_true", help="输出 .csv.gz")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单坐标失败后继续，并写入 _failures.csv",
    )
    parser.add_argument(
        "--restart", action="store_true", help="覆盖已有输出文件"
    )
    parser.add_argument("--api", default=API, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="只估算，不请求")
    args = parser.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        parser.error("--end 不能早于 --start")
    if args.workers <= 0:
        parser.error("--workers 必须大于 0")

    coords_dir = Path(args.coords_dir)
    coord_files = sorted(coords_dir.glob(args.pattern))
    if not coord_files:
        parser.error(f"{coords_dir} 中没有匹配 {args.pattern} 的文件")

    all_points = {path: read_points(path) for path in coord_files}
    total_points = sum(len(points) for points in all_points.values())
    total_days = (end - start).days + 1
    estimated_rows = total_points * total_days * ROWS_PER_DAY
    estimated_gb = estimated_rows * 50 / (1024**3)

    output_root = Path(args.out)
    tasks: list[Task] = []
    skipped = 0

    for coord_path, points in all_points.items():
        draw_dir = output_root / coord_path.stem
        for point_index, point in enumerate(points, start=1):
            final_path = make_output_path(
                draw_dir=draw_dir,
                point_index=point_index,
                point=point,
                use_gzip=args.gzip,
            )
            if final_path.exists() and not args.restart:
                skipped += 1
                continue
            tasks.append(
                Task(
                    coord_path=coord_path,
                    point_index=point_index,
                    point=point,
                    final_path=final_path,
                )
            )

    if args.max_requests:
        tasks = tasks[: args.max_requests]

    print(f"[*] 坐标文件数: {len(coord_files)}")
    print(f"[*] 坐标点总数: {total_points:,}")
    print(f"[*] 已完成并跳过: {skipped:,}")
    print(f"[*] 本次待请求: {len(tasks):,}")
    print(f"[*] 日期范围: {start} 至 {end}（{total_days:,} 天）")
    print("[*] 请求策略: 直接 GET，1 个坐标 = 1 次请求")
    print(f"[*] 并发线程数: {args.workers}")
    print(f"[*] 每个坐标预计数据行数: {total_days * ROWS_PER_DAY:,}")
    print(f"[*] 全部预计数据行数: {estimated_rows:,}")
    print(f"[*] 全部预计 CSV 体积: 约 {estimated_gb:,.0f} GB")

    if args.dry_run:
        return 0

    if not tasks:
        print("[*] 没有需要下载的坐标")
        return 0

    for task in tasks:
        task.final_path.parent.mkdir(parents=True, exist_ok=True)
        if args.restart:
            task.final_path.unlink(missing_ok=True)

    limiter = RateLimiter(args.min_interval)
    stop_event = threading.Event()
    print_lock = threading.Lock()

    def log(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    requests_done = 0
    rows_done = 0
    failed = 0
    consecutive_failures = 0
    started_at = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                download_one,
                task,
                start,
                end,
                args,
                limiter,
                stop_event,
            ): task
            for task in tasks
        }

        try:
            for future in as_completed(future_map):
                result = future.result()

                if result.status == "cancelled":
                    continue

                requests_done += 1
                point = result.task.point

                if result.status == "ok":
                    consecutive_failures = 0
                    rows_done += result.rows
                    elapsed = (time.monotonic() - started_at) / 60
                    log(
                        f"[{requests_done}/{len(tasks)}] "
                        f"{result.task.coord_path.name} "
                        f"{point.point_id} {point.lat_text},{point.lon_text} "
                        f"-> {result.rows:,} 行；累计 {rows_done:,} 行；"
                        f"{elapsed:.1f} 分钟"
                    )
                else:
                    failed += 1
                    consecutive_failures += 1
                    failure_path = (
                        result.task.final_path.parent / "_failures.csv"
                    )
                    append_failure(
                        failure_path,
                        result.task,
                        result.error,
                    )
                    log(
                        f"[{requests_done}/{len(tasks)}] "
                        f"{point.point_id} 失败: {result.error}"
                    )

                    hit_limit = (
                        args.max_consecutive_failures > 0
                        and consecutive_failures
                        >= args.max_consecutive_failures
                    )
                    if not args.continue_on_error or hit_limit:
                        if hit_limit:
                            log(
                                "[stop] 连续失败达到 "
                                f"{args.max_consecutive_failures} 次"
                            )
                        stop_event.set()
                        for pending in future_map:
                            pending.cancel()
                        break
        except KeyboardInterrupt:
            stop_event.set()
            print("\n[stop] 收到中断信号")
            return 130

    elapsed = (time.monotonic() - started_at) / 60
    print(
        f"\n[*] 完成：{requests_done:,} 次请求，"
        f"{rows_done:,} 行，跳过 {skipped:,} 个，失败 {failed:,} 个，"
        f"用时 {elapsed:.1f} 分钟"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())