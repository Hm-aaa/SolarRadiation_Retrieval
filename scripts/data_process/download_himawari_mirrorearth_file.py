#!/usr/bin/env python3
"""按 biz_id 批量下载 Mirror Earth Himawari 短波辐射数据。"""

import argparse
import csv
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.mirror-earth.com/v1/archive"
DEFAULT_INPUT_CSV = Path(
    "/home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/system_id_information.csv"
)
DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-10-01"
DEFAULT_TIMEZONE = "UTC"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def parse_args():
    parser = argparse.ArgumentParser(
        description="按 biz_id 下载 Mirror Earth 短波辐射数据，并分别保存为 CSV"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="包含 biz_id、longitude、latitude 的 CSV 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认是输入 CSV 同级的 himawari 目录",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="开始日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="结束日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="API 时区，默认 UTC",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MIRROR_EARTH_API_KEY", ""),
        help="API Key；建议使用环境变量 MIRROR_EARTH_API_KEY",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每次请求后的等待秒数",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="请求失败时的最大尝试次数",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="单次请求超时秒数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只下载前 N 个站点，用于测试",
    )
    return parser.parse_args()


def safe_filename(value):
    """将 biz_id 转换成安全文件名。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def load_stations(input_csv):
    """读取并校验站点信息。"""
    stations = []
    seen_biz_ids = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required_fields = {"biz_id", "longitude", "latitude"}
        missing_fields = required_fields - fieldnames

        if missing_fields:
            raise ValueError(
                f"输入 CSV 缺少字段：{', '.join(sorted(missing_fields))}"
            )

        for row_number, row in enumerate(reader, start=2):
            biz_id = str(row.get("biz_id", "")).strip()
            longitude_text = str(row.get("longitude", "")).strip()
            latitude_text = str(row.get("latitude", "")).strip()

            if not biz_id or not longitude_text or not latitude_text:
                print(f"跳过第 {row_number} 行：biz_id 或坐标为空")
                continue

            if biz_id in seen_biz_ids:
                print(f"跳过重复 biz_id={biz_id}")
                continue

            try:
                longitude = float(longitude_text)
                latitude = float(latitude_text)
            except ValueError:
                print(f"跳过第 {row_number} 行：经纬度格式错误")
                continue

            if not -90 <= latitude <= 90:
                raise ValueError(
                    f"第 {row_number} 行纬度超出范围：{latitude}"
                    "，请检查经纬度列是否传反"
                )

            if not -180 <= longitude <= 180:
                raise ValueError(
                    f"第 {row_number} 行经度超出范围：{longitude}"
                )

            seen_biz_ids.add(biz_id)
            stations.append(
                {
                    "biz_id": biz_id,
                    "longitude": longitude,
                    "latitude": latitude,
                }
            )

    return stations


def fetch_payload(params, retries, timeout):
    """请求 API，并对临时错误进行重试。"""
    url = f"{API_URL}?{urlencode(params)}"

    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "mirror-earth-csv-downloader/1.0",
            },
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset))

        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]

            if error.code not in RETRYABLE_HTTP_CODES or attempt == retries:
                raise RuntimeError(
                    f"HTTP {error.code}: {body or error.reason}"
                ) from error

            wait_seconds = min(2 ** attempt, 30)

        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries:
                raise RuntimeError(f"请求失败：{error}") from error

            wait_seconds = min(2 ** attempt, 30)

        print(
            f"  请求失败，{wait_seconds} 秒后重试 "
            f"({attempt}/{retries})"
        )
        time.sleep(wait_seconds)

    raise RuntimeError("API 请求失败")


def extract_rows(payload, station, timezone):
    """将 API JSON 转换为 CSV 行。"""
    if payload.get("error"):
        raise ValueError(
            f"API 返回错误：{payload.get('reason', payload)}"
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(
            "API 返回中缺少 hourly 字段："
            f"{json.dumps(payload, ensure_ascii=False)[:500]}"
        )

    times = hourly.get("time")
    radiation_values = hourly.get("shortwave_radiation")

    if not isinstance(times, list) or not isinstance(radiation_values, list):
        raise ValueError(
            "API 返回中缺少 hourly.time 或 "
            "hourly.shortwave_radiation"
        )

    if len(times) != len(radiation_values):
        raise ValueError(
            "时间和辐射数据长度不一致："
            f"{len(times)} != {len(radiation_values)}"
        )

    return [
        {
            "biz_id": station["biz_id"],
            "time": timestamp,
            "shortwave_radiation": value,
            "longitude": station["longitude"],
            "latitude": station["latitude"],
            "timezone": timezone,
        }
        for timestamp, value in zip(times, radiation_values)
    ]


def save_csv(output_path, rows):
    """先写临时文件，再原子替换，避免留下半截 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.tmp")

    fieldnames = [
        "biz_id",
        "time",
        "shortwave_radiation",
        "longitude",
        "latitude",
        "timezone",
    ]

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_failures(output_dir, failures):
    """保存失败站点列表。"""
    if not failures:
        return

    failure_path = output_dir / "download_failures.csv"

    with failure_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["biz_id", "longitude", "latitude", "error"],
        )
        writer.writeheader()
        writer.writerows(failures)

    print(f"\n失败站点详情：{failure_path}")


def main():
    args = parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        raise SystemExit(
            "缺少 API Key。请设置环境变量 MIRROR_EARTH_API_KEY，"
            "或通过 --api-key 传入。"
        )

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if end_date < start_date:
        raise ValueError("结束日期不能早于开始日期")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_csv.parent / "himawari"

    stations = load_stations(args.input_csv)
    if args.limit is not None:
        stations = stations[: max(args.limit, 0)]

    if not stations:
        raise SystemExit("没有可下载的站点")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    downloaded = 0
    skipped = 0

    for index, station in enumerate(stations, start=1):
        biz_id = station["biz_id"]
        output_path = output_dir / f"{safe_filename(biz_id)}.csv"

        if output_path.exists() and not args.overwrite:
            print(
                f"[{index}/{len(stations)}] 已存在，跳过："
                f"{output_path}"
            )
            skipped += 1
            continue

        print(
            f"[{index}/{len(stations)}] 下载 biz_id={biz_id} "
            f"(longitude={station['longitude']}, "
            f"latitude={station['latitude']})"
        )

        params = {
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "hourly": "shortwave_radiation",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "models": "himawari",
            "temporal_resolution": "minutely_10",
            "timezone": args.timezone,
            "apikey": api_key,
        }

        try:
            payload = fetch_payload(
                params=params,
                retries=args.retries,
                timeout=args.timeout,
            )
            rows = extract_rows(payload, station, args.timezone)
            save_csv(output_path, rows)
            downloaded += 1
            print(f"  完成：{len(rows)} 行 -> {output_path}")

        except Exception as error:
            print(f"  失败：{error}")
            failures.append(
                {
                    "biz_id": biz_id,
                    "longitude": station["longitude"],
                    "latitude": station["latitude"],
                    "error": str(error),
                }
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_failures(output_dir, failures)

    print("\n下载结束")
    print(f"成功：{downloaded}")
    print(f"跳过：{skipped}")
    print(f"失败：{len(failures)}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
    # export MIRROR_EARTH_API_KEY='FCGC1mFso15vDAX1GSKYtlEYaY1UKC3T'
    # python3 /home/jinhm/PhD_Project/SolarRadiation_Retrieval/scripts/data_process/download_himawari_mirrorearth_file.py \
    #     --input-csv '/home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/system_id_information.csv' \
    #     --output-dir '/home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/himawari' \
    #     --start 2025-01-01 \
    #     --end 2025-12-31
