#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 JAXA P-Tree Himawari-8/9 L2 PAR (RFL021) 数据

目录:
    /pub/himawari/L2/PAR/021/YYYYMM/DD/HH/

文件名:
    H09_20251206_0210_RFL021_FLDK.02401_02401.nc
    H09_20250901_0010_RFL021_FLDK.02801_02401.nc

输出:
    {out}/YYYY/MM/DD/H09_YYYYMMDD_HHMM_RFL021_FLDK.NNNNN_NNNNN.nc
    {out}/manifest.csv

用法:
    export PTREE_USER='你的邮箱'
    export PTREE_PASS='你的密码'

    python download_himawari_par.py \
        --start "2020-01-01 00:00" --end "2020-01-01 23:59" \
        --out /home/jinhm/data/himawari/par
"""

import argparse
import csv
import ftplib
import os
import re
import sys
import time
from datetime import datetime, timedelta

HOST = "ftp.ptree.jaxa.jp"
PORT = 21
BASE = "/pub/himawari/L2/PAR/021"

DIR_TEMPLATE = "{base}/{ym}/{dd}/{hh}"

# 注意：区分大小写，避免匹配到 rFL021（另一套产品）
FNAME_RE = re.compile(
    r"^(?P<sat>H(?P<num>\d{2}))_"
    r"(?P<datestr>\d{8})_"
    r"(?P<hhmm>\d{4})_"
    r"RFL021_FLDK\.(?P<ny>\d{5})_(?P<nx>\d{5})"
    r"\.nc$"
)


def parse_dt(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"无法解析时间: {s}")


class FtpSession:
    """带自动重连的 FTP 会话"""

    def __init__(self, user, password, tls=False, timeout=300, max_retry=5):
        self.user = user
        self.password = password
        self.tls = tls
        self.timeout = timeout
        self.max_retry = max_retry
        self.ftp = None

    def connect(self):
        self.close()

        if self.tls:
            ftp = ftplib.FTP_TLS(timeout=self.timeout)
            ftp.connect(HOST, PORT)
            ftp.login(self.user, self.password)
            ftp.prot_p()
        else:
            ftp = ftplib.FTP(timeout=self.timeout)
            ftp.connect(HOST, PORT)
            ftp.login(self.user, self.password)

        ftp.set_pasv(True)
        ftp.voidcmd("TYPE I")
        if ftp.sock is not None:
            ftp.sock.settimeout(self.timeout)

        self.ftp = ftp
        return ftp

    def close(self):
        if self.ftp is not None:
            try:
                self.ftp.quit()
            except Exception:
                try:
                    self.ftp.close()
                except Exception:
                    pass
            self.ftp = None

    def ensure(self):
        if self.ftp is None:
            return self.connect()
        return self.ftp

    def call(self, func, *args, **kwargs):
        """执行操作；网络类异常自动重连重试"""
        last = None

        for attempt in range(1, self.max_retry + 1):
            try:
                return func(self.ensure(), *args, **kwargs)

            except ftplib.error_perm:
                # 550 等业务错误，不重连
                raise

            except (OSError, EOFError, ftplib.error_temp,
                    ftplib.error_reply) as exc:
                last = exc
                print(f"    [reconnect {attempt}/{self.max_retry}] {exc}",
                      file=sys.stderr, flush=True)
                self.close()
                time.sleep(min(60, 2 * attempt))

        raise last


def list_hour(ftp: ftplib.FTP, t: datetime):
    path = DIR_TEMPLATE.format(
        base=BASE,
        ym=f"{t.year:04d}{t.month:02d}",
        dd=f"{t.day:02d}",
        hh=f"{t.hour:02d}",
    )

    ftp.cwd(path)
    names = [os.path.basename(x) for x in ftp.nlst()]
    return path, names


def filter_files(names, hour_dt, start, end, sat_mode, grid_ny):
    chosen = {}

    for name in names:
        m = FNAME_RE.match(name)
        if not m:
            continue

        hh = int(m.group("hhmm")[:2])
        mi = int(m.group("hhmm")[2:])
        ts = hour_dt.replace(hour=hh, minute=mi, second=0, microsecond=0)

        if ts < start or ts > end:
            continue

        sat = f"H{m.group('num')}"
        if sat_mode != "any" and sat != sat_mode:
            continue

        ny = int(m.group("ny"))
        nx = int(m.group("nx"))
        if grid_ny is not None and ny != grid_ny:
            continue

        meta = {"sat": sat, "ny": ny, "nx": nx}
        prev = chosen.get(ts)
        if prev is None or (prev[1]["sat"] == "H08" and sat == "H09"):
            chosen[ts] = (name, meta)

    return sorted(chosen.items(), key=lambda kv: kv[0])


def download_one(session: FtpSession, remote: str, local: str,
                 max_retry: int = 5):
    """断点续传 + 自动重连"""
    os.makedirs(os.path.dirname(local), exist_ok=True)

    try:
        remote_size = session.call(lambda ftp: ftp.size(remote))
    except Exception:
        remote_size = None

    if os.path.exists(local) and os.path.getsize(local) > 0:
        if remote_size is None or os.path.getsize(local) == remote_size:
            return "skip"

    tmp = local + ".part"

    for attempt in range(1, max_retry + 1):
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0

        def _retr(ftp):
            ftp.voidcmd("TYPE I")
            with open(tmp, "ab" if offset else "wb") as f:
                ftp.retrbinary(
                    f"RETR {remote}",
                    f.write,
                    blocksize=1024 * 1024,
                    rest=offset or None,
                )

        try:
            session.call(_retr)

            if remote_size is not None and os.path.getsize(tmp) != remote_size:
                raise IOError(
                    f"大小不一致: {os.path.getsize(tmp)} != {remote_size}"
                )

            os.replace(tmp, local)
            return "ok"

        except ftplib.error_perm as exc:
            print(f"    [retry {attempt}/{max_retry}] {exc}",
                  file=sys.stderr, flush=True)
            # 可能不支持 REST，或文件已不存在，清空重来
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == max_retry:
                raise
            time.sleep(3)

        except Exception as exc:
            print(f"    [retry {attempt}/{max_retry}] {exc}",
                  file=sys.stderr, flush=True)
            if attempt == max_retry:
                raise
            time.sleep(5 * attempt)


def main():
    p = argparse.ArgumentParser(description="下载 Himawari L2 PAR (RFL021)")
    p.add_argument("--start", required=True, type=parse_dt)
    p.add_argument("--end", required=True, type=parse_dt)
    p.add_argument("--out", default="./himawari_par")
    p.add_argument("--sat", default="any", choices=["any", "H08", "H09"])
    p.add_argument("--grid", type=int, default=None,
                   help="按网格行数过滤，如 2401 / 2801；默认不过滤")
    p.add_argument("--timeout", type=int, default=300,
                   help="socket 超时秒数，默认 300")
    p.add_argument("--user", default=os.getenv("PTREE_USER"))
    p.add_argument("--password", default=os.getenv("PTREE_PASS"))
    p.add_argument("--tls", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.user or not args.password:
        p.error("请设置 PTREE_USER / PTREE_PASS，或传 --user/--password")
    if args.end < args.start:
        p.error("--end 早于 --start")

    print(f"连接 {HOST} ...", flush=True)
    session = FtpSession(
        args.user, args.password, tls=args.tls, timeout=args.timeout
    )
    session.connect()
    print("登录成功\n", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")
    new_manifest = not os.path.exists(manifest_path)

    manifest = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    if new_manifest:
        writer.writerow([
            "datetime", "satellite", "ny", "nx",
            "filename", "size_bytes", "path",
        ])

    hour = args.start.replace(minute=0, second=0, microsecond=0)
    last = args.end.replace(minute=0, second=0, microsecond=0)

    n_ok = n_skip = n_missing = n_fail = 0

    try:
        while hour <= last:
            print(f"== {hour:%Y-%m-%d %H}:00 ==", flush=True)

            try:
                dirpath, names = session.call(list_hour, hour)
            except ftplib.error_perm as exc:
                print(f"  目录不存在: {exc}", file=sys.stderr, flush=True)
                n_missing += 1
                hour += timedelta(hours=1)
                continue
            except Exception as exc:
                print(f"  列出目录失败: {exc}", file=sys.stderr, flush=True)
                n_fail += 1
                hour += timedelta(hours=1)
                continue

            hits = filter_files(
                names, hour, args.start, args.end, args.sat, args.grid
            )
            print(f"  {dirpath} -> {len(hits)} 个文件", flush=True)

            if not hits:
                n_missing += 1

            for ts, (name, meta) in hits:
                remote = f"{dirpath}/{name}"
                local = os.path.join(
                    args.out, f"{ts:%Y}", f"{ts:%m}", f"{ts:%d}", name
                )

                if args.dry_run:
                    print(f"  [list] {name}")
                    continue

                print(f"  [{ts:%H:%M}] {name}", flush=True)

                try:
                    status = download_one(session, remote, local)
                except Exception as exc:
                    print(f"    下载失败: {exc}", file=sys.stderr, flush=True)
                    n_fail += 1
                    continue

                if status == "skip":
                    n_skip += 1
                    print("    skip（已存在）", flush=True)
                else:
                    n_ok += 1
                    size = os.path.getsize(local)
                    print(f"    ok（{size / 1e6:.1f} MB）", flush=True)
                    writer.writerow([
                        ts.strftime("%Y-%m-%d %H:%M"),
                        meta["sat"], meta["ny"], meta["nx"],
                        name, size, local,
                    ])
                    manifest.flush()

            print(flush=True)
            hour += timedelta(hours=1)

    finally:
        manifest.close()
        session.close()

    if args.dry_run:
        print("dry-run 结束。")
    else:
        print(f"完成：新下载 {n_ok}，跳过 {n_skip}，"
              f"空/缺失 {n_missing} 小时，失败 {n_fail}。")
        print(f"清单：{manifest_path}")


if __name__ == "__main__":
    main()