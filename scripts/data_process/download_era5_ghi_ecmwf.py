#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 逐年下载：2000-2025
变量：2m_temperature + surface_solar_radiation_downwards
区域：70-140E, 15-55N
输出：/home/jinhm/data/solar/solar_YYYY.nc
"""

import os
import sys
import time
import cdsapi

OUT_DIR = "/home/jinhm/data/solar"
DATASET = "reanalysis-era5-single-levels"
AREA = [55, 70, 15, 140]          # [N, W, S, E]

VARIABLES = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
]

START_YEAR = 2010
END_YEAR   = 2025

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
TIMES  = [f"{h:02d}:00" for h in range(24)]

MAX_RETRY  = 5
RETRY_WAIT = 120


def build_request(year):
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def download_year(client, year):
    target = os.path.join(OUT_DIR, f"solar_{year}.nc")
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"[skip] {target} 已存在", flush=True)
        return

    tmp = target + ".part"
    print(f"[{year}] 提交请求 ...", flush=True)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            client.retrieve(DATASET, build_request(year), tmp)
            os.replace(tmp, target)          # 下载完整后再改名
            print(f"[{year}] 完成 -> {target} "
                  f"({os.path.getsize(target)/1e9:.2f} GB)", flush=True)
            return
        except Exception as e:
            print(f"[{year}] 第 {attempt} 次失败: {e}",
                  file=sys.stderr, flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == MAX_RETRY:
                raise
            time.sleep(RETRY_WAIT * attempt)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = cdsapi.Client()

    years = range(START_YEAR, END_YEAR + 1)
    for i, year in enumerate(years, 1):
        print(f"\n===== [{i}/{len(years)}] {year} =====", flush=True)
        t0 = time.time()
        download_year(client, year)
        print(f"[{year}] 用时 {time.time()-t0:.0f} 秒", flush=True)

    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()