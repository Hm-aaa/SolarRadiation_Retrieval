#!/usr/bin/env python3
"""Generate globally unique random coordinate samples inside China.

Supports arbitrary STEP values (down to 1e-6 degree) and splits the sampled
points into multiple CSV files.
"""
from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

# Store coordinates as integer microdegrees to avoid floating-point drift.
SCALE = 1_000_000
MAX_DECIMALS = 6


def parse_decimal(value: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"无效数字: {value}") from exc
    if not result.is_finite():
        raise argparse.ArgumentTypeError(f"数字必须是有限值: {value}")
    return result


def to_scaled(value: Decimal) -> int:
    """Convert degrees to integer microdegrees."""
    return int((value * SCALE).to_integral_value())


def decimal_places(value: Decimal) -> int:
    """Number of decimal places after normalization, capped at 6."""
    normalized = value.normalize()
    return min(MAX_DECIMALS, max(0, -normalized.as_tuple().exponent))


def format_deg(value_int: int, decimals: int) -> str:
    """Format an integer microdegree value as a fixed-decimal degree string."""
    sign = "-" if value_int < 0 else ""
    value = abs(int(value_int))
    whole, frac = divmod(value, SCALE)
    if decimals <= 0:
        return f"{sign}{whole}"
    frac_str = f"{frac:0{MAX_DECIMALS}d}"[:decimals]
    return f"{sign}{whole}.{frac_str}"


def build_axis(
    low: Decimal,
    high: Decimal,
    step: Decimal,
    name: str,
) -> np.ndarray:
    """Build an integer-microdegree axis, inclusive of endpoints when possible."""
    low_i = to_scaled(low)
    high_i = to_scaled(high)
    step_i = to_scaled(step)

    if low_i > high_i:
        raise ValueError(f"{name}: 最小值 {low} 大于最大值 {high}")
    if step_i <= 0:
        raise ValueError(f"{name}: STEP={step} 太小，必须 >= 1e-6 度")

    values = np.arange(low_i, high_i + 1, step_i, dtype=np.int64)
    if values.size == 0:
        raise ValueError(f"{name}: 无法生成坐标轴")

    if int(values[-1]) != high_i:
        print(
            f"警告: {name} 范围不能被 STEP={step} 整除；"
            f"最后一个值为 {values[-1] / SCALE:.6f}，最大值为 {high}",
            file=sys.stderr,
        )

    return values


def load_china_boundary(shapefile: Path) -> shapely.Geometry:
    """Read, repair, and union the China boundary polygons."""
    if not shapefile.exists():
        raise FileNotFoundError(f"找不到中国边界文件: {shapefile}")

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise ValueError("中国边界文件缺少坐标系（CRS）")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    polygons = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        fixed = shapely.make_valid(geom)
        for part in shapely.get_parts(fixed):
            if part.geom_type == "Polygon" and not part.is_empty:
                polygons.append(part)

    if not polygons:
        raise ValueError("中国边界文件中没有可用的 Polygon/MultiPolygon 几何")

    boundary = shapely.union_all(polygons)
    if boundary.is_empty:
        raise ValueError("中国边界几何为空")

    if not boundary.is_valid:
        boundary = shapely.make_valid(boundary)

    shapely.prepare(boundary)
    return boundary


def find_inside_indices(
    boundary: shapely.Geometry,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    chunk_points: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_index, lon_index) arrays for grid points inside the boundary."""
    nlon = int(lon_values.size)
    nlat = int(lat_values.size)
    if nlon == 0 or nlat == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rows_per_chunk = max(1, int(chunk_points) // nlon)
    lon_float = lon_values.astype(np.float64) / SCALE

    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []

    total_chunks = (nlat + rows_per_chunk - 1) // rows_per_chunk
    for chunk_no, start in enumerate(range(0, nlat, rows_per_chunk), start=1):
        stop = min(start + rows_per_chunk, nlat)
        lat_chunk = lat_values[start:stop].astype(np.float64) / SCALE
        lon_grid, lat_grid = np.meshgrid(lon_float, lat_chunk)
        mask = shapely.intersects_xy(boundary, lon_grid, lat_grid)
        ii, jj = np.nonzero(mask)

        if ii.size:
            lat_parts.append(ii.astype(np.int64) + start)
            lon_parts.append(jj.astype(np.int64))

        if total_chunks >= 5 and (
            chunk_no % max(1, total_chunks // 10) == 0
            or chunk_no == total_chunks
        ):
            print(
                f"  几何筛选进度: {chunk_no}/{total_chunks} 批",
                file=sys.stderr,
            )

    if not lat_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(lat_parts), np.concatenate(lon_parts)


def write_samples(
    output_dir: Path,
    selected_linear: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    nlon: int,
    draw_count: int,
    points_per_draw: int,
    decimals: int,
    keep_old: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not keep_old:
        old_files = sorted(output_dir.glob("coords_*.csv"))
        for old_file in old_files:
            old_file.unlink()
        if old_files:
            print(f"已删除旧文件: {len(old_files)} 个")

    for draw_no in range(1, draw_count + 1):
        start = (draw_no - 1) * points_per_draw
        stop = start + points_per_draw
        chunk = selected_linear[start:stop]

        lat_indices = chunk // nlon
        lon_indices = chunk % nlon
        lat_ints = lat_values[lat_indices]
        lon_ints = lon_values[lon_indices]

        pairs = set(zip(lat_ints.tolist(), lon_ints.tolist()))
        if len(pairs) != points_per_draw:
            raise RuntimeError(f"第 {draw_no} 个文件中出现重复坐标")

        output_path = output_dir / f"coords_{draw_no:02d}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["lat", "lon"])
            writer.writerows(
                (
                    format_deg(lat_int, decimals),
                    format_deg(lon_int, decimals),
                )
                for lat_int, lon_int in zip(lat_ints, lon_ints)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在中国边界内按任意 STEP 抽取全局不重复的经纬度点。"
    )
    parser.add_argument(
        "--step",
        type=parse_decimal,
        default=Decimal("0.05"),
        help="网格步长（度），例如 0.05、0.1、0.25；默认 0.05",
    )
    parser.add_argument(
        "--draw-count",
        type=int,
        default=10,
        help="输出文件数量；默认 10",
    )
    parser.add_argument(
        "--points-per-draw",
        type=int,
        default=10_000,
        help="每个 CSV 的坐标点数；默认 10000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261019,
        help="随机种子；默认 20261019",
    )
    parser.add_argument(
        "--lat-min",
        type=parse_decimal,
        default=Decimal("15.0"),
    )
    parser.add_argument(
        "--lat-max",
        type=parse_decimal,
        default=Decimal("55.0"),
    )
    parser.add_argument(
        "--lon-min",
        type=parse_decimal,
        default=Decimal("70.0"),
    )
    parser.add_argument(
        "--lon-max",
        type=parse_decimal,
        default=Decimal("140.0"),
    )
    parser.add_argument(
        "--china-shp",
        type=Path,
        default=Path(
            "/home/jinhm/PhD_Project/SolarRadiation_Retrieval/"
            "data/china_shp/China.shp"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/coords"
        ),
    )
    parser.add_argument(
        "--chunk-points",
        type=int,
        default=2_000_000,
        help="内部几何判断每次处理的网格点数，控制内存；默认 200 万",
    )
    parser.add_argument(
        "--max-grid-points",
        type=int,
        default=50_000_000,
        help="矩形网格点数安全上限，防止 STEP 过小；默认 5000 万",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="保留输出目录中已有的 coords_*.csv（默认会删除）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计网格点数量，不抽样、不写文件",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.draw_count <= 0:
        parser.error("--draw-count 必须大于 0")
    if args.points_per_draw <= 0:
        parser.error("--points-per-draw 必须大于 0")
    if args.step <= 0:
        parser.error("--step 必须大于 0")
    if args.chunk_points <= 0:
        parser.error("--chunk-points 必须大于 0")
    if args.max_grid_points <= 0:
        parser.error("--max-grid-points 必须大于 0")

    total_needed = args.draw_count * args.points_per_draw

    print("读取中国边界...")
    boundary = load_china_boundary(args.china_shp)
    print(f"中国边界范围 (minx, miny, maxx, maxy): {boundary.bounds}")

    print(f"生成网格轴: STEP={args.step}°")
    lon_values = build_axis(args.lon_min, args.lon_max, args.step, "经度")
    lat_values = build_axis(args.lat_min, args.lat_max, args.step, "纬度")
    nlon = int(lon_values.size)
    nlat = int(lat_values.size)

    print(
        f"网格范围: 经度 {lon_values[0] / SCALE:.6f}–"
        f"{lon_values[-1] / SCALE:.6f}, "
        f"纬度 {lat_values[0] / SCALE:.6f}–"
        f"{lat_values[-1] / SCALE:.6f}"
    )

    rectangle_points = nlat * nlon
    print(f"矩形网格点数: {nlat:,} × {nlon:,} = {rectangle_points:,}")

    if rectangle_points > args.max_grid_points:
        raise SystemExit(
            f"STEP={args.step}° 生成的矩形网格有 {rectangle_points:,} 个点，"
            f"超过安全上限 {args.max_grid_points:,}。\n"
            "请增大 --step，或显式调大 --max-grid-points。"
        )

    print("筛选中国边界内网格点...")
    valid_lat_idx, valid_lon_idx = find_inside_indices(
        boundary,
        lon_values,
        lat_values,
        chunk_points=args.chunk_points,
    )
    valid_count = int(valid_lat_idx.size)

    if valid_count == 0:
        raise SystemExit(
            "中国区域内有效网格点为 0。\n"
            f"中国边界范围: {boundary.bounds}\n"
            f"网格范围: 经度 {lon_values[0] / SCALE:.6f}–"
            f"{lon_values[-1] / SCALE:.6f}, "
            f"纬度 {lat_values[0] / SCALE:.6f}–"
            f"{lat_values[-1] / SCALE:.6f}\n"
            "请检查 STEP、经纬度范围和中国边界文件的 CRS。"
        )

    print(f"中国区域内有效网格点: {valid_count:,}")

    if args.dry_run:
        max_full_draws = valid_count // args.points_per_draw
        print(
            f"DRY RUN: 当前 --points-per-draw="
            f"{args.points_per_draw:,}，最多可生成 "
            f"{max_full_draws} 个完整文件"
        )
        return

    if total_needed > valid_count:
        max_full_draws = valid_count // args.points_per_draw
        raise SystemExit(
            f"需要 {total_needed:,} 个点，但中国区域内只有 "
            f"{valid_count:,} 个。\n"
            f"当前 --points-per-draw={args.points_per_draw:,} 时，"
            f"最多支持 {max_full_draws} 个完整文件。\n"
            "请减小 --points-per-draw / --draw-count，"
            "或减小 --step。"
        )

    # Unique linear index for every inside-grid point.
    valid_linear = valid_lat_idx * nlon + valid_lon_idx
    rng = np.random.default_rng(args.seed)
    selected_linear = rng.choice(
        valid_linear,
        size=total_needed,
        replace=False,
    )

    if np.unique(selected_linear).size != total_needed:
        raise RuntimeError("抽样结果中存在重复坐标")

    decimals = decimal_places(args.step)
    write_samples(
        output_dir=args.output_dir,
        selected_linear=selected_linear,
        lat_values=lat_values,
        lon_values=lon_values,
        nlon=nlon,
        draw_count=args.draw_count,
        points_per_draw=args.points_per_draw,
        decimals=decimals,
        keep_old=args.keep_old,
    )

    print("\n生成完成")
    print(f"STEP: {args.step}°")
    print(f"网格: {nlat:,} lat × {nlon:,} lon = {nlat * nlon:,}")
    print(f"中国边界内网格点: {valid_count:,}")
    print(f"抽样文件数: {args.draw_count}")
    print(f"每文件行数: {args.points_per_draw:,}（不含表头）")
    print(f"全局唯一坐标对: {total_needed:,}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

    # 如果要用 0.25°，先 dry-run 看有效点数：
    # python3 generate_coordinate_samples.py --step 0.25 --dry-run

    # 0.25° 下中国区域大约只有 1.5 万个网格点，通常需要把总数降到 1.5 万以内，例如：
    # python3 generate_coordinate_samples.py --step 0.05 --draw-count 38 --points-per-draw 10000