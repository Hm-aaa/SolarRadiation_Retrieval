import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import xarray as xr
import pvlib

def read_dem_elevation(
    dem_file: str | Path,
    latitude,
    longitude,
    method: str = "linear",
) -> float | np.ndarray:
    """
    根据经纬度读取 DEM 高程。

    Parameters
    ----------
    dem_file:
        ETOPO1 NetCDF 文件路径
    latitude:
        纬度，支持标量或数组
    longitude:
        经度，支持标量或数组
    method:
        "linear" 或 "nearest"

    Returns
    -------
    elevation:
        高程，单位通常为米
    """
    lat_arr = np.asarray(latitude, dtype=float)
    lon_arr = np.asarray(longitude, dtype=float)

    if lat_arr.shape != lon_arr.shape:
        raise ValueError("latitude 和 longitude 的形状必须一致")

    scalar_input = lat_arr.ndim == 0
    original_shape = lat_arr.shape

    with xr.open_dataset(dem_file) as ds:
        # 自动识别高程变量
        elevation_name = None
        for name in ("z", "elevation", "topo", "height", "Band1"):
            if name in ds.data_vars:
                elevation_name = name
                break

        if elevation_name is None:
            raise ValueError(
                f"未识别到高程变量，当前变量为: {list(ds.data_vars)}"
            )

        # 自动识别经纬度坐标
        lat_name = None
        for name in ("lat", "latitude", "y"):
            if name in ds.coords or name in ds.variables:
                lat_name = name
                break

        lon_name = None
        for name in ("lon", "longitude", "x"):
            if name in ds.coords or name in ds.variables:
                lon_name = name
                break

        if lat_name is None or lon_name is None:
            raise ValueError("未识别到经纬度坐标")

        elevation = ds[elevation_name].squeeze(drop=True)

        lat_values = np.asarray(ds[lat_name].values, dtype=float)
        lon_values = np.asarray(ds[lon_name].values, dtype=float)

        if lat_values.ndim != 1 or lon_values.ndim != 1:
            raise ValueError(
                "当前 DEM 使用的是二维经纬度坐标，需要额外处理投影坐标"
            )

        # 保证纬度、经度按升序排列
        if lat_values[0] > lat_values[-1]:
            elevation = elevation.isel(
                {lat_name: slice(None, None, -1)}
            )
            lat_values = lat_values[::-1]

        if lon_values[0] > lon_values[-1]:
            elevation = elevation.isel(
                {lon_name: slice(None, None, -1)}
            )
            lon_values = lon_values[::-1]

        # 处理经度范围：0~360 或 -180~180
        lon_span = lon_values.max() - lon_values.min()

        if lon_span > 350:
            if lon_values.min() >= 0:
                lon_arr = np.mod(lon_arr, 360.0)
            else:
                lon_arr = (lon_arr + 180.0) % 360.0 - 180.0

        lat_flat = lat_arr.ravel()
        lon_flat = lon_arr.ravel()

        result = elevation.interp(
            {
                lat_name: xr.DataArray(lat_flat, dims="points"),
                lon_name: xr.DataArray(lon_flat, dims="points"),
            },
            method=method,
        ).values

    if scalar_input:
        return float(result[0])

    return result.reshape(original_shape)

def calculate_solar_angles(
    time,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算太阳天顶角和方位角。

    Parameters
    ----------
    time:
        单个时间、时间列表、pandas Series 或 DatetimeIndex。
        不带时区的时间按 UTC 处理。
    latitude:
        纬度，单位：度。
    longitude:
        经度，单位：度。
    elevation_m:
        海拔高程，单位：米。DEM 高程可直接传入。

    Returns
    -------
    zenith_deg:
        太阳天顶角，单位：度。
    azimuth_deg:
        太阳方位角，单位：度。
        0=N, 90=E, 180=S, 270=W。
    """
    time_utc = pd.DatetimeIndex(
        pd.to_datetime(time, utc=True, errors="raise")
    )

    latitude = float(latitude)
    longitude = (float(longitude) + 180.0) % 360.0 - 180.0
    elevation_m = float(elevation_m)

    solar = pvlib.solarposition.spa_python(
        time_utc,
        latitude=latitude,
        longitude=longitude,
        altitude=elevation_m,
    )

    zenith_deg = solar["zenith"].to_numpy(dtype=float)
    azimuth_deg = solar["azimuth"].to_numpy(dtype=float) % 360.0

    return zenith_deg, azimuth_deg

def calculate_time_features(time):
    times = pd.DatetimeIndex(
        pd.to_datetime(time, utc=True, errors="raise")
    )

    minute_of_day = (
        np.asarray(times.hour, dtype=float) * 60.0
        + np.asarray(times.minute, dtype=float)
        + np.asarray(times.second, dtype=float) / 60.0
        + np.asarray(times.microsecond, dtype=float) / 60_000_000.0
    )

    day_of_year = np.asarray(times.dayofyear, dtype=np.int16)

    return minute_of_day, day_of_year

# ===========获取项目的运行路径
project_path = os.getenv("PROJECT_PATH")
if not project_path:
    raise RuntimeError(
        "环境变量 PROJECT_PATH 未设置。请先执行：\n"
        "例如: export PROJECT_PATH=/home/jinhm/PhD_Project/SolarRadiation_Retrieval"
    )


#============获取数字高程模型数据路径
PROJECT_PATH = Path(project_path)
DEM_FILE = PROJECT_PATH / "data" / "DEM" / "etopo1.nc"

# 需要计算的坐标文件集合
Coords = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04'

# 便利处理每个坐标文件
for file_coords_item in glob.glob(f'{Coords}/*'):
    # 读取坐标文件
    _, lat, lon = file_coords_item.split('/')[-1][:-4].split('_')
    elevation = read_dem_elevation(DEM_FILE, lat, lon, method="nearest")

    # 计算坐标时间点内的太阳几何参数
    data = pd.read_csv(file_coords_item)
    zenith_deg, azimuth_deg = calculate_solar_angles(time=data["TIME_UTC"], latitude=lat, longitude=lon, elevation_m=elevation)


    # 计算时间特征
    minute_of_day, day_of_year = calculate_time_features(data["TIME_UTC"])

    # 将新生成的变量加入 data
    data["latitude"] = lat
    data["longitude"] = lon
    data["elevation_m"] = float(elevation)

    data["solar_zenith_deg"] = zenith_deg
    data["solar_azimuth_deg"] = azimuth_deg

    data["minute_of_day"] = minute_of_day
    data["day_of_year"] = day_of_year

    data.to_csv(
        file_coords_item,
        index=False,
        encoding="utf-8-sig",
    )
    print(file_coords_item)
