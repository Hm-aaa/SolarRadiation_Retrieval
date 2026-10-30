from pathlib import Path
import torch
from src.climate_gridder.smacnp import SMACNP_ALL
from src.climate_gridder.dataset import SMACNPDataset
import yaml 
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import xarray as xr
import pvlib

def regression_metrics(pred, target, eps=1e-12):
    """计算 RMSE、MAE 和 Pearson correlation。"""
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    error = pred - target

    rmse = torch.sqrt(torch.mean(error ** 2)).item()
    mae = torch.mean(torch.abs(error)).item()

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2)
        * torch.sum(target_centered ** 2)
    )

    if denominator.item() < eps:
        r = float("nan")
    else:
        r = (
            torch.sum(pred_centered * target_centered)
            / denominator
        ).item()

    return rmse, mae, r

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

def make_utc_time_df(start: str, end: str, freq: str = "1h") -> pd.DataFrame:
    start_utc = pd.Timestamp(start, tz="UTC")
    end_utc = pd.Timestamp(end, tz="UTC")

    if end_utc < start_utc:
        raise ValueError("结束时间不能早于开始时间")

    time_range = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq=freq,
        inclusive="both",
    )

    df = pd.DataFrame({
        "utc_time": time_range,
        "date_time": time_range.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return df

def bulid_feature(lat,lon,start,end):
    # ===========获取项目的运行路径
    project_path = '/home/jinhm/PhD_Project/SolarRadiation_Retrieval'

    #============获取数字高程模型数据路径
    PROJECT_PATH = Path(project_path)
    DEM_FILE = PROJECT_PATH / "data" / "DEM" / "etopo1.nc"

    elevation = read_dem_elevation(DEM_FILE, lat, lon, method="nearest")

    zenith_deg, azimuth_deg = calculate_solar_angles(time=make_utc_time_df(start,end)['utc_time'], latitude=lat, longitude=lon, elevation_m=elevation)

    # 计算时间特征
    minute_of_day, day_of_year = calculate_time_features(make_utc_time_df(start,end)['utc_time'])

    # 1 经纬度 进行归一化操作
    latitude_norm = (lat- 15) / (55 - 15)
    longitude_norm = (lon - 70) / (140 - 70)
    latitude_norm = np.full( (len(minute_of_day), 1, 1), latitude_norm, dtype=np.float32)
    longitude_norm = np.full( (len(minute_of_day), 1, 1), longitude_norm, dtype=np.float32)

    loaded = np.load(f"/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04_2026/coords_04_test.npz",allow_pickle=False)
    mu = loaded["elevation_mu"]
    sigma = loaded["elevation_sigma"]

    if sigma == 0:
        elevation_norm = elevation / 1000.0
    else:
        elevation_norm = (elevation - mu) / sigma

    elevation_norm = np.full( (len(minute_of_day), 1, 1), elevation_norm, dtype=np.float32)


    # 3 太阳几何 归一化
    zenith_rad = np.deg2rad(zenith_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # 太阳方向单位向量
    zenith_cos_norm = np.cos(zenith_rad)[:, None, None]

    # 方位角周期编码
    azimuth_sin_norm = np.sin(azimuth_rad)[:, None, None]
    azimuth_cos_norm = np.cos(azimuth_rad)[:, None, None]

    # 4 时间特征编码

    # 一天周期
    minute_phase = (
        2.0 * np.pi * minute_of_day / 1440.0
    )

    # 一年周期
    day_phase = (
        2.0 * np.pi * (day_of_year - 1.0) / 365.0
    )

    minute_sin = np.sin(minute_phase)[:, None, None]
    minute_cos = np.cos(minute_phase)[:, None, None]

    day_sin = np.sin(day_phase)[:, None, None]
    day_cos = np.cos(day_phase)[:, None, None]

    x_target = np.concatenate([
        latitude_norm,
        longitude_norm,
        elevation_norm,
        zenith_cos_norm,
        azimuth_sin_norm,
        azimuth_cos_norm,
        minute_sin,
        minute_cos,
        day_sin,
        day_cos,
    ], axis=-1)

    x_target = torch.as_tensor(
    x_target,
    dtype=torch.float32)

    return x_target, make_utc_time_df(start,end)['utc_time']


lat = 22.387555
lon = 113.523709

CONFIG_FILE = Path('/home/jinhm/PhD_Project/SolarRadiation_Retrieval/configs/config.yaml') 
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
device = torch.device(model_cfg['device'])
device = torch.device('cuda:0')

test_file = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04_2026/coords_04_test.npz'
train_dataset = SMACNPDataset(test_file, 0.5, 42)
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = SMACNP_ALL(
        input_dim=model_cfg['input_dim'],
        spatial_dim=model_cfg['spatial_dim'],
        attr_dim=model_cfg['attr_dim'],
        y_dim=model_cfg['y_dim'],
        r_dim=model_cfg['r_dim'], # 属性
        w_dim=model_cfg['w_dim'], # 坐标
        v_dim=model_cfg['v_dim'], # 方差的，我取消了
        hidden_size=model_cfg['hidden_size'], # MLP 的维度
        num_heads=model_cfg['num_heads'],
        dropout_rate=model_cfg['dropout_rate'],
        spatial_indices=model_cfg['spatial_indices'],
        use_hybrid_decoder=model_cfg['use_hybrid_decoder'],
        use_adaptive_knn=model_cfg['use_adaptive_knn'],
        use_positional_encoding=model_cfg['use_positional_encoding']
)
checkpoint = torch.load(
    model_cfg['checkpoints'],
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()

data = []

loaded = np.load(f"/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04_2026/coords_04_test.npz",allow_pickle=False)
mu = loaded["solar_mu"]
sigma = loaded["solar_sigma"]

x_target,time_df = bulid_feature(lat,lon,'2026-01-01 00:00:00','2026-10-01 23:59:59')
x_target_batches = torch.split(
    x_target,
    512,
    dim=0,
)
for (batch, x_target_batch) in zip(train_loader, x_target_batches):
    x_context = batch['x_context'].to(device,dtype=torch.float32)
    y_context = batch['y_context'].to(device,dtype=torch.float32)
    
    # 构建查询点的特征
    x_target_batch = x_target_batch.to(device,dtype=torch.float32)

    with torch.no_grad():
        pre = model(x_context, y_context, x_target_batch)
    data.append(pre.detach().float().cpu().numpy())

# 按 batch 维度合并
pred_all = np.concatenate(data, axis=0)

# 保存 CSV
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# 反归一化

pre = np.expm1(pred_all * sigma + mu).reshape(-1)

result_df = pd.DataFrame({
    'time': time_df.to_numpy(),
    "pre": pre
})

result_df.to_csv(
    output_dir / '1704274070360686592.csv',
    index=False,
    encoding="utf-8-sig",
    float_format="%.8f",
)

