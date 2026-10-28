import glob
import pandas as pd
import numpy as np

def load_data_by_time_range(
        csv_file,
        start_time,
        end_time,
    ):
        """
        读取 CSV，并按 UTC 时间范围筛选。

        Parameters
        ----------
        csv_file:
            CSV 文件路径。
        start_time:
            起始时间，包含。
        end_time:
            结束时间，包含。
        """
        data = pd.read_csv(csv_file)

        data["TIME_UTC"] = pd.to_datetime(
            data["TIME_UTC"],
            utc=True,
            errors="raise",
        )

        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)

        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")

        if end_time.tzinfo is None:
            end_time = end_time.tz_localize("UTC")
        else:
            end_time = end_time.tz_convert("UTC")

        mask = (
            (data["TIME_UTC"] >= start_time)
            & (data["TIME_UTC"] <= end_time)
        )

        data = data.loc[mask].reset_index(drop=True)

        return data

# 根据训练，验证，测试读取数据。
label = "train"
project = 'coords_04'

if label == 'train':
    start = "2015-01-01 00:00:00"
    end = "2024-12-31 23:59:59"
else:
    start = "2025-01-01 00:00:00"
    end = "2025-12-31 23:59:59"

# data_dir 数据集合路径
data_dir = f'/home/jinhm/data/solar/era5_solar/MirrorEarth/{project}'

data_List = []
for coords_file_item in glob.glob(f'{data_dir}/*'):
    data = load_data_by_time_range(
        csv_file=coords_file_item,
        start_time=start,
        end_time=end
    )
    # 先删除 TIME_UTC，避免变成 object 数组
    data = data.drop(columns=["TIME_UTC"], errors="ignore")
    data_List.append(data.to_numpy(dtype=np.float32))

# 合并
# 0 shortwave_radiation_W/m²
# 1 latitude
# 2 longitude
# 3 elevation_m
# 4 solar_zenith_deg
# 5 solar_azimuth_deg
# 6 minute_of_day
# 7 day_of_year
data_List = np.stack(data_List, axis=0)

# 1 经纬度 进行归一化操作
latitude_norm = (data_List[:,:,1] - 15) / (55 - 15)
longitude_norm = (data_List[:,:,2] - 70) / (140 - 70)

# 2 elevation_m 全局 Z-Score = (x - mu) / sigma
elev = data_List[:, :, 3]

if label == 'train':
    mu = np.nanmean(elev)
    sigma = np.nanstd(elev)
else:
    loaded = np.load(f"/home/jinhm/data/solar/era5_solar/MirrorEarth/{project}/{project}_{label}.npz",allow_pickle=False)
    mu = loaded["elevation_mu"]
    sigma = loaded["elevation_sigma"]

if sigma == 0:
    elevation_norm = elev / 1000.0
else:
    elevation_norm = (elev - mu) / sigma

# 3 太阳几何 归一化
zenith_rad = np.deg2rad(data_List[:,:,4])
azimuth_rad = np.deg2rad(data_List[:,:,5])

# 太阳方向单位向量
zenith_cos_norm = np.cos(zenith_rad)

# 方位角周期编码
azimuth_sin_norm = np.sin(azimuth_rad)
azimuth_cos_norm = np.cos(azimuth_rad)

# 4 时间特征编码

# 一天周期
minute_phase = (
    2.0 * np.pi * data_List[:,:,6] / 1440.0
)

# 一年周期
day_phase = (
    2.0 * np.pi * (data_List[:,:,7] - 1.0) / 365.0
)

minute_sin = np.sin(minute_phase)
minute_cos = np.cos(minute_phase)

day_sin = np.sin(day_phase)
day_cos = np.cos(day_phase)

# 5 辐照度
if label == 'train':
# 避免 log(0)
    ghi_log = np.log1p(data_List[:,:,0])
    solar_mu = np.nanmean(ghi_log)
    solar_sigma = np.nanstd(ghi_log)
else:
    loaded = np.load(f"/home/jinhm/data/solar/era5_solar/MirrorEarth/{project}/{project}_{label}.npz",allow_pickle=False)
    solar_mu = loaded["solar_mu"]
    solar_sigma = loaded["solar_sigma"]

ghi_norm = (ghi_log - solar_mu) / solar_sigma


# 合并新的归一化结构，保存为npz
X = np.stack(
    [
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
        ghi_norm
    ],
    axis=-1,
).astype(np.float32)

feature_names = np.array(
    [
        "latitude_norm",
        "longitude_norm",
        "elevation_norm",
        "zenith_cos",
        "azimuth_sin",
        "azimuth_cos",
        "minute_sin",
        "minute_cos",
        "day_sin",
        "day_cos",
        "ghi_norm"
    ]
)

X = np.transpose(X, (1, 0, 2))
print("X shape:", X.shape)
print("X dtype:", X.dtype)

# 6. 保存为 npz
np.savez_compressed(
    f"/home/jinhm/data/solar/era5_solar/MirrorEarth/{project}/{project}_{label}.npz",
    X=X,
    feature_names=feature_names,
    elevation_mu=np.float32(mu),
    elevation_sigma=np.float32(sigma),
    solar_mu=np.float32(solar_mu),
    solar_sigma=np.float32(solar_sigma),
)