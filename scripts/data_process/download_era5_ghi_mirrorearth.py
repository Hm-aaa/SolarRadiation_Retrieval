from mirrorearth_sdk import archive_api
import pandas as pd
import os

def download_msk_era5(
                        apikey="FCGC1mFso15vDAX1GSKYtlEYaY1UKC3T",
                        lat = 32,
                        lon = 119,
                        name = 'shortwave_radiation',
                        start_date='2015-01-01',
                        end_date="2026-01-01",
                        out_dir = '/home/jinhm/data/solar/era5_solar/MirrorEarth /coords_01'
                    ):
                    
    os.makedirs(out_dir, exist_ok=True)
    out_file = f'{out_dir}/coords_{lat}_{lon}.csv'
    if os.path.exists(out_file):
        print(f'{lat}_{lon}文件存在')
        return
    print(f'---正在下载{lat}_{lon}---')
    resp = archive_api(
        longitude=lon,
        latitude=lat,
        apikey=apikey,
        hourly=name,
        models="era5",
        start_date=start_date,
        end_date=end_date,
        temporal_resolution="hourly_1",   # 原始 1 小时分辨率
        timezone="UTC",                   # 明确使用 UTC
    )

    point = resp[0]
    df=point.hourly_df()
    df.to_csv(
        out_file,
        index=True,
        index_label="TIME_UTC"
    )

if __name__ == "__main__":

    point_file = '/home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/coords/coords_05.csv'
    point_data = pd.read_csv(point_file)
    for _, row in point_data.iterrows():
        lat = float(row.iloc[0])
        lon = float(row.iloc[1])
        try:
            download_msk_era5(lat=lat, lon=lon, out_dir = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_05')
        except:
            raise('存在问题！')

    # 后台静默下载
    # nohup python3 -u download_era5_ghi_mirrorearth.py > download_era5_ghi.log 2>&1 &