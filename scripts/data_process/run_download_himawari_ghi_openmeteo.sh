python3 download_himawari_ghi_openmeteo.py \
  --coords-dir /home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/coords_himawari \
  --pattern coords_01.csv \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --out /home/jinhm/data/solar/himawari/coords_01\
  --workers 1 \
  --min-interval 0.5 \
  --max-requests 10000 \
  --max-consecutive-failures 20 \
  --continue-on-error \
  > download_draw_01.log 2>&1 &