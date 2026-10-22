python3 download_himawari_ghi_openmeteo.py \
  --coords-dir /home/jinhm/PhD_Project/SolarRadiation_Retrieval/data/coords \
  --pattern coords_01.csv \
  --start 2015-07-01 \
  --end 2026-10-01 \
  --out /home/jinhm/data/solar/himawari/coords_01\
  --workers 1 \
  --min-interval 0.5 \
  --max-requests 10000 \
  --max-consecutive-failures 20 \
  --continue-on-error \
  # > download_draw_01.log 2>&1 &