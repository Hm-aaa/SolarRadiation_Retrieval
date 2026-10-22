
export PTREE_USER='2929811252_qq.com'
export PTREE_PASS='SP+wari8'

nohup python -u download_himawari_solar.py \
        --start "2020-01-01 00:00" --end "2026-10-31 23:59" \
        --out /home/jinhm/data/himawari_solar \
        > /home/jinhm/data/himawari_solar/log/download_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# pkill -f download_himawari_solar.py
# pgrep -af download_himawari_solar.py     # 没有输出就说明已停