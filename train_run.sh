#!/bin/bash
# 倒入项目根路径
export PROJECT_PATH='/home/jinhm/PhD_Project/SolarRadiation_Retrieval'
export PYTHONPATH="$PWD"

nohup python scripts/train/pre-trained.py > logs/config/config.log 2>&1 &

# tensorboard --logdir runs --bind_all --port 6060
# pgrep -af "python scripts/train/pre-trained.py"