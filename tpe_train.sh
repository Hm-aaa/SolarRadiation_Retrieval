#!/bin/bash
# 倒入项目根路径
export PROJECT_PATH='/home/jinhm/PhD_Project/SolarRadiation_Retrieval'
export PYTHONPATH="$PWD"

nohup python scripts/train/TPE_Trainer.py \
    --device cuda:0 \
    --train_sets /home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_train_v1.0.npz \
    --val_sets /home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_val_v1.0.npz \
    --context_fraction 0.5 \
    --send 42 \
    --saved_parame_path /home/jinhm/PhD_Project/SolarRadiation_Retrieval/outputs/checkpoints \
    --n_trials 100 \
    > logs/config/config_era_tpe.log 2>&1 &