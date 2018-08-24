#!/usr/bin/env bash
cd ~/models/research/slim
DATA_DIR=/media/yanyu/1882684582682A08/CNN/Kaggle/data
python download_and_convert_data.py  --dataset_name=catdogs --dataset_dir="${DATA_DIR}"