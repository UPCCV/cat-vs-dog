#Hyperparameters config
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/ar/gongyanhe/libs/cuda/lib64/:$LD_LIBRARY_PATH

python3 train.py --num_workers=4
