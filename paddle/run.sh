#Hyperparameters config
export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=/home/ar/gongyanhe/libs/cuda/lib64/:$LD_LIBRARY_PATH
export FLAGS_fraction_of_gpu_memory_to_use=0.9

#AlexNet:
#python train.py \
     --model=AlexNet \
     --batch_size=256 \
     --total_images=20000 \
     --class_dim=2 \
     --image_shape=3,224,224 \
     --model_save_dir=output/ \
     --with_mem_opt=False \
     --lr_strategy=piecewise_decay \
     --num_epochs=120 \
     --lr=0.01 \
     >log_AlexNet.txt 2>&1 &

#VGG11:
#python train.py \
#       --model=VGG11 \
#       --batch_size=512 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=120 \
#       --lr=0.1


#Mnasnet:
# python train.py \
#        --model=Mnasnet \
#        --batch_size=32 \
#        --total_images=20000 \
#        --class_dim=2 \
#        --image_shape=3,224,224 \
#        --model_save_dir=output/ \
#        --with_mem_opt=False \
#        --lr_strategy=piecewise_decay \
#        --num_epochs=120 \
#        --lr=0.1 \
       #>log_Mnasnet.txt 2>&1 &


#ResNet50:
#python train.py \
#       --model=ResNet50 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1

#ResNet101:
#python train.py \
#       --model=ResNet101 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=False \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1

