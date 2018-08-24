#!/usr/bin/env bash
#DATASET_NAME=
export CUDA_VISIBLE_DEVICES=0
cd ~/models/research/slim
TENSORFLOW_DIR=~/CNN/tensorflow
#DATASET_NAME=catdogs
DATASET_NAME=flowers
DATASET_DIR=/home/yanyu/data/${DATASET_NAME}
MODEL_NAME=inception_v3
#resnet_v1_50
#resnet_v1_50
#mobilenet_v1
#inception_v4
#inception_v3
CHECKPOINT_PATH=${MODEL_NAME}/${MODEL_NAME}.ckpt 
TRAIN_DIR=/tmp/${DATASET_NAME}-models/${MODEL_NAME}
STEPS=1000

#prepare data
#DATA_DIR=/media/yanyu/1882684582682A08/CNN/Kaggle/data
#python download_and_convert_data.py  --dataset_name=${DATASET_NAME}  --dataset_dir="${DATA_DIR}"

#train model
#python train_image_classifier.py --train_dir=${TRAIN_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=${DATASET_NAME} --dataset_split_name=train --model_name=${MODEL_NAME} --max_number_of_steps=${STEPS} --checkpoint_path=${CHECKPOINT_PATH} --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits

#eval model
#python eval_image_classifier.py --checkpoint_path=${TRAIN_DIR} --eval_dir=models/${DATASET_NAME}/ --dataset_name=${DATASET_NAME} --dataset_split_name=validation --dataset_dir=${DATASET_DIR} --model_name=${MODEL_NAME}

#Generating inference graph
if [ -f ${MODEL_NAME}_inf_graph.pb ]
then
echo "Inference model already exists."
else
python export_inference_graph.py --alsologtostderr --model_name=${MODEL_NAME} --output_file=${MODEL_NAME}_inf_graph.pb --dataset_dir=${DATASET_DIR} --dataset_name=${DATASET_NAME}
fi
#Get output nodes
#${TENSORFLOW_DIR}/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=${MODEL_NAME}_inf_graph.pb
#freeze_gprah
if [ -f frozen_${MODEL_NAME}.pb ]
then
echo "Freezon model already exists."
else
${TENSORFLOW_DIR}/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=${MODEL_NAME}_inf_graph.pb --input_checkpoint=${TRAIN_DIR}/model.ckpt-${STEPS} --input_binary=true --output_graph=frozen_${MODEL_NAME}.pb --output_node_names=InceptionV3/Predictions/Reshape_1
fi

#test model
${TENSORFLOW_DIR}/bazel-bin/tensorflow/examples/label_image/label_image --image=rose.jpg --input_layer=input --output_layer=InceptionV3/Predictions/Reshape_1 --graph=frozen_${MODEL_NAME}.pb --labels=/home/yanyu/data/flowers/labels.txt --input_mean=0 --input_std=255

exit
#To tflite
#${TENSORFLOW_DIR}/bazel-bin/tensorflow/contrib/lite/toco/toco --input_file=frozen_${MODEL_NAME}.pb --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE --output_file=${MODEL_NAME}.lite --inference_type=FLOAT --input_type=FLOAT --input_arrays=input --output_arrays=InceptionV3/Predictions/Reshape_1  --input_shapes=1,299,299,3

#${TENSORFLOW_DIR}/bazel-bin/tensorflow/contrib/lite/examples/label_image/label_image -i rose.jpg -l /home/yanyu/data/flowers/labels.txt -m ${MODEL_NAME}.lite -a 0 -c 1 -s 255
#optimized_graph
if [ -f optimized_graph.pb ]
then
echo "optimized_graph exists"
else
#${TENSORFLOW_DIR}/bazel-bin/tensorflow/python/tools/optimize_for_inference --input=frozen_${MODEL_NAME}.pb  --output=optimized_graph.pb --input_names=input --output_names=InceptionV1/Predictions/Reshape_1
fi
#quantize
if [ -f quantized_graph.pb ]
then
echo "quantized_graph exists"
else
#${TENSORFLOW_DIR}/bazel-bin/tensorflow/tools/quantization/quantize_graph --input=optimized_graph.pb --output_node_names=InceptionV3/Predictions/Reshape_1 --output=quantized_graph.pb --mode=eightbit
fi
if [ -f quantized_graph.pb ]
then
#${TENSORFLOW_DIR}/bazel-bin/tensorflow/examples/label_image/label_image --image=rose.jpg --input_layer=input --output_layer=InceptionV3/Predictions/Reshape_1 --graph=quantized_graph.pb --labels=/home/yanyu/data/flowers/labels.txt --input_mean=0 --input_std=255
fi