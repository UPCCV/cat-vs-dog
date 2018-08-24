
MODELS_DIR=~/models/research/slim
TENSORFLOW_DIR=~/CNN/tensorflow

MODEL_NAME=inception_v3

cd ${MODELS_DIR}


if [ -f ${MODEL_NAME}_inf_graph.pb ]
then
echo ${MODEL_NAME}_inf_graph.pb "laready exists"
python export_inference_graph.py --alsologtostderr --model_name=${MODEL_NAME} --batch_size=1 --dataset_name=imagenet --image_size=299 --output_file=${MODEL_NAME}_inf_graph.pb
fi

if [ -f frozen_${MODEL_NAME}.pb ]
then
echo frozen_${MODEL_NAME}.pb "laready exists"
${TENSORFLOW_DIR}/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=${MODEL_NAME}_inf_graph.pb --input_binary=true --input_checkpoint=${MODEL_NAME}/${MODEL_NAME}.ckpt --output_graph=frozen_${MODEL_NAME}.pb --output_node_name=InceptionV3/Predictions/Reshape_1
fi
mvNCCompile -s 12 frozen_${MODEL_NAME}.pb -in=input -on=InceptionV3/Predictions/Reshape_1