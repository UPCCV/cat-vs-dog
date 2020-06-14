#!/usr/bin/env sh
set -e
RESUME=1
RESUME_ITER=10000
TOOLS~/CNN/caffe/build/tools
MODEL=AlexNet
#AlexNet
#BN-GoogleNet
#VGG16
#VGG19
#MobileNet
#SqueezeNet
#ResNet50
#ResNet101
#VGG19
#GoogleNet

#resumemodel="/home/yanyu/CNN/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel"
resumemodel="/home/yanyu/CNN/caffe/models/AlexNet_BN/AlexNet_BN.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/VGG16/VGG_ILSVRC_16_layers.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/VGG19/VGG_ILSVRC_19_layers.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/BN-GoogLeNet/Inception21k.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/ResNet-50/ResNet-50-model.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/ResNet-101/ResNet-101-model.caffemodel"
#resumemodel="/home/yanyu/CNN/caffe/models/ShuffleNet/shufflenet_1x_g3.caffemodel"
if [ $RESUME -eq 1 ]; then
echo "Resume from $resumemodel"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt --weights="$resumemodel"  2>&1 | tee train.log
else
echo "Start Training"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt  2>&1 | tee train.log
fi
echo "Training Done"
