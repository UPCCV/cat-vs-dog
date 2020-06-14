#!/usr/bin/env sh
set -e
RESUME=0
TOOLS=~/CNN/caffe/build/tools
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
if [ $RESUME -eq 1 ]; then
echo "Resume from $resumemodel"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt --weights="$resumemodel"  2>&1 | tee train.log
else
echo "Start Training"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt  2>&1 | tee train.log
fi
echo "Training Done"
