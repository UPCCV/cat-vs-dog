#!/usr/bin/env sh
set -e
RESUME=0
TOOLS=~/CNN/caffe/build/tools
MODEL=AlexNet

#compute mean
if [ ! -f "modeldef/mean.binaryproto" ]; then
    $TOOLS/compute_image_mean lmdb/train_lmdb modeldef/mean.binaryproto
fi

if [ $RESUME -eq 1 ]; then
echo "Resume from $resumemodel"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt --weights="$resumemodel"  2>&1 | tee train.log
else
echo "Start Training"
$TOOLS/caffe train --solver=modeldef/${MODEL}/solver.prototxt  2>&1 | tee train.log
fi
echo "Training Done"