#!/usr/bin/env sh
set -e
TOOLS=/home/yanyu/CNN/caffe/build/tools
DATA=../
TRAIN_DATA_ROOT=$DATA/
VAL_DATA_ROOT=$DATA/
EXAMPLE=lmdb
RESIZE=true
rm lmdb -r
mkdir lmdb
if $RESIZE;then
RESIZE_HEIGHT=256
RESIZE_WIDTH=256
else
RESIZE_HEIGHT=256
RESIZE_WIDTH=256
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    util/train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    util/val.txt \
    $EXAMPLE/val_lmdb

echo "Start compute mean."

"$TOOLS/compute_image_mean" "lmdb/train_lmdb" "modeldef/mean.binaryproto"

echo "Start Training"
$TOOLS/caffe train --solver=modeldef/AlexNet/solver.prototxt
echo "Training Done"
