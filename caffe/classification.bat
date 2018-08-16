@echo off
set CAFFE_DIR=..
set eval_iter=10000
set DATA="D:/OCR/traffic-sign"
set imagepath=%DATA%/test/00000/00017_00000.png
set trainedmodel=trainedmodels/lenet_iter_%eval_iter%.caffemodel
::set trainedmodel=platere996.caffemodel
echo %imagepath% %eval_iter%
"%CAFFE_DIR%/build/examples/cpp_classification/classification" "modeldef/deploy.prototxt" "%trainedmodel%" "modeldef/mean.binaryproto" "modeldef/labels.txt" "%imagepath%"

pause