@echo off
set CAFFE_DIR=D:/CNN/caffe
"%CAFFE_DIR%/build/tools/caffe" train --solver="solver.prototxt"
pause