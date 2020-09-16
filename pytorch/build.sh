#/bin/bash
if [ ! -d build ] ; then
    mkdir build
fi
cd build
cmake -DCMAKE_PREFIX_PATH=~/CNN/libtorch ..
make -j4
./example-app ../data/5.jpg