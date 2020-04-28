mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/CNN/libtorch ..
make -j4
./example-app ../data/5.jpg