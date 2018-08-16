caffe_dir=~/CNN/caffe
python ${caffe_dir}/tools/extra/parse_log.py train.log ./
python ${caffe_dir}/tools/extra/plot_training_log.py 0 test_acc.png train.log
python ${caffe_dir}/tools/extra/plot_training_log.py 6 train_loss.png train.log