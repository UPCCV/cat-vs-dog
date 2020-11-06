#coding=utf-8
import os
import sys
caffe_root = os.path.expanduser("~")+"/CNN/caffe"
#sys.path.insert(0, caffe_root + '/python')
import caffe
from caffe import layers as L,params as P,to_proto
from caffe.proto import caffe_pb2

root_folder="../data/train/"
batch_size=32

def create_data_layer(split="train",from_lmdb=True):
    if split=="train":
        shuffle=True
        batch_size = 32
        mirror = True
    else:
        shuffle = False
        batch_size = 8
        mirror = False
    transform_param=dict(crop_size=227, mean_file="modeldef/mean.binaryproto", mirror=mirror)
    if from_lmdb:
        source = "lmdb/"+split+"_lmdb"
        data, label = L.Data(data_param=dict(source=source,batch_size=batch_size,backend=P.Data.LMDB),ntop=2,transform_param=transform_param)
    else:
        source = "util/"+split+".txt"
        data, label = L.ImageData(image_data_param=dict(source=source,root_folder=root_folder,batch_size=batch_size,shuffle=shuffle,new_width=256,new_height=256),ntop=2,
        transform_param=transform_param)
    return data, label

def conv_bn_relu(input,num_output,stride=1,is_train=True):
    conv=L.Convolution(input,kernel_size=3, stride=stride,num_output=num_output,weight_filler=dict(type='xavier'))
    if is_train:
        bn=L.BatchNorm(
            conv, batch_norm_param = dict(use_global_stats = False), 
                in_place = True, param = [dict(lr_mult = 0, decay_mult = 0), 
                                          dict(lr_mult = 0, decay_mult = 0), 
                                          dict(lr_mult = 0, decay_mult = 0)])
    else:
        bn=L.BatchNorm(
            conv, batch_norm_param = dict(use_global_stats = True), 
                in_place = True, param = [dict(lr_mult = 0, decay_mult = 0), 
                                          dict(lr_mult = 0, decay_mult = 0), 
                                          dict(lr_mult = 0, decay_mult = 0)])
    relu = L.ReLU(bn, in_place = True)
    return relu

def create_mrnet(net,num_class=2,is_train=True):
    x=conv_bn_relu(net.data,32,stride=2)
    x=conv_bn_relu(x,32)
    x=conv_bn_relu(x,64,stride=2)
    x=conv_bn_relu(x,64)
    x=conv_bn_relu(x,64)
    x=conv_bn_relu(x,128,stride=2)
    x=conv_bn_relu(x,128)
    x=conv_bn_relu(x,128)
    x=conv_bn_relu(x,128)
    x=conv_bn_relu(x,256,stride=2)
    x=conv_bn_relu(x,256)
    x=conv_bn_relu(x,256)
    x=conv_bn_relu(x,512,stride=2)
    x=conv_bn_relu(x,512)
    x=conv_bn_relu(x,512)
    x=L.InnerProduct(x,num_output=1024,weight_filler=dict(type='xavier'))
    x=L.InnerProduct(x,num_output=100,weight_filler=dict(type='xavier'))
    x=L.InnerProduct(x,num_output=num_class,weight_filler=dict(type='xavier'))
    return x

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1),dict(lr_mult = 2, decay_mult = 0)],
                                num_output=nout, pad=pad, group=group)
    return L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,weight_filler=dict(type='xavier'))
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def create_alexnet(net,num_class=2):
    x = conv_relu(net.data, 11, 96, stride=4)
    x = max_pool(x, 3, stride=2)
    x = L.LRN(x, local_size=5, alpha=1e-4, beta=0.75)
    x = conv_relu(x, 5, 256, pad=2, group=2)
    x = max_pool(x, 3, stride=2)
    x = L.LRN(x, local_size=5, alpha=1e-4, beta=0.75)
    x = conv_relu(x, 3, 384, pad=1)
    x = conv_relu(x, 3, 384, pad=1, group=2)
    x = conv_relu(x, 3, 256, pad=1, group=2)
    x = max_pool(x, 3, stride=2)
    x = fc_relu(x, 4096)
    x = L.Dropout(x, in_place=True)
    x = fc_relu(x, 4096)
    x = L.Dropout(x, in_place=True)
    x = L.InnerProduct(x, num_output=num_class,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)])
    return x

def gen_prototxt():
    num_class=0
    with open("modeldef/labels.txt") as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split(" "))==2:
                num_class+=1
    net = caffe.NetSpec()
    net.data, net.label = create_data_layer()
    net.feature = create_mrnet(net,num_class=2)
    net.loss = L.SoftmaxWithLoss(net.feature, net.label)
    net.acc = L.Accuracy(net.feature, net.label)
    with open("train.prototxt","w")as f:
        f.write(str(net.to_proto()))

    net.data, net.label = create_data_layer("val")
    net.feature = create_mrnet(net,num_class=2,is_train=False)
    net.acc = L.Accuracy(net.feature, net.label)
    with open("val.prototxt","w")as f:
        f.write(str(net.to_proto()))

def gen_solver_txt(solver_file="solver.prototxt"):
    test_iter=1
    with open("util/val.txt")as f:
        lines=f.readlines()
        test_iter=(int)(len(lines)/batch_size)
    s = caffe_pb2.SolverParameter()
    s.train_net = 'train.prototxt'
    s.test_net.append('val.prototxt')
    s.test_interval = 10000
    s.test_iter.append(test_iter)
    s.max_iter = 500000
    s.base_lr = 0.01 
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'poly'
    s.stepsize=50000
    s.gamma = 0.9
    s.power=1
    s.display = 100
    s.snapshot = 10000
    s.snapshot_prefix = 'trainedmodels/'
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    with open(solver_file, 'w') as f:
        f.write(str(s))

def train(solver_file='solver.prototxt'):
    caffe.set_mode_gpu()
    if not os.path.exists('trainedmodels'):
        os.makedirs('trainedmodels')
    solver = caffe.SGDSolver(solver_file)
    solver.solve()

if __name__=="__main__":
    gen_prototxt()
    gen_solver_txt()
    train()