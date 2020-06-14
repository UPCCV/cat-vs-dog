#coding=utf-8
import os
import sys
caffe_root = os.path.expanduser("~")+"/CNN/caffe"
sys.path.insert(0, caffe_root + '/python')
import caffe
from caffe import layers as L,params as P,to_proto
from caffe.proto import caffe_pb2

root_folder="../data/train/"
batch_size=32

def create_data_layer(split="train"):
    if split=="train":
        txtfile = "util/train.txt"
        shuffle=True
        batch_size = 32
        mirror = True
    else:
        txtfile = "util/val.txt"
        shuffle = False
        batch_size = 8
        mirror = False
    data, label = L.ImageData(image_data_param=dict(source=txtfile,root_folder=root_folder,batch_size=batch_size,shuffle=shuffle,new_width=256,new_height=256),ntop=2,
        transform_param=dict(crop_size=227, mean_value=[127.5,127.5,127.5],scale=0.007843, mirror=mirror))
    return data, label

def conv_bn_scale_relu(input,num_output=64,stride=2,is_train=True):
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
    scale= L.Scale(bn, scale_param = dict(bias_term = True))
    relu = L.ReLU(scale, in_place = True)
    return relu

def create_mrnet(data,num_class=2):
    relu1=conv_bn_scale_relu(data)
    #pool1=L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    relu2=conv_bn_scale_relu(relu1)
    #pool2=L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    relu3=conv_bn_scale_relu(relu2)
    #pool3=L.Pooling(relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    relu4=conv_bn_scale_relu(relu3)
    #pool4=L.Pooling(relu4, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    relu5=conv_bn_scale_relu(relu4)
    #pool5=L.Pooling(relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    fc4=L.InnerProduct(relu5, num_output=512,weight_filler=dict(type='xavier'))
    drop4 = L.Dropout(fc4, in_place=True)
    fc5 = L.InnerProduct(drop4, num_output=num_class,weight_filler=dict(type='xavier'))
    return fc5

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1),dict(lr_mult = 2, decay_mult = 0)],
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,weight_filler=dict(type='xavier'))
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def create_alexnet(data,num_class=2):
    _, relu1 = conv_relu(data, 11, 96, stride=4)
    pool1 = max_pool(relu1, 3, stride=2)
    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    _, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2)
    pool2 = max_pool(relu2, 3, stride=2)
    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    _, relu3 = conv_relu(norm2, 3, 384, pad=1)
    _, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2)
    _, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = max_pool(relu5, 3, stride=2)
    _, relu6 = fc_relu(pool5, 4096)
    drop6 = L.Dropout(relu6, in_place=True)
    _, relu7 = fc_relu(drop6, 4096)
    drop7 = L.Dropout(relu7, in_place=True)
    fc8 = L.InnerProduct(drop7, num_output=num_class,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1), dict(lr_mult = 2, decay_mult = 0)])
    return fc8

def gen_prototxt():
    num_class=0
    with open("modeldef/labels.txt") as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split(" "))==2:
                num_class+=1
    data,label = create_data_layer()
    feature = create_mrnet(data,num_class)
    loss = L.SoftmaxWithLoss(feature, label)
    acc = L.Accuracy(feature, label)
    with open("train.prototxt","w")as f:
        f.write(str(to_proto(loss,acc)))

    data,label = create_data_layer("val")
    feature = create_mrnet(data,num_class)
    acc = L.Accuracy(feature, label)
    with open("val.prototxt","w")as f:
        f.write(str(to_proto(acc)))

def gen_solver_txt(solver_file="solver.prototxt"):
    test_iter=1
    with open("util/val.txt")as f:
        lines=f.readlines()
        test_iter=(int)(len(lines)/batch_size)
    s = caffe_pb2.SolverParameter()
    s.train_net = 'train.prototxt'
    s.test_net.append('val.prototxt')
    s.test_interval = 1000
    s.test_iter.append(test_iter)
    s.max_iter = 500000
    s.base_lr = 0.1 
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'poly'
    s.stepsize=1000
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
    solver = caffe.SGDSolver(solver_file)
    solver.solve()

if __name__=="__main__":
    gen_prototxt()
    gen_solver_txt()
    train()