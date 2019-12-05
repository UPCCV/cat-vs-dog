#coding=utf-8
import platform,sys
if platform.system()=="Windows":
    caffe_root = 'D:/CNN/caffe'
else:
    caffe_root = '/home/yanyu/CNN/caffe'
sys.path.insert(0, caffe_root + '/python')
from caffe import layers as L,params as P,to_proto
from caffe.proto import caffe_pb2
import caffe

def gen_data_layer(txtfile="util/train.txt",root_folder="../data/train/",crop_size=227,batch_size=32,deploy=False):
    data, label = L.ImageData(image_data_param=dict(source=txtfile,root_folder=root_folder,batch_size=batch_size,shuffle=shuffle,new_width=256,new_height=256),ntop=2,
        transform_param=dict(crop_size=crop_size, mean_value=[104, 117, 123], mirror=True))
    return data,label

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

def create_mrnet(txtfile="util/train.txt",num_class=2,root_folder="../data/train/",crop_size=227,batch_size=32,deploy=False):
    if deploy:
        shuffle=False
    else:
        shuffle=True
    data, label =L.ImageData(image_data_param=dict(source=txtfile,root_folder=root_folder,batch_size=batch_size,shuffle=shuffle,new_width=256,new_height=256),
            transform_param=dict(crop_size=crop_size,scale=0.0078125,mean_value=127.5),ntop=2)
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
    loss = L.SoftmaxWithLoss(fc5, label)
    if deploy:
        return to_proto(loss)
    else:
        acc = L.Accuracy(fc5, label)
        return to_proto(loss, acc)
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1),dict(lr_mult = 2, decay_mult = 0)],
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,weight_filler=dict(type='xavier'))
    return fc, L.ReLU(fc, in_place=True)
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
def create_alexnet(txtfile="util/train.txt",num_class=2,root_folder="../data/train/",crop_size=227,batch_size=32,deploy=False):
    if deploy:
        shuffle=False
    else:
        shuffle=True
    data, label = L.ImageData(image_data_param=dict(source=txtfile,root_folder=root_folder,batch_size=batch_size,shuffle=shuffle,new_width=256,new_height=256),ntop=2,
        transform_param=dict(crop_size=227, mean_file="modeldef/mean.binaryproto", mirror=True))
    conv1, relu1 = conv_relu(data, 11, 96, stride=4)
    pool1 = max_pool(relu1, 3, stride=2)
    norm1 = L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv2, relu2 = conv_relu(norm1, 5, 256, pad=2, group=2)
    pool2 = max_pool(relu2, 3, stride=2)
    norm2 = L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv3, relu3 = conv_relu(norm2, 3, 384, pad=1)
    conv4, relu4 = conv_relu(relu3, 3, 384, pad=1, group=2)
    conv5, relu5 = conv_relu(relu4, 3, 256, pad=1, group=2)
    pool5 = max_pool(relu5, 3, stride=2)
    fc6, relu6 = fc_relu(pool5, 4096)
    drop6 = L.Dropout(relu6, in_place=True)
    fc7, relu7 = fc_relu(drop6, 4096)
    drop7 = L.Dropout(relu7, in_place=True)
    fc8 = L.InnerProduct(drop7, num_output=num_class,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type="constant"), param = [dict(lr_mult = 1, decay_mult = 1), 
                                          dict(lr_mult = 2, decay_mult = 0)])
    loss = L.SoftmaxWithLoss(fc8, label)

    if not deploy:
        acc = L.Accuracy(fc8, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)
        
def gen_solver_txt(solver_file="solver.prototxt",batch_size=32):
    test_iter=1
    with open("util/val.txt")as f:
        lines=f.readlines()
        test_iter=(int)(len(lines)/batch_size)
    s = caffe_pb2.SolverParameter()
    path="./"
    s.train_net = path+'train.prototxt'
    s.test_net.append(path+'val.prototxt')
    s.test_interval=1000
    s.test_iter.append(test_iter)#
    s.max_iter =500000
    s.base_lr = 0.001 
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'poly'
    s.stepsize=100000
    s.gamma = 0.9
    s.power=1
    s.display = 1000
    s.snapshot = 10000
    s.snapshot_prefix = 'trainedmodels/'
    s.type = "SGD"
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    with open(solver_file, 'w') as f:
        f.write(str(s))

def gen_train_txt():
    num_class=0
    with open("modeldef/labels.txt") as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split(" "))==2:
                num_class+=1
    with open("train.prototxt","w")as f:
        f.write(str(create_alexnet("util/train.txt",num_class)))
    with open("val.prototxt","w")as f:
        f.write(str(create_alexnet("util/val.txt",num_class)))

def train(solver_file='solver.prototxt'):
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    solver.solve()

if __name__=="__main__":
    gen_train_txt()
    gen_solver_txt()
    train()