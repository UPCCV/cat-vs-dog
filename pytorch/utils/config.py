# coding:utf8
import warnings
import torch
import logging
class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097 # visdom 端口
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = 'data/dogcat/train/'  # 训练集存放路径
    test_data_root = 'data/dogcat/test1'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 40  # how many workers for loading data
    print_freq = 100  # print info every N batch
    result_file = 'result.csv'

    max_epoch = 10000
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    device = torch.device('cpu')
    dataset = "DogCat"

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                logging.info(str(k)+":"+str(getattr(self, k)))
                
opt = DefaultConfig()