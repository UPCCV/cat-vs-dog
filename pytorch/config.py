# coding:utf8
import warnings
import torch

class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097 # visdom 端口
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '../data/train/'  # 训练集存放路径
    test_data_root = '../data/test1'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 0  # how many workers for loading data
    print_freq = 20  # print info every N batch
    result_file = 'result.csv'

    max_epoch = 1000
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =torch.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

    def parse(self,args):
        self.use_gpu=args.use_gpu
        opt.device =torch.device('cuda') if opt.use_gpu else torch.device('cpu')
        self.num_workers=args.num_workers
        self.batch_size = args.batch_size
        self.model = args.model
        if args.load_model_path:
            self.load_model_path = args.load_model_path

opt = DefaultConfig()
