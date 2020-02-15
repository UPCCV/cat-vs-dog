#coding=utf8

import os
import argparse
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from data.dataset import DogCat
from config import opt

def get_lastest_model(modeldir="checkpoints"):
    files = os.listdir(modeldir)
    if len(files)==0:
        return None
    files.sort(key=lambda fn:os.path.getmtime(modeldir + "/" + fn))
    lastest = os.path.join(modeldir,files[-1])
    return lastest

@torch.no_grad()
def test(args):
    opt.parse(args)
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 2)
    if not args.load_model_path:
        args.load_model_path = get_lastest_model()
    if not args.load_model_path:
        print("No pretrained model found")
        return
    model.load_state_dict(torch.load(args.load_model_path))
    # step1: configure model
    # model = getattr(models, opt.model)()
    model.to(opt.device)
    val_data = DogCat(opt.train_data_root,train=False)
    val_dataloader = DataLoader(val_data,opt.batch_size,
            shuffle=False,num_workers=opt.num_workers)
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)

    for _, (val_input, label) in enumerate(tqdm(val_dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print("Acc:{acc}".format(acc=accuracy))
    print("confusion matrix:{val_cm}".format(val_cm = str(cm_value)))

def test_loader():
    train_data = DogCat(opt.train_data_root,train=True)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    for _,(data,label) in enumerate(train_dataloader):
        images = data.numpy()
        for image in images:
            img = np.transpose(image, (1,2,0))
            cv2.imshow("img",img)
            cv2.waitKey()

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use_gpu",type=bool,default=True,help='an integer for the accumulator')
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--print_freq",type=int,default=1)
    parser.add_argument("--model",type=str,default="SqueezeNet")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--load_model_path",type=str,default=None)
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    test(args)