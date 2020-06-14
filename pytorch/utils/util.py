import os
import torch
import argparse

def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_lastest_model(modeldir="output",prefix="MnasNet"):
    files = os.listdir(modeldir)
    if len(files)==0:
        return None
    files = list(filter(lambda x: x.startswith(prefix),files))
    files.sort(key=lambda fn:os.path.getmtime(modeldir + "/" + fn))
    if len(files) > 0:
        lastest = os.path.join(modeldir,files[-1])
    else:
        lastest = None
    return lastest

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=str,default="ShuffleNet",choices=["MRNet","AlexNet","ResNet50","SqueezeNet","MnasNet","ShuffleNet"])
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--num_workers",type=int,default=40)
    parser.add_argument("--load_model_path",type=str,default=None)
    # for demo
    parser.add_argument("--image_dir",type=str,default="data/dogcat/test1")
    parser.add_argument("--image_path",type=str,default="data/dogcat/test1/1.jpg")
    return parser.parse_args()

if __name__=="__main__":
   print(get_lastest_model())