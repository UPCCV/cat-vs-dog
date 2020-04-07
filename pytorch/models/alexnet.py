import torch
from torch import nn
import torchvision.models as models
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet,self).__init__()
        self.model_name = "AlexNet"
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(4096,2)
        self.model = model

    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    mrnet = AlexNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mrnet.to(device)
    print(summary(mrnet,(3,224,224)))