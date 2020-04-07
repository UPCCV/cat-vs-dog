from torch import nn
import torchvision.models as models
import torch
from torchsummary import summary

class MnasNet(nn.Module):
    def __init__(self,num_classes=2):
        super(MnasNet,self).__init__()
        self.model_name = "MnasNet"
        model = models.mnasnet0_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.classifier[-1] = nn.Linear(1280,2)
        self.model = model
        
    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    mrnet = MnasNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mrnet.to(device)
    print(summary(mrnet,(3,224,224)))