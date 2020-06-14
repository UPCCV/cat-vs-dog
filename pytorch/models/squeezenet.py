from torch import nn
import torchvision.models as models
import torch

class SqueezeNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SqueezeNet,self).__init__()
        self.model_name = "SqueezeNet"
        model = models.squeezenet1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-3] = nn.Conv2d(model.classifier[-3].in_channels,num_classes,kernel_size=1)
        self.model = model        
    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    net = SqueezeNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    from torchsummary import summary
    print(summary(net,(3,224,224)))