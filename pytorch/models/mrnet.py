from torch import nn
from torchsummary import summary
import torch
import math
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class MRNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(MRNet,self).__init__()
        self.model_name = "MRNet"
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.conv3 = nn.Conv2d(16,24,3)
        self.conv4 = nn.Conv2d(24,32,3)
        self.conv5 = nn.Conv2d(32,64,3)
        self.conv6 = nn.Conv2d(64,128,3)
        self.fc = nn.Linear(128,num_classes)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = F.max_pool2d(F.relu(self.conv4(x)),2)
        x = F.max_pool2d(F.relu(self.conv5(x)),2)
        x = F.max_pool2d(F.relu(self.conv6(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

if __name__=="__main__":
    net = MRNet()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    print(summary(net,(3,224,224)))
    dummy_input = torch.rand(1,3,224,224).to(device)
    with SummaryWriter(comment="mrnet") as w:
        w.add_graph(net,(dummy_input,))