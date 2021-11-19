import torch
import torch.nn as nn

import torch.nn.functional as F
class img_clif_net(nn.Module):
    def __init__(self,input_channel,num_classes):
        super(img_clif_net,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,8,5,padding=2)
        self.conv2 = nn.Conv2d(8,16,5,padding=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024,256)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),3)
        x = F.max_pool2d(F.relu(self.conv2(x)),4)
        x = x.view(x.size(0),-1)
        x = self.dropout1(x)
        x_1 = self.fc1(x)
        return  x_1,x