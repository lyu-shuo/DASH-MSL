import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LeNet_5(nn.Module):
    # 2 conv, 2 pool , 3 linear
    def __init__(self):
        super().__init__()
        self.features = []
        self.linears = []
        self.layers = collections.OrderedDict()

        # input shape = [1,28,28]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),                 # 输出[6,24,24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                              # 输出[6,12,12]
        )
        self.features.append(self.conv1)
        self.layers['conv_pool_1'] = self.conv1

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),                 # [16,8,8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                               # [16,4,4]
        )
        self.features.append(self.conv2)
        self.layers['conv_pool_2'] = self.conv2

        self.linear_1 = nn.Sequential(
            nn.Linear(in_features=16*4*4, out_features=120),
            nn.ReLU()
        )
        self.linears.append(self.linear_1)
        self.layers['linear_1'] = self.linear_1


        self.linear_2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.linears.append(self.linear_2)
        self.layers['linear_2'] = self.linear_2


        self.linear_3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )
        self.linears.append(self.linear_3)
        self.layers['linear_3'] = self.linear_3


    def forward(self,input, client_level=None, server_level=None, features=False):
        # client-side model
        if client_level == 1:
            for layers in self.features[:1]:
                input = layers(input)            # input = image
            smashed_data = input
            return smashed_data

        if client_level == 2:
            for layers in self.features[:2]:
                input = layers(input)
            smashed_data = input
            return smashed_data

        if client_level == 3:
            for layers in self.features[:2]:
                input = layers(input)
            input = input.view(-1, 256)
            for layers in self.linears[:1]:
                input = layers(input)
            smashed_data = input
            return smashed_data

        if client_level == 4:
            for layers in self.features[:2]:
                input = layers(input)
            input = input.view(-1, 256)
            for layers in self.linears[:2]:
                input = layers(input)
            smashed_data = input
            return smashed_data

        # server-side model
        if server_level == 4:
            for layers in self.features[1:]:
                output = layers(input)                       # input = smashed_data
            output = output.view(-1,256)
            for layers in self.linears:
                output = layers(output)
            return output

        if server_level == 3:
            input = input.view(-1,256)
            for layers in self.linears:
                input = layers(input)
            output = input
            return output

        if server_level == 2:
            for layers in self.linears[1:]:
                input = layers(input)
            output = input
            return output

        if server_level == 1:
            for layers in self.linears[2:]:
                input = layers(input)
            output = input
            return output

        # mal full model
        if server_level==None and client_level==None and features==False:
            out = self.conv1(input)
            out = self.conv2(out)
            out = out.view(-1,256)
            out = self.linear_1(out)
            out = self.linear_2(out)
            out = self.linear_3(out)
            return out

        if  features==True:
            out = self.conv1(input)
            out = self.conv2(out)
            return out


class ResNet_9(nn.Module):
    def __init__(self,in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )


    def forward(self, x, client_level=None, server_level=None):
        if client_level == 1:
            out = self.conv1(x)
            return out

        if client_level == 2:
            out = self.conv1(x)
            out = self.conv2(out)
            return out

        if client_level == 3:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            return out

        if server_level == 6:
            out = self.conv2(x)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.fc(out)
            return out

        if server_level == 5:
            out = self.res1(x) + x
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.fc(out)
            return out

        if server_level == 4:
            out = self.conv3(x)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.fc(out)
            return out

        if server_level == None and client_level == None:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.fc(out)
            return out



class GCN(nn.Module):
    def __init__(self, input_dim = 1433, hidden_dim = 16, output_dim = 7):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, client_level=None, server_level=None):
        if client_level == 1:
            x = self.conv1(x, edge_index)
            return x

        if server_level == 1:
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


