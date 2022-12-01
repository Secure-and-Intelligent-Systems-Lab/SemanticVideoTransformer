#@title "multi-layer perceptron (MLP) [55] which maps a(yi) in the space of xi with an MLP, which consists of two fully connected (FC) layers and a ReLU."

import torch
import torch.nn as nn

class Semantic_Mlp(nn.Module):
    def __init__(self, in_features=600, hidden_features=8192, 
                 out_features=8192, act_layer=nn.ReLU, drop=0.1):
        super(Semantic_Mlp,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.fcout = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.act(x1)
        x3 = self.drop(x2)
        x4 = self.fc2(x3)
        x5 = self.act(x4)
        x6 = self.drop(x5)
        x7 = self.fcout(x6)
        x8 = self.act(x7)
        return x8
    

    
class Claster_Mlp(nn.Module):
    def __init__(self, in_features=8192, hidden_features=8192, 
                 out_features=8192, act_layer=nn.ReLU, drop=0.1):
        super(Claster_Mlp,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)
        self.fcout = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.act(x1)
        x3 = self.drop(x2)
        x4 = self.fc2(x3)
        x5 = self.act(x4)
        x6 = self.drop(x5)
        x7 = self.fcout(x6)
        return x7
    
    
#@title "The final Pytorch classification MLP consists of two convolutional layers and two FC layers, where the last layer equals the number of unseen classes in the dataset we are looking at."

class Classification_Mlp(nn.Module):
    def __init__(self, in_features=16384, hidden_features=4096, 
                 out_features=51, act_layer=nn.ReLU, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(16384, 16384, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv1d(16384, 16384, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.act = act_layer()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
#         x = self.act(self.conv1(x))
        # #Tensor with shape torch.Size([10, 16384, 1])
#         x = self.pool1(x)
        # #Tensor with shape torch.Size([10, 16384, 1])
        # x = self.act(self.conv2(x))
        # #Tensor with shape torch.Size([10, 16384, 1])
        x = self.fc1(x[:,:,0])
        #Tensor with shape torch.Size([10, 16384])
        # Getting rid of the Length of signal sequence
        x = self.act(x)
        #Tensor with shape torch.Size([10, 16384])
        #Tensor with shape torch.Size([10, 16384])
        x = self.fc2(x)
        #Tensor with shape torch.Size([10, 51])
        return x
    

class Classde2(nn.Module):
    def __init__(self, in_features=8192, hidden_features=8192, 
                 out_features=51, act_layer=nn.ReLU, drop=0.2):
        super().__init__()
#         self.conv1 = nn.Conv1d(16384, 16384, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv1d(16384, 16384, kernel_size=1, padding=0)
#         self.pool1 = nn.MaxPool1d(kernel_size=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.act = act_layer()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x