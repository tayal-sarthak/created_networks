import torch
from torch import nn
import torch.nn.functional as F

from KANConv import KAN_Convolutional_Layer

# kan convolutional mlp model
class KANC_MLP(nn.Module):
    def __init__(self,grid_size: int = 5):
        super().__init__()
        # first kan convolutional layer
        self.conv1 = KAN_Convolutional_Layer(in_channels=1,
            out_channels= 5,
            kernel_size= (3,3),
            grid_size = grid_size
        )

        # second kan convolutional layer
        self.conv2 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 5,
            kernel_size = (3,3),
            grid_size = grid_size
        )

        # max pooling layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        # flatten layer
        self.flat = nn.Flatten() 
        
        # final linear layer
        self.linear1 = nn.Linear(125, 10)
        self.name = f"kanc mlp (small) (gs = {grid_size})"


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        # print(x.shape) # debug print for shape
        x = self.linear1(x)
        x = F.log_softmax(x, dim=1)
        return x