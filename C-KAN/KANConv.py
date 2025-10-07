import torch
import math
import sys
sys.path.append('./kan_convolutional') # path for kan_convolutional module
from KANLinear import KANLinear
import convolution


# kan convolutional layer implementation
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        """kan convolutional layer.
        
        args:
            in_channels: input image channels.
            out_channels: output feature map channels.
            kernel_size: convolution kernel dimensions.
            stride: kernel movement step.
            padding: border added to input.
            dilation: spacing between kernel elements.
            grid_size: kan spline grid points.
            spline_order: b-spline order.
            scale_noise: noise for spline initialization.
            scale_base: base function scaling.
            scale_spline: spline function scaling.
            base_activation: activation function for base.
            grid_eps: grid epsilon for adaptive grid.
            grid_range: grid value range.
            device: cpu or cuda.
        """


        super(KAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        # device for pytorch ddp
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.stride = stride

        
        # create kan_convolution objects
        for _ in range(in_channels*out_channels):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,
                    stride = stride,
                    padding=padding,
                    dilation = dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # device allocation as per input device for pytorch ddp
                )
            )

    def forward(self, x: torch.Tensor):
        # apply all convolutions
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.out_channels,self.stride,self.dilation,self.padding,self.device)
        
        

# single kan convolution
class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu"
        ):
        """kan convolution initialization.
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # device for pytorch ddp
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor):
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(x, [self],self.kernel_size[0],1,self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # sum of regularization losses for layers
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)