"""
Notes:

1. What is x?
--> x.shape = (batch_size, in_channels, height, width)
--> x is a tensor that represents the input feature maps
--> ex. 32 256^2 RGB images would mean that x.shape = (32, 3, 256, 256)

2. Why is "x" the only input to the operations?
--> self.conv_op(x) == self.conv_op.__call__(x)
--> core feature of nn.Module to string operations together

3. What is Conv2d?
--> Convolution kernel values themselves apply a transform which outputs a feature map better suited for feature detection and understanding, i.e. Sobel for edge detection. 
--> in this case, the weights of the network are the values of the convolution kernels themselves! and each kernel gets one bias to adjust the magnitude of it's influence on what features "pass" the ReLU threshold (0). Also a learnable parameter for when a feature detector is useful (i.e. same feature in low vs high contrast - avoiding false positive by adjusting sensitivity)
--> this is all managed internally (lends itself to doing this from scratch to have intuition!)

4. Why ReLU afer Conv2d? / Why Double Conv?
--> introduces nonlinearity (conv only would prevent unet from capturing nonlinearity)
--> double conv is empirical design choice - just a dimension to play with

5. Why cat x1/2?
--> part of more information "perspective" to feed into transpose convs, it's "messy" but combines encoder info about objects and positions with pixel-level segmentation info from the decoder as the decoding continues

6. What is Transposed convolution (upconvolution)?
--> a transform to extrapolate spatial information from more abstract feature spaces eventually into the original image size
--> learnable transform which expands the input, the weights and biases tune the nature of the expansion.
--> zero insertions based on stride, then convolution applied to this expanded matrix with zeros interspersed
--> not de-convolution or inverse convolution, it's not undoing prior convolution, it's a separate transform


"""


import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
    

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
        
        