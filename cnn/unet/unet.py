"""
Notes:

1. What is out_channels=num_classes?
--> Conv2d takes a desired output number of classes to segment
--> A kernel size of 1 means each pixel is mapped to one of num_classes categories given the number of in_channels
--> "A small fully-connected layer outputting the likely output class"
--> 

2. what is model(input_image)?
--> inherited from nn.Module, is same as model.__call__(input_image)
--> every Conv2d layer is initialized with random weights by default

3. Why does unet skip-connect like-sized layers from contracting and expanding paths??
--> The skip connections smash together downsampled feature maps and the upsampled feature maps "imprecisely". These are an empirically added technique which adds another information "perspective" to the upsampling.
--> Without skip connections, the network would be a more basic sequential encoder-decoder network (can be a "line" since skip connections no longer necessitate the "U")

4. Why are kernel sizes starting at 64 and doubling/halving?
--> Arbitrary lol. In theory it could be any sequence I think. Doubling and halving seems to be most common and effective.

"""
import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)
        
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        
        b = self.bottle_neck(p4)
        
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        
        out = self.out(up_4)
        return out
    
    
if __name__ == "__main__":
    double_conv = DoubleConv(256, 256)
    print(double_conv)
    
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    output = model(input_image)
    print(output.size()) #(1,10,512,512) expected?
    
    
# if __name__ == "__main__":
#     # Create two models with same architecture
#     model1 = UNet(3, 10)
#     model2 = UNet(3, 10)
    
#     # Same input
#     input_image = torch.rand((1, 3, 512, 512))
    
#     # Get outputs
#     output1 = model1(input_image)
#     output2 = model2(input_image)
    
#     # Compare outputs (they should be different due to random initialization)
#     print("Are outputs identical?", torch.allclose(output1, output2))
    
#     # Look at some actual weights from the first conv layer
#     print("\nModel 1 first conv weights sample:")
#     print(model1.down_convolution_1.conv.conv_op[0].weight[0,0,:3,:3])
#     print("\nModel 2 first conv weights sample:")
#     print(model2.down_convolution_1.conv.conv_op[0].weight[0,0,:3,:3])
    
        
        
        