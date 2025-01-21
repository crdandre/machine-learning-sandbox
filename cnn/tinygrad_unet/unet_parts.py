from tinygrad import Tensor, nn, function

# https://github.com/search?q=repo%3Atinygrad%2Ftinygrad%20sequential&type=code
class DownSample:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        self.conv_op = Tensor.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding),
                    function.Relu(in_channels, out_channels)????/
                    # nn.Relu(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size, padding),
                    # nn.ReLU(inplace=True)
                )

class UNet:

