# NEW architecture:

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, forward_expansion: int,  out_channels: int, expand: bool):
        super(BasicBlock, self).__init__()
        """
        A very simple convlution block. Reduces or expands the size of the image by a factor of 2.
        When using batchnorm, you can set bias=False to preceding convolution.
        """
        self.conv1 = nn.Conv2d(in_channels, forward_expansion, 3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(forward_expansion)
        self.conv2 = nn.Conv2d(forward_expansion, out_channels, 3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if expand:
            self.scaling = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.scaling = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.scaling(x)

        return x

    
class SampleModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=25):
        super(SampleModel, self).__init__()

        # Create downsizing pass
        self.b1 = BasicBlock(3, 256, 256, expand=False)
        self.b2 = BasicBlock(256, 512, 512, expand=False)
        self.b3 = BasicBlock(512, 1024, 1024, expand=False)
        self.b4 = BasicBlock(1024, 2048, 2048, expand=False)
        
        # Create bottleneck
        self.bottleneck = nn.Conv2d(2048, 2048, 3, stride=1, padding='same')
        
        # Create upsizing pass
        self.b52 = BasicBlock(4096, 4096, 2048, expand=True)
        self.b5 = BasicBlock(2048, 2048, 1024, expand=True)
        self.extrapool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.b62 = BasicBlock(2048, 2048, 1024, expand=True)
        self.b6 = BasicBlock(1024, 1024, 512, expand=True)
        self.extrapool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.b72 = BasicBlock(1024, 1024, 512, expand=True)
        self.b7 = BasicBlock(512, 512, 256, expand=True)
        self.extrapool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.b82 = BasicBlock(512, 512, 256, expand=True)
        self.b8 = BasicBlock(256, 256, out_channels, expand=True)
        self.extrapool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        
    def forward(self, x):

        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        
        x5 = self.bottleneck(x4)
        
        x6 = torch.cat((x4, x5), dim=1)
        x72 = self.b52(x6)
        x7 = self.b5(x72)
        x7e = self.extrapool1(x7)

        x8 = torch.cat((x3, x7e), dim=1)
        x92 = self.b62(x8)
        x9 = self.b6(x92)
        x9e = self.extrapool2(x9)

        x10 = torch.cat((x2, x9e), dim=1)
        x112 = self.b72(x10)
        x11 = self.b7(x112)
        x11e = self.extrapool3(x11)

        x12 = torch.cat((x1, x11e), dim=1)
        x132 = self.b82(x12)
        x13 = self.b8(x132)
        output = self.extrapool3(x13)

        return output
