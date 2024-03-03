# Defining CNN Block 
import sys
sys.path.append('./')

import torch 
import torch.nn as nn 

class CNNBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.activation = nn.LeakyReLU(0.1) 
        self.use_batch_norm = use_batch_norm 
  
    def forward(self, x): 
        # Applying convolution 
        x = self.conv(x) 
        # Applying BatchNorm and activation if needed 
        if self.use_batch_norm: 
            x = self.bn(x) 
            return self.activation(x) 
        else: 
            return x

# Defining residual block 
class ResidualBlock(nn.Module): 
    def __init__(self, channels, use_residual=True, num_repeats=1): 
        super().__init__() 
          
        # Defining all the layers in a list and adding them based on number of  
        # repeats mentioned in the design 
        res_layers = [] 
        for _ in range(num_repeats): 
            res_layers += [ 
                nn.Sequential( 
                    nn.Conv2d(channels, channels // 2, kernel_size=1), 
                    nn.BatchNorm2d(channels // 2), 
                    nn.LeakyReLU(0.1), 
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(channels), 
                    nn.LeakyReLU(0.1) 
                ) 
            ] 
        self.layers = nn.ModuleList(res_layers) 
        self.use_residual = use_residual 
        self.num_repeats = num_repeats 
      
    # Defining forward pass 
    def forward(self, x): 
        for layer in self.layers: 
            residual = x 
            x = layer(x) 
            if self.use_residual: 
                x = x + residual 
        return x

class SSPblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5,9,13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = CNNBlock(in_channels, hidden_channels , kernel_size=1, stride=1)
        self.mx = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = CNNBlock(conv2_channels, out_channels , kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.mx], dim=1)
        x = self.conv2(x)
        return x

class Darknet53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1_res1 = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1)
        )
        in_channels = 64
        self.block2_res2 = nn.Sequential(
            CNNBlock(in_channels, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2)

        )
        in_channels *=2
        self.block3_res8 = nn.Sequential(
            CNNBlock(in_channels, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8)
        )
        in_channels *=2

        self.block4_res8 = nn.Sequential(
            CNNBlock(in_channels, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8)  
        )
        in_channels *=2

        self.block5_res4 = nn.Sequential(
            CNNBlock(in_channels, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),  
            self.create_spp_block()
        )


    def forward(self, x):

        x = self.block1_res1(x) # 128x128
        x = self.block2_res2(x) # 64x64
        x = self.block3_res8(x) # 32x32
        l_output = x
        x = self.block4_res8(x) # 16x16
        h_output = x
        x = self.block5_res4(x) # 8x8
        ssp_output = x
        return l_output, h_output, ssp_output

    def create_spp_block(self):
        outputs = nn.Sequential(
            CNNBlock(1024, 512, kernel_size=1, stride=1),
            CNNBlock(512, 1024, kernel_size=3, stride=1),
            SSPblock(1024, 512),
            CNNBlock(512, 1024, kernel_size=3, stride=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1)
        )
        return outputs

if __name__=="__main__":
    _input = torch.randn(1,3,640,640)
    model = Darknet53()
    l_out, h_out, spp_out = model(_input)
    print(l_out.shape)
    print(h_out.shape)
    print(spp_out.shape)