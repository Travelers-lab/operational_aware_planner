import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Dict, Any

class MapCompletionNet(nn.Module):
    """
    U-Net like architecture for map completion tasks.
    Encoder-decoder structure with skip connections.
    """

    def __init__(self, init_channels=8, bottleneck_channels=32):
        super(MapCompletionNet, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels * 2, init_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_channels * 2, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling path)
        self.up1 = nn.ConvTranspose2d(bottleneck_channels, init_channels * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, init_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels * 2, init_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(init_channels * 2, init_channels, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(init_channels * 2, init_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.final_conv = nn.Conv2d(init_channels, 1, kernel_size=1)

        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        """Forward pass through the network"""
        # Encoder path
        e1 = self.enc1(x)  # [B, init_channels, H, W]
        p1 = self.pool1(e1)  # [B, init_channels, H/2, W/2]

        e2 = self.enc2(p1)  # [B, init_channels*2, H/2, W/2]
        p2 = self.pool2(e2)  # [B, init_channels*2, H/4, W/4]

        # Bottleneck layer
        b = self.bottleneck(p2)  # [B, bottleneck_channels, H/4, W/4]

        # Decoder path with skip connections
        u1 = self.up1(b)  # [B, init_channels*2, H/2, W/2]
        # Skip connection from encoder
        c1 = torch.cat([e2, u1], dim=1)  # [B, bottleneck_channels, H/2, W/2]
        d1 = self.dec1(c1)  # [B, init_channels*2, H/2, W/2]

        u2 = self.up2(d1)  # [B, init_channels, H, W]
        # Skip connection from encoder
        c2 = torch.cat([e1, u2], dim=1)  # [B, init_channels*2, H, W]
        d2 = self.dec2(c2)  # [B, init_channels, H, W]

        # Output
        out = self.final_conv(d2)
        return torch.sigmoid(out)

    def initialize_weights(self):
        """
        Initialize model weights using appropriate methods.
        Uses Kaiming initialization for convolutional layers with ReLU activation.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming initialization for ReLU activations
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def load_checkpoint(self, checkpoint_dict: Dict[str, Any]):
        """
        Load model parameters from a checkpoint dictionary.

        Args:
            checkpoint_dict: Dictionary containing model parameters under 'params' key
        """
        if 'params' in checkpoint_dict:
            self.load_state_dict(checkpoint_dict['params'])
            print("Model parameters loaded successfully.")
        else:
            print("Warning: 'params' key not found in checkpoint. Model not loaded.")




# 示例使用
if __name__ == "__main__":
    # init model
    model = MapCompletionNet()

    # print total param number
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"param number: {count_parameters(model)}")
    # inputs: [batch_size, 1, 100, 100]
    input_map = torch.rand(1, 1, 100, 100)

    # forward computing
    output_map = model(input_map)

    # binary output
    binary_output = (output_map > 0.5).float()

    print("inputs shape:", input_map.shape)
    print("output shape:", binary_output.shape)