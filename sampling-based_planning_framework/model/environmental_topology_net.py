import torch
import torch.nn as nn
import torch.nn.functional as F


class MapCompletionNet(nn.Module):
    """
    Lightweight CNN for grid map completion and obstacle prediction.

    Input:  Binary grid tensor of shape (batch_size, 1, 100, 100)
            with partial obstacle information (1=obstacle, 0=free)
    Output: Probability grid tensor of shape (batch_size, 1, 100, 100)
            with predicted obstacle occupancy probabilities [0, 1]

    Architecture features:
    - Encoder-decoder structure with skip connections
    - Depthwise separable convolutions for efficiency
    - Multi-scale feature fusion
    - Output probability map with sigmoid activation
    - Optimized for 100x100 grid processing
    """

    def __init__(self):
        super().__init__()

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32, downsample=True)
        self.enc3 = self._conv_block(32, 64, downsample=True)

        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            self._dilated_conv(64, 128, dilation=2),
            self._dilated_conv(128, 128, dilation=3),
            self._dilated_conv(128, 64, dilation=1)
        )

        # Decoder (upsampling path) with skip connections
        self.dec3 = self._upconv_block(64, 32)
        self.dec2 = self._upconv_block(32 + 32, 16)  # Skip connection from enc2
        self.dec1 = self._upconv_block(16 + 16, 8)  # Skip connection from enc1

        # Probability prediction head
        self.prob_head = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_ch, out_ch, downsample=False):
        """Create a convolutional block with optional downsampling"""
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if downsample:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def _dilated_conv(self, in_ch, out_ch, dilation):
        """Create a dilated convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch, out_ch):
        """Create an upsampling block with transposed convolution"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1_out = self.enc1(x)  # [B, 16, 100, 100]
        enc2_out = self.enc2(enc1_out)  # [B, 32, 50, 50]
        enc3_out = self.enc3(enc2_out)  # [B, 64, 25, 25]

        # Bottleneck with expanded receptive field
        bottleneck_out = self.bottleneck(enc3_out)  # [B, 64, 25, 25]

        # Decoder path with skip connections
        dec3_out = self.dec3(bottleneck_out)  # [B, 32, 50, 50]
        dec2_in = torch.cat([dec3_out, enc2_out], dim=1)  # Skip connection
        dec2_out = self.dec2(dec2_in)  # [B, 16, 100, 100]
        dec1_in = torch.cat([dec2_out, enc1_out], dim=1)  # Skip connection
        dec1_out = self.dec1(dec1_in)  # [B, 8, 100, 100]

        # Probability prediction
        prob_map = self.prob_head(dec1_out)  # [B, 1, 100, 100]
        return prob_map


# Utility function for map processing
def process_map(input_map, model, threshold=0.5):
    """
    Process incomplete map and generate completed obstacle probability map

    Args:
        input_map: Incomplete binary map tensor [1, 100, 100]
        model: Trained MapCompletionNet
        threshold: Probability threshold for binary output

    Returns:
        prob_map: Predicted probability map [100, 100]
        binary_map: Thresholded binary map [100, 100]
    """
    model.eval()
    with torch.no_grad():
        # Add batch and channel dimensions
        input_tensor = input_map.unsqueeze(0).unsqueeze(0).float()

        # Predict probabilities
        prob_output = model(input_tensor)

        # Remove batch and channel dimensions
        prob_map = prob_output.squeeze(0).squeeze(0)

        # Create binary map
        binary_map = (prob_map > threshold).float()

    return prob_map, binary_map