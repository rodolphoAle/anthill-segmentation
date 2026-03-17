"""U-Net architecture for semantic segmentation.

The layer topology is preserved **exactly** so that weight files
(``u_net.pth`` / ``best_model_params.pth``) remain loadable without
any key-mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Encoder-decoder U-Net with skip connections.

    Args:
        n_channels: Number of input channels (default 3 for RGB).
        n_classes: Number of output segmentation classes.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 2) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def down(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch),
            )

        def up(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                double_conv(in_ch, out_ch),
            )

        #  Encoder 
        self.inc = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        #  Decoder 
        self.up1 = up(1024, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.outc = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Logits tensor ``(B, n_classes, H, W)``.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)

        return self.outc(x)
