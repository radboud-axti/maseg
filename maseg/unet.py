# from pl_bolts.models.vision.unet import DoubleConv, Up, Down
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np


class Unet(pl.LightningModule):
    """
    Based on pl_bolts.models.vision.unet.UNet
    Args:
        input_channels: Number of channels in input images (default 1)
        depth: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
        self,
        input_channels: int = 1,
        depth: int = 5,
        features_start: int = 64,
        bilinear: bool = True,
        model_name="model",
    ):

        if depth < 1:
            raise ValueError(f"num_layers = {depth}, expected: num_layers > 0")

        super().__init__()

        self.depth = depth
        layers = [DoubleConv(input_channels, features_start)]  # initial layer
        feats = features_start
        for _ in range(depth - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        for _ in range(depth - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2
        layers.append(nn.Sequential(nn.Conv2d(feats, 3, kernel_size=1), nn.Dropout(inplace=True)))
        self.layers = nn.ModuleList(layers)
        self.num_classes = 3
        self.model_name = model_name

    def forward(self, x, pad_tensor=False, softmax=False):
        output_shape = x.shape
        if pad_tensor:
            required_input_size = self.get_input_size(x.shape[2:4])
            x = self.pad_tensor(x, required_input_size)

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.depth]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.depth: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        y_hat = self.layers[-1](xi[-1])
        if pad_tensor:
            y_hat = self.crop_tensor(y_hat, output_shape)
        if softmax:
            y_hat = nn.functional.softmax(y_hat, dim=1)
        return y_hat

    def get_output_size(self, input_size):
        """Gets the size of an output image for a given input size

        Args:
            input_size (tuple or int): Input size of image. If int, square image is made.

        """
        assert isinstance(input_size, (int, tuple))
        if isinstance(input_size, int):
            output_size = np.array((input_size, input_size))
        else:
            assert len(input_size) == 2
            output_size = np.array(input_size)

        # Input layer
        output_size = output_size - 4

        # Down path
        for _ in self.layers[1: self.depth]:
            output_size = np.floor(output_size / 2) - 4

        # Up path
        for _ in self.layers[self.depth: -1]:
            output_size = output_size * 2 - 4

        return int(output_size[0]), int(output_size[1])

    def get_input_size(self, output_size):
        """Calculates the minimal required input_size for a desired output size.
        If the given output size can not be reached it returns an input size that gives the smallest output size that
        is larger than the given output_size

        Args:
            output_size (tuple or int): Output size the network. If int, square image is made.

        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            input_size = np.array((output_size, output_size))
        else:
            assert len(output_size) == 2
            input_size = np.array(output_size)

        # Up path
        for _ in self.layers[self.depth: -1]:
            input_size = np.ceil((input_size + 4) / 2)

        # Down path
        for _ in self.layers[1: self.depth]:
            input_size = (input_size + 4) * 2

        # Input layer
        input_size = input_size + 4

        return int(input_size[0]), int(input_size[1])

    @staticmethod
    def crop_tensor(tensor, target_shape):
        original_shape = tensor.shape
        diff_y = (original_shape[2] - target_shape[2]) // 2
        diff_x = (original_shape[3] - target_shape[3]) // 2
        tensor = tensor[
            :,
            :,
            diff_y: (diff_y + target_shape[2]),
            diff_x: (diff_x + target_shape[3]),
        ]

        return tensor

    @staticmethod
    def pad_tensor(tensor, target_shape):
        original_shape = tensor.shape
        target_shape = target_shape[-2:]

        diff_y = (target_shape[0] - original_shape[2]) // 2
        diff_y_2 = diff_y if ((target_shape[0] - original_shape[2]) / 2) % 1 == 0 else diff_y + 1
        diff_x = (target_shape[1] - original_shape[3]) // 2
        diff_x_2 = diff_x if ((target_shape[1] - original_shape[3]) / 2) % 1 == 0 else diff_x + 1

        padding_layer = nn.ZeroPad2d((diff_x, diff_x_2, diff_y, diff_y_2))
        tensor = padding_layer(tensor)

        return tensor


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = functional.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
