from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, use_batch_normalization: bool = True) -> None:
		super().__init__()
		layers = [
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_normalization),
		]
		if use_batch_normalization:
			layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nn.ReLU(inplace=True))
		layers.append(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_normalization),
		)
		if use_batch_normalization:
			layers.append(nn.BatchNorm2d(out_channels))
		layers.append(nn.ReLU(inplace=True))
		self.layers = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


class DownBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, use_batch_normalization: bool = True) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			DoubleConv(in_channels, out_channels, use_batch_normalization=use_batch_normalization),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


class UpBlock(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batch_normalization: bool = True) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, use_batch_normalization=use_batch_normalization)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		diff_y = skip.size(2) - x.size(2)
		diff_x = skip.size(3) - x.size(3)
		x = nn.functional.pad(
			x,
			[
				diff_x // 2,
				diff_x - diff_x // 2,
				diff_y // 2,
				diff_y - diff_y // 2,
			],
		)
		x = torch.cat((skip, x), dim=1)
		return self.conv(x)


class UNet(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, use_batch_normalization: bool = True) -> None:
		super().__init__()
		self.stem = DoubleConv(in_channels, base_channels, use_batch_normalization=use_batch_normalization)
		self.down1 = DownBlock(base_channels, base_channels * 2, use_batch_normalization=use_batch_normalization)
		self.down2 = DownBlock(base_channels * 2, base_channels * 4, use_batch_normalization=use_batch_normalization)
		self.down3 = DownBlock(base_channels * 4, base_channels * 8, use_batch_normalization=use_batch_normalization)
		self.down4 = DownBlock(base_channels * 8, base_channels * 16, use_batch_normalization=use_batch_normalization)
		self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8, use_batch_normalization=use_batch_normalization)
		self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, use_batch_normalization=use_batch_normalization)
		self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, use_batch_normalization=use_batch_normalization)
		self.up4 = UpBlock(base_channels * 2, base_channels, base_channels, use_batch_normalization=use_batch_normalization)
		self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		skip1 = self.stem(x)
		skip2 = self.down1(skip1)
		skip3 = self.down2(skip2)
		skip4 = self.down3(skip3)
		bottleneck = self.down4(skip4)
		x = self.up1(bottleneck, skip4)
		x = self.up2(x, skip3)
		x = self.up3(x, skip2)
		x = self.up4(x, skip1)
		return self.head(x)
