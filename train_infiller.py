from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader

from data_generator import build_dataloader
from unet_utils import UNet
from train_utils import (
	TrainingConfig,
	train_model,
)


ORIGINAL_CHANNEL_START = 0
ORIGINAL_CHANNEL_END = 3
MASK_CHANNEL_START = 3
MASK_CHANNEL_END = 4
SUBTITLED_CHANNEL_START = 3
SUBTITLED_CHANNEL_END = 7


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a U-Net that inpaints original pixels from subtitled images.")
	parser.add_argument("--train-root", type=Path, default=Path(".\\cocodataset\\train"), help="Path to training dataset directory.")
	parser.add_argument("--validate-root", type=Path, default=Path(".\\cocodataset\\validate"), help="Path to validation dataset directory.")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation (default: 8).")
	parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
	parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam optimizer learning rate (default: 0.001).")
	parser.add_argument("--height", type=int, default=256, help="Input image height in pixels (default: 256).")
	parser.add_argument("--width", type=int, default=256, help="Input image width in pixels (default: 256).")
	parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loading (default: 0, set to >0 for parallel loading).")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility (default: 1337).")
	parser.add_argument("--save-path", type=Path, default=Path(".\\artifacts\\infiller_last.pt"), help="Path to save last checkpoint with optimizer state for resuming training.")
	parser.add_argument("--best-path", type=Path, default=Path(".\\artifacts\\infiller_best.pt"), help="Path to save best checkpoint (lowest validation loss, no optimizer state).")
	parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels in U-Net (default: 64).")
	parser.add_argument("--max-train-samples", type=int, default=1000, help="Maximum number of training samples to use (default: 1000).")
	parser.add_argument("--max-validation-samples", type=int, default=1000, help="Maximum number of validation samples to use (default: 1000).")
	parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training when CUDA is available.")
	parser.add_argument(
		"--show-examples",
		type=int,
		default=0,
		help="If > 0, show this many samples split into channels 1-3, 4, and 5-7 before training.",
	)
	parser.add_argument(
		"--save-example-grid",
		type=Path,
		default=Path(".\\example_infiller.png"),
		help="Optional output file path for the example grid preview.",
	)
	return parser.parse_args()


def extract_model_inputs(batch: dict[str, torch.Tensor | list]) -> torch.Tensor:
	features = batch["features"]
	if not isinstance(features, torch.Tensor):
		raise TypeError("Expected batch['features'] to be a torch.Tensor")
	return features[:, SUBTITLED_CHANNEL_START:SUBTITLED_CHANNEL_END]


def extract_targets(batch: dict[str, torch.Tensor | list]) -> torch.Tensor:
	features = batch["features"]
	if not isinstance(features, torch.Tensor):
		raise TypeError("Expected batch['features'] to be a torch.Tensor")
	return features[:, ORIGINAL_CHANNEL_START:ORIGINAL_CHANNEL_END]


def extract_mask_from_batch(batch: dict[str, torch.Tensor | list]) -> torch.Tensor:
	features = batch["features"]
	if not isinstance(features, torch.Tensor):
		raise TypeError("Expected batch['features'] to be a torch.Tensor")
	return features[:, MASK_CHANNEL_START:MASK_CHANNEL_END]


def masked_loss_wrapper(predictions: torch.Tensor, batch: dict) -> torch.Tensor:
	"""Weighted L1 Loss: masked pixels (100x weight) + other pixels (1x weight).
	
	This balances learning to fill masked regions while maintaining quality in unaffected areas.
	Masked pixels get 100x higher per-pixel weight since they're much less numerous.
	"""
	device = predictions.device
	targets = extract_targets(batch).to(device, non_blocking=True)
	mask = extract_mask_from_batch(batch).to(device, non_blocking=True)
	
	# Compute per-pixel L1 loss
	pixel_loss = torch.abs(predictions - targets)
	
	# Weight: masked pixels get 100x weight, other pixels get 1x weight
	weight_map = mask * 100.0 + (1.0 - mask) * 1.0
	
	# Apply weights and compute mean
	weighted_loss = (pixel_loss * weight_map).mean()
	return weighted_loss


def compute_masked_mae_metric(predictions: torch.Tensor, batch: dict) -> torch.Tensor:
	"""Compute MAE metric for the generic training loop."""
	device = predictions.device
	targets = extract_targets(batch).to(device, non_blocking=True)
	mask = extract_mask_from_batch(batch).to(device, non_blocking=True)
	epsilon = 1e-6
	mae = torch.abs(predictions - targets)
	masked_mae_val = (mae * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + epsilon)
	return masked_mae_val.mean()


def main() -> None:
	args = parse_args()

	config = TrainingConfig(
		model_name="Pixel Infiller",
		description="Train a U-Net that inpaints original pixels from subtitled images.",
		loss_fn=masked_loss_wrapper,  # Custom loss that handles masking
		in_channels=4,
		out_channels=3,
		extract_model_inputs=extract_model_inputs,
		extract_targets=None,  # Loss fn and metric handle full batch
		compute_metric=compute_masked_mae_metric,
		metric_name="mae",
		info_text=(
			f"Model input channels: mask + subtitled RGB ({MASK_CHANNEL_START}-{SUBTITLED_CHANNEL_END - 1})\n"
			f"Model output: original RGB ({ORIGINAL_CHANNEL_START}-{ORIGINAL_CHANNEL_END - 1})"
		),
		use_batch_normalization=False,
	)

	train_model(config, args)


if __name__ == "__main__":
	main()
