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
	parser.add_argument("--train-root", type=Path, default=Path(".\\cocodataset\\train"))
	parser.add_argument("--validate-root", type=Path, default=Path(".\\cocodataset\\validate"))
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--save-path", type=Path, default=Path(".\\artifacts\\infiller_last.pt"))
	parser.add_argument("--best-path", type=Path, default=Path(".\\artifacts\\infiller_best.pt"))
	parser.add_argument("--base-channels", type=int, default=64)
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
	"""L1 Loss that applies masking to focus on affected regions."""
	device = predictions.device
	targets = extract_targets(batch).to(device, non_blocking=True)
	mask = extract_mask_from_batch(batch).to(device, non_blocking=True)
	masked_predictions = predictions * mask
	masked_targets = targets * mask
	loss_fn = nn.L1Loss()
	return loss_fn(masked_predictions, masked_targets)


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
	)

	train_model(config, args)


if __name__ == "__main__":
	main()
