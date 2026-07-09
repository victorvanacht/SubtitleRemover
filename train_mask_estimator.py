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


SUBTITLED_CHANNEL_START = 4
SUBTITLED_CHANNEL_END = 7


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a U-Net that predicts subtitle masks from subtitled images.")
	parser.add_argument("--train-root", type=Path, default=Path(".\\cocodataset\\train"))
	parser.add_argument("--validate-root", type=Path, default=Path(".\\cocodataset\\validate"))
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--save-path", type=Path, default=Path(".\\artifacts\\mask_estimator_last.pt"))
	parser.add_argument("--best-path", type=Path, default=Path(".\\artifacts\\mask_estimator_best.pt"))
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
		default=Path(".\\example.png"),
		help="Optional output file path for the example grid preview.",
	)
	return parser.parse_args()


def extract_model_inputs(features: torch.Tensor) -> torch.Tensor:
	return features[:, SUBTITLED_CHANNEL_START:SUBTITLED_CHANNEL_END]


def extract_targets(batch: dict[str, torch.Tensor | list]) -> torch.Tensor:
	"""Extract targets from batch."""
	mask = batch["mask"]
	if not isinstance(mask, torch.Tensor):
		raise TypeError("Expected batch['mask'] to be a torch.Tensor")
	return mask


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	preds = (probs >= 0.5).float()
	intersection = (preds * targets).sum(dim=(1, 2, 3))
	denominator = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
	return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def main() -> None:
	args = parse_args()

	config = TrainingConfig(
		model_name="Mask Estimator",
		description="Train a U-Net that predicts subtitle masks from subtitled images.",
		loss_fn=nn.BCEWithLogitsLoss(),
		in_channels=3,
		out_channels=1,
		extract_model_inputs=lambda batch: extract_model_inputs(batch["features"]),
		extract_targets=extract_targets,
		compute_metric=dice_score_from_logits,
		metric_name="dice",
		info_text=f"Model input channels: subtitled RGB only ({SUBTITLED_CHANNEL_START + 1}-{SUBTITLED_CHANNEL_END})",
	)

	train_model(config, args)


if __name__ == "__main__":
	main()
