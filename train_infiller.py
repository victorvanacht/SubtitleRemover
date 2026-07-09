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
	preview_generated_samples,
	set_seed,
	build_loaders,
	save_checkpoint,
	try_resume_from_checkpoint,
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


def extract_model_inputs(features: torch.Tensor) -> torch.Tensor:
	return features[:, SUBTITLED_CHANNEL_START:SUBTITLED_CHANNEL_END]


def extract_targets(features: torch.Tensor) -> torch.Tensor:
	return features[:, ORIGINAL_CHANNEL_START:ORIGINAL_CHANNEL_END]


def extract_mask(features: torch.Tensor) -> torch.Tensor:
	return features[:, MASK_CHANNEL_START:MASK_CHANNEL_END]


def masked_mae(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
	"""Calculate Mean Absolute Error only on masked (affected) pixels."""
	mae = torch.abs(predictions - targets)
	masked_mae_val = (mae * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + epsilon)
	return masked_mae_val.mean()


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None,
	loss_fn: nn.Module,
	scaler: torch.amp.GradScaler | None,
	use_amp: bool,
) -> tuple[float, float]:
	is_training = optimizer is not None
	model.train(mode=is_training)
	total_loss = 0.0
	total_mae = 0.0
	total_examples = 0

	context_manager = torch.enable_grad if is_training else torch.no_grad
	with context_manager():
		for batch in loader:
			features = batch["features"]
			if not isinstance(features, torch.Tensor):
				raise TypeError("Expected batch['features'] to be a torch.Tensor")

			inputs = extract_model_inputs(features).to(device, non_blocking=True)
			targets = extract_targets(features).to(device, non_blocking=True)
			mask = extract_mask(features).to(device, non_blocking=True)
			batch_size = inputs.size(0)

			if is_training:
				optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(device_type=device.type, enabled=use_amp):
				predictions = model(inputs)
				# Apply mask to loss to focus on affected regions
				masked_predictions = predictions * mask
				masked_targets = targets * mask
				loss = loss_fn(masked_predictions, masked_targets)

			if is_training and optimizer is not None:
				if scaler is not None:
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
				else:
					loss.backward()
					optimizer.step()

			total_loss += loss.detach().item() * batch_size
			total_mae += masked_mae(predictions.detach(), targets, mask).item() * batch_size
			total_examples += batch_size

	if total_examples == 0:
		return math.nan, math.nan
	return total_loss / total_examples, total_mae / total_examples


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = args.amp and device.type == "cuda"
	train_loader, validate_loader = build_loaders(args)

	model = UNet(in_channels=4, out_channels=3, base_channels=args.base_channels).to(device)
	loss_fn = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scaler = torch.amp.GradScaler("cuda") if use_amp else None
	start_epoch, best_val_loss = try_resume_from_checkpoint(
		model=model,
		optimizer=optimizer,
		checkpoint_path=args.save_path,
		device=device,
	)

	print(f"Training on device: {device}")
	print(f"Training samples: {len(train_loader.dataset)}")
	print(f"Validation samples: {len(validate_loader.dataset)}")
	print(f"Model input channels: mask + subtitled RGB ({MASK_CHANNEL_START}-{SUBTITLED_CHANNEL_END - 1})")
	print(f"Model output: original RGB ({ORIGINAL_CHANNEL_START}-{ORIGINAL_CHANNEL_END - 1})")

	if args.show_examples > 0:
		preview_batch = next(iter(train_loader))
		preview_generated_samples(
			batch=preview_batch,
			max_examples=args.show_examples,
			save_path=args.save_example_grid,
		)

	end_epoch = start_epoch + args.epochs - 1
	for epoch in range(start_epoch, end_epoch + 1):
		train_loss, train_mae = run_epoch(
			model=model,
			loader=train_loader,
			device=device,
			optimizer=optimizer,
			loss_fn=loss_fn,
			scaler=scaler,
			use_amp=use_amp,
		)
		val_loss, val_mae = run_epoch(
			model=model,
			loader=validate_loader,
			device=device,
			optimizer=None,
			loss_fn=loss_fn,
			scaler=None,
			use_amp=use_amp,
		)

		print(
			f"Epoch {epoch:03d}/{end_epoch:03d} "
			f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
			f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
		)

		# Always save as last checkpoint
		save_checkpoint(model=model, optimizer=optimizer, args=args, epoch=epoch, best_val_loss=best_val_loss, save_as_best=False)
		print(f"Saved checkpoint to: {args.save_path}")
		
		# Save as best checkpoint if validation loss improved
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_checkpoint(model=model, optimizer=optimizer, args=args, epoch=epoch, best_val_loss=best_val_loss, save_as_best=True)
			print(f"Saved best checkpoint to: {args.best_path}")


if __name__ == "__main__":
	main()
