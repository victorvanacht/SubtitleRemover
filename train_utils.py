from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader

from data_generator import build_dataloader


def _tensor_rgb_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
	array = image_tensor.detach().cpu().clamp(0.0, 1.0).numpy()
	array = np.transpose(array, (1, 2, 0))
	return (array * 255.0).round().astype(np.uint8)


def _tensor_mask_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
	array = mask_tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
	return (array * 255.0).round().astype(np.uint8)


def build_example_grid(batch: dict[str, torch.Tensor | list[str] | list[float] | list[int]], max_examples: int) -> Image.Image:
	features = batch["features"]
	if not isinstance(features, torch.Tensor):
		raise TypeError("Expected batch['features'] to be a torch.Tensor")

	labels = ["Channels 1-3: original", "Channel 4: mask", "Channels 5-7: subtitled"]
	try:
		font = ImageFont.truetype("arial.ttf", 18)
	except OSError:
		font = ImageFont.load_default()

	count = max(1, min(max_examples, features.shape[0]))
	first = features[0]
	h, w = first.shape[1], first.shape[2]
	label_h = 34
	gap = 8
	panel_w = w
	panel_h = label_h + h
	row_w = panel_w * 3 + gap * 4
	row_h = panel_h + gap * 2
	grid = Image.new("RGB", (row_w, row_h * count), (20, 20, 20))

	for idx in range(count):
		sample = features[idx]
		original = _tensor_rgb_to_uint8(sample[0:3])
		mask = _tensor_mask_to_uint8(sample[3:4])
		subtitled = _tensor_rgb_to_uint8(sample[4:7])

		panels = [
			Image.fromarray(original, mode="RGB"),
			Image.fromarray(mask, mode="L").convert("RGB"),
			Image.fromarray(subtitled, mode="RGB"),
		]

		for panel_idx, panel in enumerate(panels):
			x0 = gap + panel_idx * (panel_w + gap)
			y0 = idx * row_h + gap
			grid.paste(panel, (x0, y0 + label_h))

			draw = ImageDraw.Draw(grid)
			draw.text((x0 + 2, y0 + 8), labels[panel_idx], fill=(230, 230, 230), font=font)

	return grid


def preview_generated_samples(
	batch: dict[str, torch.Tensor | list[str] | list[float] | list[int]],
	max_examples: int,
	save_path: Path | None = None,
) -> None:
	grid = build_example_grid(batch=batch, max_examples=max_examples)
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		grid.save(save_path)
		print(f"Saved example preview to: {save_path}")

	grid.show(title="Generated training data channels")


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_loaders(args) -> tuple[DataLoader, DataLoader]:
	image_size = (args.width, args.height)
	train_loader = build_dataloader(
		image_root=args.train_root,
		batch_size=args.batch_size,
		image_size=image_size,
		shuffle=True,
		num_workers=args.num_workers,
		random_seed=args.seed,
		max_samples=args.max_train_samples,
	)
	validate_loader = build_dataloader(
		image_root=args.validate_root,
		batch_size=args.batch_size,
		image_size=image_size,
		shuffle=False,
		num_workers=args.num_workers,
		random_seed=args.seed + 1,
		max_samples=args.max_validation_samples,
	)
	return train_loader, validate_loader


def save_checkpoint(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	args,
	epoch: int,
	best_val_loss: float,
	save_as_best: bool = False,
) -> None:
	# Save as last checkpoint (includes optimizer state for resuming training)
	last_checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"best_val_loss": best_val_loss,
		"image_size": [args.width, args.height],
		"base_channels": args.base_channels,
	}
	args.save_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(last_checkpoint, args.save_path)
	
	# Save as best checkpoint if validation loss improved (excludes optimizer state for smaller file size)
	if save_as_best:
		best_checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"best_val_loss": best_val_loss,
			"image_size": [args.width, args.height],
			"base_channels": args.base_channels,
		}
		args.best_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(best_checkpoint, args.best_path)


def try_resume_from_checkpoint(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	checkpoint_path: Path,
	device: torch.device,
) -> tuple[int, float]:
	if not checkpoint_path.exists():
		return 1, float("inf")

	checkpoint = torch.load(checkpoint_path, map_location=device)
	model_state = checkpoint.get("model_state_dict")
	optimizer_state = checkpoint.get("optimizer_state_dict")
	if model_state is None or optimizer_state is None:
		raise KeyError("Checkpoint is missing 'model_state_dict' or 'optimizer_state_dict'.")

	model.load_state_dict(model_state)
	optimizer.load_state_dict(optimizer_state)
	last_epoch = int(checkpoint.get("epoch", 0))
	best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
	start_epoch = last_epoch + 1
	print(f"Resumed training from checkpoint: {checkpoint_path} (last_epoch={last_epoch})")
	return start_epoch, best_val_loss
