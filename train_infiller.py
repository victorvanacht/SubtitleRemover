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


ORIGINAL_CHANNEL_START = 0
ORIGINAL_CHANNEL_END = 3
MASK_CHANNEL_START = 3
MASK_CHANNEL_END = 4
SUBTITLED_CHANNEL_START = 3
SUBTITLED_CHANNEL_END = 7


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


class DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


class DownBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			DoubleConv(in_channels, out_channels),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


class UpBlock(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

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


class UNetPixelInfiller(nn.Module):
	def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64) -> None:
		super().__init__()
		self.stem = DoubleConv(in_channels, base_channels)
		self.down1 = DownBlock(base_channels, base_channels * 2)
		self.down2 = DownBlock(base_channels * 2, base_channels * 4)
		self.down3 = DownBlock(base_channels * 4, base_channels * 8)
		self.down4 = DownBlock(base_channels * 8, base_channels * 16)
		self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
		self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
		self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
		self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
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


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
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


def save_checkpoint(
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	args: argparse.Namespace,
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


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = args.amp and device.type == "cuda"
	train_loader, validate_loader = build_loaders(args)

	model = UNetPixelInfiller(in_channels=4, out_channels=3, base_channels=args.base_channels).to(device)
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
