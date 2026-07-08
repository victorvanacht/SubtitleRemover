from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data_generator import build_dataloader
from train import SUBTITLED_CHANNEL_END, SUBTITLED_CHANNEL_START, UNetMaskEstimator


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run U-Net inference and save a visual comparison grid.")
	parser.add_argument("--data-root", type=Path, default=Path(".\\cocodataset\\validate"))
	parser.add_argument("--checkpoint", type=Path, default=Path(".\\artifacts\\best.pt"))
	parser.add_argument("--output", type=Path, default=Path(".\\artifacts\\inference_preview.png"))
	parser.add_argument("--num-examples", type=int, default=4)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--threshold", type=float, default=0.5)
	return parser.parse_args()


def _tensor_rgb_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
	array = image_tensor.detach().cpu().clamp(0.0, 1.0).numpy()
	array = np.transpose(array, (1, 2, 0))
	return (array * 255.0).round().astype(np.uint8)


def _tensor_mask_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
	array = mask_tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
	return (array * 255.0).round().astype(np.uint8)


def load_model(checkpoint_path: Path, device: torch.device) -> UNetMaskEstimator:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location=device)
	base_channels = int(checkpoint.get("base_channels", 64))
	model = UNetMaskEstimator(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
	model_state = checkpoint.get("model_state_dict")
	if model_state is None:
		raise KeyError("Checkpoint is missing 'model_state_dict'.")
	model.load_state_dict(model_state)
	model.eval()
	return model


def build_preview_grid(
	subtitled_images: torch.Tensor,
	true_masks: torch.Tensor,
	pred_masks: torch.Tensor,
	max_examples: int,
) -> Image.Image:
	labels = ["Input (subtitled RGB)", "Ground-truth mask", "Estimated mask"]
	count = max(1, min(max_examples, subtitled_images.shape[0]))

	first = subtitled_images[0]
	h, w = int(first.shape[1]), int(first.shape[2])
	label_h = 34
	gap = 8
	panel_w = w
	panel_h = label_h + h
	row_w = panel_w * 3 + gap * 4
	row_h = panel_h + gap * 2
	grid = Image.new("RGB", (row_w, row_h * count), (20, 20, 20))

	try:
		font = ImageFont.truetype("arial.ttf", 18)
	except OSError:
		font = ImageFont.load_default()

	for idx in range(count):
		subtitled = _tensor_rgb_to_uint8(subtitled_images[idx])
		true_mask = _tensor_mask_to_uint8(true_masks[idx])
		pred_mask = _tensor_mask_to_uint8(pred_masks[idx])

		panels = [
			Image.fromarray(subtitled, mode="RGB"),
			Image.fromarray(true_mask, mode="L").convert("RGB"),
			Image.fromarray(pred_mask, mode="L").convert("RGB"),
		]

		for panel_idx, panel in enumerate(panels):
			x0 = gap + panel_idx * (panel_w + gap)
			y0 = idx * row_h + gap
			grid.paste(panel, (x0, y0 + label_h))
			draw = ImageDraw.Draw(grid)
			draw.text((x0 + 2, y0 + 8), labels[panel_idx], fill=(230, 230, 230), font=font)

	return grid


def _next_available_output_path(path: Path) -> Path:
	if not path.exists():
		return path

	stem = path.stem
	suffix = path.suffix
	parent = path.parent
	index = 1
	while True:
		candidate = parent / f"{stem} ({index}){suffix}"
		if not candidate.exists():
			return candidate
		index += 1


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(args.checkpoint, device)

	dataloader = build_dataloader(
		image_root=args.data_root,
		batch_size=max(args.batch_size, args.num_examples),
		image_size=(args.width, args.height),
		shuffle=True,
		num_workers=args.num_workers,
		random_seed=args.seed,
	)

	batch = next(iter(dataloader))
	features = batch["features"]
	true_masks = batch["mask"]
	if not isinstance(features, torch.Tensor) or not isinstance(true_masks, torch.Tensor):
		raise TypeError("Expected tensor batches for 'features' and 'mask'.")

	subtitled_images = features[:, SUBTITLED_CHANNEL_START:SUBTITLED_CHANNEL_END]
	with torch.no_grad():
		logits = model(subtitled_images.to(device, non_blocking=True))
		probs = torch.sigmoid(logits).cpu()
	pred_masks = (probs >= args.threshold).float()

	grid = build_preview_grid(
		subtitled_images=subtitled_images,
		true_masks=true_masks,
		pred_masks=pred_masks,
		max_examples=args.num_examples,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	output_path = _next_available_output_path(args.output)
	grid.save(output_path)
	print(f"Saved inference preview to: {output_path}")


if __name__ == "__main__":
	main()
