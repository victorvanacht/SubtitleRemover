from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create subtitle-removal training samples.")
	parser.add_argument(
		"--image-root",
		type=Path,
		default=Path(".\\cocodataset\\train"),
		help="Folder containing clean source images.",
	)
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--show-examples",
		type=int,
		default=2,
		help="If > 0, show this many samples split into channels 1-3, 4, and 5-7.",
	)
	parser.add_argument(
		"--save-example-grid",
		type=Path,
		default=".\\example.png",
		help="Optional output file path to save the example grid image.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	dataloader = build_dataloader(
		image_root=args.image_root,
		batch_size=args.batch_size,
		image_size=(args.width, args.height),
		num_workers=args.num_workers,
		random_seed=args.seed,
	)

	batch = next(iter(dataloader))
	print(f"Samples found: {len(dataloader.dataset)}")
	print(f"Feature batch shape: {tuple(batch['features'].shape)}")
	print(f"Target batch shape: {tuple(batch['target'].shape)}")
	print(f"Mask batch shape: {tuple(batch['mask'].shape)}")

	if args.show_examples > 0:
		preview_generated_samples(
			batch=batch,
			max_examples=args.show_examples,
			save_path=args.save_example_grid,
		)
	print(f"First sample text: {batch['text'][0]}")
	print(f"First sample image: {batch['image_path'][0]}")


if __name__ == "__main__":
	main()
