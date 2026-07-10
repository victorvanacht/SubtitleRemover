from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from data_generator import build_dataloader
from train_mask_estimator import SUBTITLED_CHANNEL_END, SUBTITLED_CHANNEL_START
from unet_utils import UNet


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run mask estimator and pixel infiller inference.")
	parser.add_argument("--data-root", type=Path, default=Path(".\\cocodataset\\validate"), help="Path to dataset directory for inference.")
	parser.add_argument("--checkpoint-mask-estimator", type=Path, default=Path(".\\artifacts\\mask_estimator_best.pt"), help="Path to mask estimator checkpoint file.")
	parser.add_argument("--checkpoint-infiller", type=Path, default=Path(".\\artifacts\\infiller_best.pt"), help="Path to pixel infiller checkpoint file.")
	parser.add_argument("--output", type=Path, default=Path(".\\artifacts\\inference_preview.png"), help="Path to save inference preview grid image.")
	parser.add_argument("--num-examples", type=int, default=4, help="Number of examples to display in preview grid (default: 4).")
	parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference (default: 4).")
	parser.add_argument("--height", type=int, default=256, help="Input image height in pixels (default: 256, must match training height).")
	parser.add_argument("--width", type=int, default=256, help="Input image width in pixels (default: 256, must match training width).")
	parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loading (default: 0, set to >0 for parallel loading).")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility (default: 1337).")
	parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing mask predictions (default: 0.5, range: 0.0-1.0).")
	return parser.parse_args()


def _tensor_rgb_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
	array = image_tensor.detach().cpu().clamp(0.0, 1.0).numpy()
	array = np.transpose(array, (1, 2, 0))
	return (array * 255.0).round().astype(np.uint8)


def _tensor_mask_to_uint8(mask_tensor: torch.Tensor) -> np.ndarray:
	array = mask_tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy()
	return (array * 255.0).round().astype(np.uint8)


def blend_images_with_mask(subtitled: torch.Tensor, inpainted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""Blend subtitled and inpainted images based on mask.
	- Where mask is 0 (not affected): use subtitled image
	- Where mask is 1 (affected): use inpainted image
	"""
	blended = subtitled * (1.0 - mask) + inpainted * mask
	return blended


def dilate_mask(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
	"""Dilate the mask by 1 pixel in all directions for each iteration.
	
	Args:
		mask: Tensor of shape (batch_size, 1, height, width)
		iterations: Number of dilation iterations (default: 1 for 1 pixel expansion)
	
	Returns:
		Dilated mask tensor
	"""
	kernel = ndimage.generate_binary_structure(2, 2)  # 3x3 kernel with 8-connectivity
	dilated_mask = mask.clone()
	
	for batch_idx in range(dilated_mask.shape[0]):
		# Extract single image mask (remove batch and channel dims for processing)
		mask_2d = dilated_mask[batch_idx, 0].numpy()
		# Apply dilation
		for _ in range(iterations):
			mask_2d = ndimage.binary_dilation(mask_2d, structure=kernel).astype(np.float32)
		# Put back in tensor
		dilated_mask[batch_idx, 0] = torch.from_numpy(mask_2d)
	
	return dilated_mask


def load_model(checkpoint_mask_estimator_path: Path, device: torch.device) -> UNet:
	if not checkpoint_mask_estimator_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_mask_estimator_path}")

	checkpoint_mask_estimator = torch.load(checkpoint_mask_estimator_path, map_location=device)
	base_channels = int(checkpoint_mask_estimator.get("base_channels", 64))
	model = UNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
	model_state = checkpoint_mask_estimator.get("model_state_dict")
	if model_state is None:
		raise KeyError("checkpoint_mask_estimator is missing 'model_state_dict'.")
	model.load_state_dict(model_state)
	model.eval()
	return model


def load_infiller_model(checkpoint_infiller_path: Path, device: torch.device) -> UNet:
	if not checkpoint_infiller_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_infiller_path}")

	checkpoint_infiller = torch.load(checkpoint_infiller_path, map_location=device)
	base_channels = int(checkpoint_infiller.get("base_channels", 64))
	model = UNet(in_channels=4, out_channels=3, base_channels=base_channels).to(device)
	model_state = checkpoint_infiller.get("model_state_dict")
	if model_state is None:
		raise KeyError("checkpoint_infiller is missing 'model_state_dict'.")
	model.load_state_dict(model_state)
	model.eval()
	return model


def build_preview_grid(
	original_images: torch.Tensor,
	subtitled_images: torch.Tensor,
	true_masks: torch.Tensor,
	pred_masks: torch.Tensor,
	inpainted_images: torch.Tensor,
	blended_images: torch.Tensor,
	max_examples: int,
) -> Image.Image:
	labels = ["Original", "Input (subtitled)", "True mask", "Pred mask", "Inpainted", "Blended"]
	count = max(1, min(max_examples, original_images.shape[0]))

	first = original_images[0]
	h, w = int(first.shape[1]), int(first.shape[2])
	label_h = 34
	gap = 8
	panel_w = w
	panel_h = label_h + h
	row_w = panel_w * 6 + gap * 7
	row_h = panel_h + gap * 2
	grid = Image.new("RGB", (row_w, row_h * count), (20, 20, 20))

	try:
		font = ImageFont.truetype("arial.ttf", 18)
	except OSError:
		font = ImageFont.load_default()

	for idx in range(count):
		original = _tensor_rgb_to_uint8(original_images[idx])
		subtitled = _tensor_rgb_to_uint8(subtitled_images[idx])
		true_mask = _tensor_mask_to_uint8(true_masks[idx])
		pred_mask = _tensor_mask_to_uint8(pred_masks[idx])
		inpainted = _tensor_rgb_to_uint8(inpainted_images[idx])
		blended = _tensor_rgb_to_uint8(blended_images[idx])

		panels = [
			Image.fromarray(original, mode="RGB"),
			Image.fromarray(subtitled, mode="RGB"),
			Image.fromarray(true_mask, mode="L").convert("RGB"),
			Image.fromarray(pred_mask, mode="L").convert("RGB"),
			Image.fromarray(inpainted, mode="RGB"),
			Image.fromarray(blended, mode="RGB"),
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
	
	print("Loading mask estimator model...")
	mask_model = load_model(args.checkpoint_mask_estimator, device)
	
	print("Loading infiller model...")
	infiller_model = load_infiller_model(args.checkpoint_infiller, device)

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

	# Extract channels from features
	original_images = features[:, 0:3]
	subtitled_images = features[:, SUBTITLED_CHANNEL_START:SUBTITLED_CHANNEL_END]
	
	# Step 1: Estimate mask using mask estimator
	print("Estimating masks...")
	with torch.no_grad():
		mask_logits = mask_model(subtitled_images.to(device, non_blocking=True))
		mask_probs = torch.sigmoid(mask_logits).cpu()
	pred_masks = (mask_probs >= args.threshold).float()
	
	# Dilate mask by 1 pixel in all directions for better edge coverage
	print("Dilating masks...")
	pred_masks = dilate_mask(pred_masks, iterations=1)
	
	# Step 2: Inpaint using infiller model (mask + subtitled RGB as input)
	print("Inpainting pixels...")
	infiller_input = torch.cat([pred_masks.to(device, non_blocking=True), 
	                             subtitled_images.to(device, non_blocking=True)], dim=1)
	with torch.no_grad():
		inpainted_images = infiller_model(infiller_input).cpu()
		inpainted_images = inpainted_images.clamp(0.0, 1.0)

	# Step 3: Blend inpainted and subtitled images using the mask
	print("Blending images...")
	blended_images = blend_images_with_mask(subtitled_images, inpainted_images, pred_masks)

	grid = build_preview_grid(
		original_images=original_images,
		subtitled_images=subtitled_images,
		true_masks=true_masks,
		pred_masks=pred_masks,
		inpainted_images=inpainted_images,
		blended_images=blended_images,
		max_examples=args.num_examples,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	output_path = _next_available_output_path(args.output)
	grid.save(output_path)
	print(f"Saved inference preview to: {output_path}")
	grid.show(title="Mask Estimation and Inpainting Results")


if __name__ == "__main__":
	main()
