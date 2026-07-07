from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SUBTITLE_LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class SubtitleSample:
	features: torch.Tensor
	target: torch.Tensor
	mask: torch.Tensor
	alpha: float
	font_size: int
	angle: float
	text: str
	image_path: str


class SubtitleTrainingDataset(Dataset[SubtitleSample]):
	def __init__(
		self,
		image_root: str | Path,
		image_size: tuple[int, int] | None = (256, 256),
		random_seed: int | None = None,
	) -> None:
		self.image_root = Path(image_root)
		self.image_size = image_size
		self.random = random.Random(random_seed)
		self.image_paths = self._collect_image_paths(self.image_root)
		if not self.image_paths:
			raise ValueError(f"No images found under {self.image_root}")

		self.font_paths = self._collect_font_paths()

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, index: int) -> SubtitleSample:
		image_path = self.image_paths[index]
		with Image.open(image_path) as source_image:
			original_image = source_image.convert("RGB")
		if self.image_size is not None:
			original_image = original_image.resize(self.image_size, Image.Resampling.BICUBIC)

		original_array = np.asarray(original_image, dtype=np.uint8)
		subtitle = self._generate_subtitle_assets(original_image.size)

		composite = self._blend_subtitle(
			original_array=original_array,
			mask_array=subtitle["mask_array"],
			color=subtitle["color"],
			alpha=subtitle["alpha"],
		)

		original_tensor = self._to_float_tensor(original_array)
		mask_tensor = torch.from_numpy(subtitle["mask_array"]).unsqueeze(0).float() / 255.0
		composite_tensor = self._to_float_tensor(composite)
		features = torch.cat((original_tensor, mask_tensor, composite_tensor), dim=0)

		return SubtitleSample(
			features=features,
			target=original_tensor,
			mask=mask_tensor,
			alpha=subtitle["alpha"],
			font_size=subtitle["font_size"],
			angle=subtitle["angle"],
			text=subtitle["text"],
			image_path=str(image_path),
		)

	@staticmethod
	def _collect_image_paths(image_root: Path) -> list[Path]:
		return sorted(
			path
			for path in image_root.rglob("*")
			if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
		)

	@staticmethod
	def _to_float_tensor(image_array: np.ndarray) -> torch.Tensor:
		return torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

	@staticmethod
	def _collect_font_paths() -> list[Path]:
		candidate_dirs = []
		windows_dir = Path.home().drive + "\\Windows\\Fonts"
		if windows_dir:
			candidate_dirs.append(Path(windows_dir))
		candidate_dirs.extend(
			[
				Path("C:/Windows/Fonts"),
				Path("/usr/share/fonts"),
				Path("/Library/Fonts"),
			]
		)

		font_paths: list[Path] = []
		for directory in candidate_dirs:
			if not directory.exists():
				continue
			for extension in ("*.ttf", "*.otf", "*.ttc"):
				font_paths.extend(directory.rglob(extension))

		unique_paths = sorted({path.resolve() for path in font_paths})
		return list(unique_paths)

	def _random_text(self) -> str:
		line_count = self.random.randint(1, 2)
		lines = []
		for _ in range(line_count):
			char_count = self.random.randint(10, 42)
			chunks: list[str] = []
			for idx in range(char_count):
				if idx > 0 and self.random.random() < 0.22:
					chunks.append(" " * self.random.randint(1, 3))

				char = self.random.choice(DEFAULT_SUBTITLE_LETTERS)
				if self.random.random() < 0.28:
					char = char.upper()
				chunks.append(char)

				if self.random.random() < 0.12:
					chunks.append(" " * self.random.randint(1, 2))

			line = "".join(chunks).strip()
			if not line:
				line = self.random.choice(DEFAULT_SUBTITLE_LETTERS).upper()
			lines.append(line)
		return "\n".join(lines)

	def _load_random_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
		shuffled_fonts = self.font_paths[:]
		self.random.shuffle(shuffled_fonts)

		for font_path in shuffled_fonts[:32]:
			try:
				return ImageFont.truetype(str(font_path), size=font_size)
			except OSError:
				continue

		for fallback_font in ("arial.ttf", "DejaVuSans.ttf"):
			try:
				return ImageFont.truetype(fallback_font, size=font_size)
			except OSError:
				continue

		return ImageFont.load_default()

	def _generate_subtitle_assets(self, image_size: tuple[int, int]) -> dict[str, object]:
		width, height = image_size
		mask_image = Image.new("L", (width, height), 0)

		text = self._random_text()
		font_size = self.random.randint(max(14, height // 22), max(18, height // 9))
		font = self._load_random_font(font_size)

		temp_draw = ImageDraw.Draw(Image.new("L", (1, 1), 0))
		text_box = temp_draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
		# Some fonts/renderers can return float bbox values; Pillow image sizes must be ints.
		text_width = int(round(text_box[2] - text_box[0]))
		text_height = int(round(text_box[3] - text_box[1]))

		# Keep the text canvas tight around glyph bounds so placement is based on subtitle size,
		# not on the full image dimensions.
		padding = max(4, font_size // 4)
		canvas_width = int(max(1, text_width + padding * 2))
		canvas_height = int(max(1, text_height + padding * 2))
		text_canvas = Image.new("L", (canvas_width, canvas_height), 0)
		text_draw = ImageDraw.Draw(text_canvas)
		origin = (
			int(round(padding - text_box[0])),
			int(round(padding - text_box[1])),
		)
		text_draw.multiline_text(
			origin,
			text,
			fill=255,
			font=font,
			spacing=4,
			align="center",
			stroke_width=max(1, font_size // 16),
		)

		min_visible_ratio = 0.95
		best_candidate: tuple[float, int, int, Image.Image, np.ndarray] | None = None
		best_ratio = -1.0

		for _ in range(30):
			angle_candidate = self.random.uniform(-9.0, 9.0)
			rotated = text_canvas.rotate(angle_candidate, resample=Image.Resampling.BICUBIC, expand=True)
			rotated_array = np.asarray(rotated, dtype=np.uint8)
			rot_h, rot_w = rotated_array.shape

			x_overflow = int(round(0.04 * rot_w))
			y_overflow = int(round(0.04 * rot_h))

			x_min = -x_overflow
			x_max = width - rot_w + x_overflow
			if x_min > x_max:
				x_min, x_max = x_max, x_min

			# Anchor subtitle around lower-center region while still allowing small overflow.
			y_min = int(round(height * 0.58 - rot_h))
			y_max = int(round(height * 0.92 - rot_h))
			y_min = max(y_min, -y_overflow)
			y_max = min(y_max, height - rot_h + y_overflow)
			if y_min > y_max:
				fallback_y = int(round(height * 0.75 - rot_h))
				y_min = fallback_y
				y_max = fallback_y

			subtitle_x_candidate = int(self.random.randint(x_min, x_max))
			subtitle_y_candidate = int(self.random.randint(y_min, y_max))

			ratio = self._visible_mask_ratio(
				rotated_mask=rotated_array,
				image_width=width,
				image_height=height,
				offset_x=subtitle_x_candidate,
				offset_y=subtitle_y_candidate,
			)

			if ratio > best_ratio:
				best_ratio = ratio
				best_candidate = (
					angle_candidate,
					subtitle_x_candidate,
					subtitle_y_candidate,
					rotated,
					rotated_array,
				)

			if ratio >= min_visible_ratio:
				break

		if best_candidate is None:
			raise RuntimeError("Failed to generate subtitle placement candidate")

		angle, subtitle_x, subtitle_y, rotated, _ = best_candidate
		mask_image.paste(rotated, (subtitle_x, subtitle_y), rotated)

		alpha = self.random.uniform(0.35, 0.95)
		color = np.array(
			[
				self.random.randint(160, 255),
				self.random.randint(160, 255),
				self.random.randint(160, 255),
			],
			dtype=np.float32,
		)

		return {
			"mask_array": np.asarray(mask_image, dtype=np.uint8),
			"alpha": alpha,
			"color": color,
			"font_size": font_size,
			"angle": angle,
			"text": text,
		}

	@staticmethod
	def _visible_mask_ratio(
		rotated_mask: np.ndarray,
		image_width: int,
		image_height: int,
		offset_x: int,
		offset_y: int,
	) -> float:
		total_mask_pixels = int(np.count_nonzero(rotated_mask))
		if total_mask_pixels == 0:
			return 0.0

		mask_height, mask_width = rotated_mask.shape
		dst_x0 = max(offset_x, 0)
		dst_y0 = max(offset_y, 0)
		dst_x1 = min(offset_x + mask_width, image_width)
		dst_y1 = min(offset_y + mask_height, image_height)
		if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
			return 0.0

		src_x0 = dst_x0 - offset_x
		src_y0 = dst_y0 - offset_y
		src_x1 = src_x0 + (dst_x1 - dst_x0)
		src_y1 = src_y0 + (dst_y1 - dst_y0)

		visible_pixels = int(np.count_nonzero(rotated_mask[src_y0:src_y1, src_x0:src_x1]))
		return float(visible_pixels) / float(total_mask_pixels)

	@staticmethod
	def _blend_subtitle(
		original_array: np.ndarray,
		mask_array: np.ndarray,
		color: np.ndarray,
		alpha: float,
	) -> np.ndarray:
		original_float = original_array.astype(np.float32)
		mask_float = (mask_array.astype(np.float32) / 255.0)[..., None]
		opacity = mask_float * alpha
		composite = original_float * (1.0 - opacity) + color[None, None, :] * opacity
		return np.clip(composite, 0, 255).astype(np.uint8)


def subtitle_sample_collate(batch: list[SubtitleSample]) -> dict[str, torch.Tensor | list[str] | list[float] | list[int]]:
	return {
		"features": torch.stack([sample.features for sample in batch]),
		"target": torch.stack([sample.target for sample in batch]),
		"mask": torch.stack([sample.mask for sample in batch]),
		"alpha": [sample.alpha for sample in batch],
		"font_size": [sample.font_size for sample in batch],
		"angle": [sample.angle for sample in batch],
		"text": [sample.text for sample in batch],
		"image_path": [sample.image_path for sample in batch],
	}


def build_dataloader(
	image_root: str | Path,
	batch_size: int = 4,
	image_size: tuple[int, int] | None = (256, 256),
	shuffle: bool = True,
	num_workers: int = 0,
	random_seed: int | None = None,
) -> DataLoader:
	dataset = SubtitleTrainingDataset(
		image_root=image_root,
		image_size=image_size,
		random_seed=random_seed,
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=subtitle_sample_collate,
	)


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
		default=Path("cocodataset"),
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
		default=1,
		help="If > 0, show this many samples split into channels 1-3, 4, and 5-7.",
	)
	parser.add_argument(
		"--save-example-grid",
		type=Path,
		default=None,
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
