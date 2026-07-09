# Subtitle Remover

A deep learning pipeline that automatically removes subtitles from images using two specialized U-Net models.

## Overview

Subtitle Remover uses a two-stage neural network approach to detect and remove subtitles from images:

1. **Mask Estimator** - A U-Net that predicts where subtitles are located by generating a binary mask
2. **Pixel Infiller** - A U-Net that reconstructs the original pixels under the detected subtitle regions

This approach is more effective than single-stage models because it allows the network to first identify subtitle locations, then focus restoration efforts on those specific areas.

### How It Works

**Training Pipeline:**
- The system generates synthetic training data by overlaying randomly positioned and rotated text onto real images
- The mask estimator learns to identify subtitle pixels
- The pixel infiller learns to reconstruct original image content using the predicted mask as guidance
- Both models use L1 reconstruction loss and are optimized with the Adam optimizer
- Training supports GPU acceleration via PyTorch CUDA and mixed-precision training

**Inference Pipeline:**
- Load both trained U-Net models from checkpoints
- Run mask estimation to detect subtitle regions
- Apply morphological dilation to ensure complete subtitle coverage
- Run pixel inpainting to reconstruct the original pixels
- Composite the results back into the original image

### Architecture

Both models use **U-Net**, a proven encoder-decoder architecture with skip connections:
- **Encoder**: Progressive downsampling (max pooling) with feature extraction
- **Decoder**: Progressive upsampling (transpose convolution) with skip connections
- **DoubleConv blocks**: Dual 3×3 convolutions with batch normalization and ReLU activation

Models are parameterizable for different channel depths and input/output dimensions.

## How to install
0. Make your current directory equal to this repository
```
cd e:\dev\subtitleremover
```
1. Download & install miniforge
2. Create environment & activate it
```
conda create -n SubtitleRemover python=3.13
conda activate SubtitleRemover
```
3. Get PyTorch & check if it works on your GPU
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python checkpytorchgpu.py
```
4. Install other required Python packages
```
pip install -r requirements.txt
```
5. Download a large image data set. for example the coco dataset from [here](https://cocodataset.org/#download) (since the coco dataset is downloaded over HTTP instead of HTTPS you may need to convince your browser to download it anyway.) <br>
Alternatively, a small sub selection of the coco dataset is contained in this repository as well. Training performance is not very good when using this limited dataset. But it is good enough for quick testing.

## Usage

### Training the Mask Estimator

The mask estimator learns to predict subtitle locations as a binary mask:

```bash
python train_mask_estimator.py \
  --train-root ./cocodataset/train \
  --validate-root ./cocodataset/validate \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --height 256 \
  --width 256
```

**Key Arguments:**
- `--train-root`: Path to training images
- `--validate-root`: Path to validation images  
- `--epochs`: Number of training epochs
- `--batch-size`: Samples per batch
- `--learning-rate`: Adam optimizer learning rate
- `--height`, `--width`: Image dimensions for training
- `--base-channels`: U-Net base channel count (default: 64)
- `--save-path`: Checkpoint path for resuming training
- `--best-path`: Best checkpoint path (lowest validation loss)

Run `python train_mask_estimator.py --help` for all available options.

### Training the Pixel Infiller

The pixel infiller learns to reconstruct original pixels in subtitle regions:

```bash
python train_infiller.py \
  --train-root ./cocodataset/train \
  --validate-root ./cocodataset/validate \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --height 256 \
  --width 256
```

Arguments are the same as mask estimator training. The pixel infiller uses a masked loss function that only applies reconstruction loss to detected subtitle pixels.

Run `python train_infiller.py --help` for all available options.

### Running Inference

Once both models are trained, generate subtitle-removed images:

```bash
python infere.py \
  --data-root ./cocodataset/validate \
  --checkpoint-mask-estimator ./artifacts/mask_estimator_best.pt \
  --checkpoint-infiller ./artifacts/infiller_best.pt \
  --output ./artifacts/inference_preview.png \
  --num-examples 4 \
  --threshold 0.5
```

**Key Arguments:**
- `--data-root`: Path to images for inference
- `--checkpoint-mask-estimator`: Path to mask estimator model
- `--checkpoint-infiller`: Path to pixel infiller model
- `--output`: Path to save preview grid image
- `--num-examples`: Number of examples in preview grid
- `--threshold`: Mask binarization threshold (0.0-1.0)
- `--batch-size`: Inference batch size

Run `python infere.py --help` for all available options.

## Project Structure

- **train_mask_estimator.py** - Training script for subtitle mask estimation model
- **train_infiller.py** - Training script for pixel inpainting model
- **infere.py** - Inference pipeline that applies both models to generate subtitle-removed images
- **train_utils.py** - Shared training infrastructure (generic training loop, checkpoint management, metrics)
- **unet_utils.py** - U-Net architecture components (DoubleConv, DownBlock, UpBlock, generic UNet class)
- **data_generator.py** - Synthetic training data generation with random text overlay
- **checkpytorchgpu.py** - Utility to verify PyTorch GPU support
- **cocodataset/** - Training/validation/test images (small subset for quick testing)
- **artifacts/** - Trained model checkpoints and inference outputs

## Technical Details

- **Framework**: PyTorch with CUDA GPU support
- **Python**: 3.13
- **Loss Functions**: BCEWithLogitsLoss for mask estimation, L1Loss for pixel inpainting
- **Optimizer**: Adam with learning rate scheduling
- **Data Format**: 7-channel tensors [original_RGB(3), binary_mask(1), subtitled_RGB(3)]
- **Image Resolution**: Default 256×256 (configurable)
- **Mixed Precision Training**: Supported via torch.amp for faster training on supported GPUs


