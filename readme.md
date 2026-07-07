# Subtitle remover
#
## How to install
1. Download & install miniforge
2. Create environment & activate it
```
conda create -n SubtitleRemover python=3.13
conda activate SubtitleRemover
```
3. Get PyTorch
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
4. Download a large image data set. for example the coco dataset from [here](https://cocodataset.org/#download) (since the coco dataset is downloaded over HTTP instead of HTTPS you may need to convince your browser to download it anyway.)
5. ALternatively, a small sub selection of the coco dataset is contained in this repository as well. Training performance is not very good when using this limited dataset. But it is good enough for quick testing.



