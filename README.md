# CODiff

This repository contains the official PyTorch implementation of [CODiff: One-Step Diffusion Model for Camouflaged Object Detection](https://openreview.net/forum?id=eDlsO4kFaX).

## Requirements

The complete environment specification is provided in [`requirements.yml`](requirements.yml). The main dependencies include Python 3.8.10, PyTorch 2.1.0, and torchvision 0.16.0. PyTorch packages are installed from the CUDA 12.1 wheel index.

Create and activate the Conda environment with:

```bash
conda env create -f requirements.yml
conda activate codiff
```

## Dataset

Download the CAMO, COD10K, and NC4K datasets from the [COD dataset collection](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod), place them into the `datasets/` directory, and appropriately modify the path configurations in `config/dataset.yaml`.

CAMO and COD10K are used for training. Evaluation is performed on CAMO, COD10K, and NC4K.

## DINOv2 Pretrained Weights

Download the official [DINOv2 ViT-B/14 pretrained weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and place the file at:

```text
pretrained_weights/dinov2_vitb14_pretrain.pth
```

Alternatively, download the weights from the command line:

```bash
mkdir -p pretrained_weights
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
  -O pretrained_weights/dinov2_vitb14_pretrain.pth
```

## Training

After preparing the datasets and DINOv2 pretrained weights, run:

```bash
bash train.sh
```

The default training parameters are defined in [`train.sh`](train.sh), while the model and dataset configurations are provided under [`config/`](config/).

## Evaluation

To evaluate the best CODiff model, download `CODiff_model.pt` from the [CODiff weights repository](https://huggingface.co/datasets/kiisooo/codiff_weights/) and save it as:

```text
checkpoints/model-best.pt
```

The checkpoint can also be downloaded directly with:

```bash
mkdir -p checkpoints
wget https://huggingface.co/datasets/kiisooo/codiff_weights/resolve/main/CODiff_model.pt \
  -O checkpoints/model-best.pt
```

Then run:

```bash
bash eval.sh
```

By default, the script evaluates the model on CAMO, COD10K, and NC4K and saves predictions under `test_results/`.

> **Note:** `eval.sh` removes and recreates the existing `test_results/` directory before evaluation. Please preserve any results that you wish to keep before running the script.

## Acknowledgements

Our implementation is based on the [denoising diffusion PyTorch repository](https://github.com/lucidrains/denoising-diffusion-pytorch) by lucidrains. We also thank the authors of [CamoDiffusion](https://github.com/Rapisurazurite/CamoDiffusion) and [DINOv2](https://github.com/facebookresearch/dinov2) for making their work publicly available.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{fu2026codiff,
  title={{COD}iff: One-Step Diffusion Model for Camouflaged Object Detection},
  author={Xiaotong Fu and Qian Liu and Qihang Zhou and Wenchao Meng and Qinmin Yang and Shibo He},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=eDlsO4kFaX}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party components remain subject to their respective licenses.