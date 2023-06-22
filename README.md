#  ResFormer: Scaling ViTs with Multi-Resolution Training
Official PyTorch implementation of ResFormer: Scaling ViTs with Multi-Resolution Training, CVPR2023 | [Paper](https://arxiv.org/abs/2212.00776)

## Overview

<p align="center">
<img src="./imgs/network.png" width=100% height=100% 
class="center">
</p>
We introduce, ResFormer, a framework that is built upon the seminal idea of multi-resolution training for improved performance on a wide spectrum of, mostly unseen, testing resolutions. In particular, ResFormer operates on replicated images of different resolutions and enforces a scale consistency loss to engage interactive information across different scales. More importantly, to alternate among varying resolutions effectively, especially novel ones in testing, we propose a global-local positional embedding strategy that changes smoothly conditioned on input sizes. 


## Installation

### Image Classification 

```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.5.4
pip install tensorboard
```

## Scripts
### Training on ImageNet-1k
The default script for training ResFormer-S-MR with training resolutions of 224, 160 and 128. 
```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py  --data-path  YOUR_DATA_PATH  --model resformer_small_patch16  --output_dir YOUR_OUTPUT_PATH --batch-size 128 --pin-mem --input-size 224 160 128 --auto-resume  --distillation-type 'smooth-l1' --distillation-target cls --sep-aug
```
The default script for training ResFormer-B-MR with training resolutions of 224, 160 and 128. 

```bash
python -m torch.distributed.launch --nproc_per_node 8 main.py  --data-path  YOUR_DATA_PATH  --model resformer_base_patch16  --output_dir YOUR_OUTPUT_PATH --batch-size 128 --pin-mem --input-size 224 160 128 --auto-resume  --distillation-type 'smooth-l1' --distillation-target cls --sep-aug --epochs 200 --drop-path 0.2  --lr 8e-4 --warmup-epochs 20 --clip-grad 5.0 --epochs 200  --cooldown-epochs 0  
```


## Model Zoo

### Image Classification on ImageNet-1k
| name | Training Res| Top-1(96)| Top-1(128)| Top-1(160)| Top-1(224) | Top-1(384) | Top-1(512) | model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResFormer-T-MR | 128, 160, 224|  61.40| 67.78| 71.09| 73.85 | 75.04| 73.77| [google](https://drive.google.com/file/d/1Cw2eyFYepGYUvrilUan8JvK6XGxoHOVS/view?usp=sharing) |
| ResFormer-S-MR | 128, 160, 224| 73.59 | 78.24 | 80.39| 82.16| 82.72 |82.00|[google](https://drive.google.com/file/d/1WmaUI0ps4f5RAN9bCgFroyPU7OXeQi5O/view?usp=sharing) |
| ResFormer-S-MR | 128, 224, 384| 72.92 |77.84 |80.09| 82.28| 83.70| 83.86| [google](https://drive.google.com/file/d/1f2HSTmRkJAkIcHlwL9_TwJiPcIbXgDzK/view?usp=sharing) |
| ResFormer-B-MR | 128, 160, 224|  75.86| 79.74|81.52|82.72| 83.29| 82.63| [google](https://drive.google.com/file/d/1m9yFtXZoKdgKsylww3zez20YODaydLSJ/view?usp=sharing) |



## Catalog
- [x] image classification 
- [ ] object detection
- [ ] semantic segmentation
- [ ] action recognition


## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.



## Citation

```
@inproceedings{tian2022resformer,
  title={ResFormer: Scaling ViTs with Multi-Resolution Training},
  author={Tian, Rui and Wu, Zuxuan and Dai, Qi and Hu, Han and Qiao, Yu and Jiang, Yu-Gang},
  booktitle={CVPR},
  year={2023}
}
```