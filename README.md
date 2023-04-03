#  ResFormer: Scaling ViTs with Multi-Resolution Training
Official PyTorch implementation of ResFormer: Scaling ViTs with Multi-Resolution Training, CVPR2023 | [Paper](https://arxiv.org/abs/2212.00776)

## Overview

<p align="center">
<img src="./imgs/network.png" width=100% height=100% 
class="center">
</p>
We introduce, ResFormer, a framework that is built upon the seminal idea of multi-resolution training for improved performance on a wide spectrum of, mostly unseen, testing resolutions. In particular, ResFormer operates on replicated images of different resolutions and enforces a scale consistency loss to engage interactive information across different scales. More importantly, to alternate among varying resolutions effectively, especially novel ones in testing, we propose a global-local positional embedding strategy that changes smoothly conditioned on input sizes. 

**Code and models will be available soon!**

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