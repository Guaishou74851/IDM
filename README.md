# (TPAMI 2025) Invertible Diffusion Models for Compressed Sensing [PyTorch]

[![IEEE-Xplore](https://img.shields.io/badge/IEEE_Xplore-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/document/10874182) [![icon](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.17006) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Guaishou74851.IDM)

[Bin Chen](https://scholar.google.com/citations?user=aZDNm98AAAAJ), [Zhenyu Zhang](https://scholar.google.com/citations?user=4TbicrcAAAAJ), [Weiqi Li](https://scholar.google.com/citations?user=SIkQdEsAAAAJ), [Chen Zhao](https://chenzhao.netlify.app/)†, [Jiwen Yu](https://vvictoryuki.github.io/website/), Shijie Zhao, [Jie Chen](https://aimia-pku.github.io/), and [Jian Zhang](https://jianzhang.tech/)

*School of Electronic and Computer Engineering, Peking University, Shenzhen, China.*

*King Abdullah University of Science and Technology, Thuwal, Saudi Arabia.*

*ByteDance Inc, Shenzhen, China.*

† Corresponding author

Accepted for publication in [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) (TPAMI) 2025.

⭐ If IDM is helpful to you, please star this repo. Thanks! 🤗

## 📝 Abstract

While deep neural networks (NN) significantly advance image compressed sensing (CS) by improving reconstruction quality, the necessity of training current CS NNs from scratch constrains their effectiveness and hampers rapid deployment. Although recent methods utilize pre-trained diffusion models for image reconstruction, they struggle with slow inference and restricted adaptability to CS. To tackle these challenges, this paper proposes **I**nvertible **D**iffusion **M**odels (**IDM**), a novel efficient, end-to-end diffusion-based CS method. IDM repurposes a large-scale diffusion sampling process as a reconstruction model, and fine-tunes it end-to-end to recover original images directly from CS measurements, moving beyond the traditional paradigm of one-step noise estimation learning. To enable such memory-intensive end-to-end fine-tuning, we propose a novel two-level invertible design to transform both (1) multi-step sampling process and (2) noise estimation U-Net in each step into invertible networks. As a result, most intermediate features are cleared during training to reduce up to 93.8% GPU memory. In addition, we develop a set of lightweight modules to inject measurements into noise estimator to further facilitate reconstruction. Experiments demonstrate that IDM outperforms existing state-of-the-art CS networks by up to 2.64dB in PSNR. Compared to the recent diffusion-based approach DDNM, our IDM achieves up to 10.09dB PSNR gain and 14.54 times faster inference. Code is available at https://github.com/Guaishou74851/IDM.

## 🍭 Overview

![teaser](figs/teaser.png)

![framework](figs/framework.png)

## ⚙ Environment

```shell
torch==2.3.1+cu121
diffusers==0.30.2
transformers==4.44.2
numpy==1.26.3
opencv-python==4.10.0
scikit-image==0.24.0
```

## ⚡ Test

Download the pretrained models ([Google Drive](https://drive.google.com/file/d/1UzUg0lFqwWfmeXi8gqAOeQA3yt4HpPNy/view?usp=sharing), [PKU Disk 北大网盘](https://disk.pku.edu.cn/link/AA0B0294E9BCF64185B677BDF0951A7D54)) and put the `weight` directory into `./`, then run the following command:

```shell
python test.py --cs_ratio=0.1/0.3/0.5 --testset_name=Set11/CBSD68/Urban100/DIV2K
```

The reconstructed images will be in `./result`.

The test sets CBSD68, Urban100, and DIV2K are available at https://github.com/Guaishou74851/SCNet/tree/main/data.

For easy comparison, test results of various existing image CS methods are available on [Google Drive](https://drive.google.com/drive/folders/1Lif_7N_bCyILFLac5JcOtJ9cWpGBNVCd) and [PKU Disk](https://disk.pku.edu.cn/link/AA1C2D8A08050744449CBFCAB51A846B2D).

## 🔥 Train

Download the dataset of [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) ([Google Drive](https://drive.google.com/file/d/1TOg7BZE1XsJ7l2VzMoqFRAETk7OLcv75/view?usp=drive_link), [PKU Disk 北大网盘](https://disk.pku.edu.cn/link/AAD0DCBBD65D744526921B334ED2AB4F76)) and put the `pristine_images` directory (containing 4744 `.bmp` image files) into `./data`, then run the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=23333 train.py --cs_ratio=0.1/0.3/0.5
```

The log and model files will be in `./log` and `./weight`, respectively.

## 😍 Results

![res1](figs/res1.png)

![res2](figs/res2.png)

## 🎓 Citation

If you find the code helpful in your research or work, please cite the following paper:

```latex
@article{chen2025invertible,
  title={Invertible Diffusion Models for Compressed Sensing},
  author={Chen, Bin and Zhang, Zhenyu and Li, Weiqi and Zhao, Chen and Yu, Jiwen and Zhao, Shijie and Chen, Jie and Zhang, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
}
```
