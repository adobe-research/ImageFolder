## ImageFolder🚀: Autoregressive Image Generation with Folded Tokens

<div align="center">

[![project page](https://img.shields.io/badge/ImageFolder%20project%20page-lightblue)](https://lxa9867.github.io/works/imagefolder/index.html)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/abs/2410.01756)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/ang9867/imagefolder/tree/main)&nbsp;

</div>
<!-- <p align="center" style="font-size: larger;">
  <a href="placeholder">🔥ImageFolder: Autoregressive Image Generation with Folded Tokens</a>
</p> -->

<p align="center">

<div align=center>
	<img src=assets/teaser.png/>
</div>


## Updates 
- (2024.11.14) Code will be released in two weeks (company approval in progress).
- (2024.10.03) We are working on advanced training for the ImageFolder tokenizer. 
- (2024.10.01) Repo created. Code and checkpoints will be released soon.


## Model Zoo

We provide pre-trained tokenizers for image reconstruction on ImageNet.

![image](https://github.com/user-attachments/assets/d4c81ae5-e9a0-4f45-bec5-2a4db870818f)

| Training  | Eval | Codebook Size | rFID ↓  | Link | Resolution | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |  
| ImageNet | ImageNet | 4096 | 0.80 | [Huggingface](https://huggingface.co/ang9867/imagefolder/resolve/main/imagenet-4096.pt?download=true) | 256x256 | 200 Epoch |
| ImageNet | ImageNet | 8192 | 0.70 | [Huggingface](https://huggingface.co/ang9867/imagefolder/resolve/main/imagenet-8192.pt?download=true) | 256x256 | 200 Epoch |
| ImageNet | ImageNet | 16384 | 0.67 | [Huggingface](https://huggingface.co/ang9867/imagefolder/resolve/main/imagenet-16384.pt?download=true) | 256x256 | 200 Epoch |

---

We provide a pre-trained generator for class-conditioned image generation on ImageNet 256x256 resolution.

| Type | Dataset | Model Size | gFID ↓ | Link | Resolution |
| :---: | :---: | :---: | :---: | :---: | :---: |
| VAR | ImageNet | 362M | 2.60 | [Huggingface](https://huggingface.co/ang9867/imagefolder/resolve/main/imagenet-var-4096.pt?download=true) | 256x256 |


## Installation

Install all packages as
```
conda env create -f environment.yml
```

## Dataset 

We download the ImageNet2012 from the website and collect it as 

```
ImageNet2012
├── train
└── val
```

If you want to train or finetune on other datasets, collect them in the format that ImageFolder (pytorch's [ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)) can recognize.

```
Dataset
├── train
│   ├── Class1
│   │   ├── 1.png
│   │   └── 2.png
│   ├── Class2
│   │   ├── 1.png
│   │   └── 2.png
├── val
```

## Training code for tokenizer

Please login to Wandb first using
```
wandb login
```
rFID will be automatically evaluated and reported on Wandb. The best checkpoint on the val set will be saved.
```
torchrun --nproc_per_node=8 tokenizer/tokenizer_image/msvq_train.py --config configs/tokenizer.yaml
```

Please modify the configuration file as needed for your specific dataset. We list some important ones here.
```
vq_ckpt: ckpt_best.pt                # resume
cloud_save_path: output/exp-xx       # output dir
data_path: ImageNet2012/train        # training set dir
val_data_path: ImageNet2012/val      # val set dir
enc_tuning_method: 'full'            # ['full', 'lora', 'frozen']
dec_tuning_method: 'full'            # ['full', 'lora', 'frozen']
codebook_embed_dim: 32               # codebook dim
codebook_size: 4096                  # codebook size
product_quant: 2                     # branch number
codebook_drop: 0.1                   # quantizer dropout rate
semantic_guide: dinov2               # ['none', 'dinov2']
```

## Tokenizer linear probing
```
torchrun --nproc_per_node=8 tokenizer/tokenizer_image/linear_probing.py --config configs/tokenizer.yaml
```

## Training code for VAR

We follow the VAR training code and our training cmd for reproducibility is 

```
torchrun --nproc_per_node=8 train.py --bs=768 --alng=1e-4 --fp16=1 --alng=1e-4 --wpe=0.01 --tblr=8e-5 --data_path /mnt/localssd/ImageNet2012/ --encoder_model vit_base_patch14_dinov2.lvd142m --decoder_model vit_base_patch14_dinov2.lvd142m --product_quant 2 --semantic_guide dinov2 --num_latent_tokens 121 --v_patch_nums 1 1 2 3 3 4 5 6 8 11 --pn 1_1_2_3_3_4_5_6_8_11 --patch_size 11 --vae_ckpt /path/to/ckpt.pt --sem_half True 
```

## Inference code for ImageFolder

```
torchrun --nproc_per_node=8 inference.py --infer_ckpt /path/to/ckpt --data_path /path/to/ImageNet --encoder_model vit_base_patch14_dinov2.lvd142m --decoder_model vit_base_patch14_dinov2.lvd142m --product_quant 2 --semantic_guide dinov2 --num_latent_tokens 121 --v_patch_nums 1 1 2 3 3 4 5 6 8 11 --pn 1_1_2_3_3_4_5_6_8_11 --patch_size 11 --sem_half True --cfg 3.25 3.25 --top_k 750 --top_p 0.95
```


## Ablation
| ID  | Method                                              | Length | rFID ↓  | gFID ↓ | ACC ↑|
| --- | --------------------------------------------------- | ------ | ------- | ------- |------- |
| 🔶1   | Multi-scale residual quantization (Tian et al., 2024) | 680    | 1.92    | 7.52 | - |
| 🔶2   | + Quantizer dropout                                  | 680    | 1.71    | 6.03 | - |
| 🔶3   | + Smaller patch size K = 11                          | 265    | 3.24    | 6.56 | - |
| 🔶4   | + Product quantization & Parallel decoding           | 265    | 2.06    | 5.96 | - |
| 🔶5   | + Semantic regularization on all branches            | 265    | 1.97    | 5.21 | - |
| 🔶6   | + Semantic regularization on one branch              | 265    | 1.57    | 3.53 | 40.5 |
| 🔷7   | + Stronger discriminator             | 265    | 1.04    | 2.94 | 50.2|
| 🔷8   | + Equilibrium enhancement    | 265    | 0.80    | 2.60 | 58.0|

🔶1-6 are already in the released paper, and after that 🔷7+ are advanced training settings used similar to VAR (gFID 3.30).


## Generation

<div align=center>
	<img src=assets/visualization.png/>
</div>

## License
[Adobe Research License](LICENSE.md)

## Acknowledge
We would like to thank the following repositories: [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR) and [ControlVAR](https://github.com/lxa9867/ControlVAR).
## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using
```
@misc{li2024imagefolderautoregressiveimagegeneration,
      title={ImageFolder: Autoregressive Image Generation with Folded Tokens}, 
      author={Xiang Li and Hao Chen and Kai Qiu and Jason Kuen and Jiuxiang Gu and Bhiksha Raj and Zhe Lin},
      year={2024},
      eprint={2410.01756},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.01756}, 
}
```
