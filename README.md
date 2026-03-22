<h1 align="center">
  DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer  
  for Text to Online Handwriting Generation
</h1>

## Updates
- [2026/3/21] Code and pretrained weights are released.
- [2026/1/29] DiffInk is accepted by ICLR 2026 🎉🎉🎉.
- [2025/10/1] Paper can be found at [arxiv](https://www.arxiv.org/pdf/2509.23624).


## Overview of TOHG
<div align="justify">

Text-to-Online Handwriting Generation (TOHG) refers to the task of synthesizing realistic pen trajectories $(G_i)$ conditioned on textual content $(T)$ and style reference $(S_i)$.

</div>

<div align="center">
  <img src="/imgs/TOHG_overview.png" alt="Overview of TOHG" width="70%">
</div>

## DiffInk vs. Character–Layout Decoupled Approaches
<div align="justify">

(a) A two-stage pipeline combining handwritten font generation with layout post-processing; (b) **DiffInk (Ours)**, which takes text and a style reference to directly output complete text lines. Unlike the two-stage pipeline, DiffInk generates more natural character connections rather than mechanically stitching bounding boxes.

</div>

<div align="center">
  <img src="/imgs/methods_compare.png" alt="Overview of TOHG" width="70%">
</div>


## Usage

### Data and Pretrained Weights

Download the dataset and pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1h_uLmn-55WmbSBGh1ES8-rftAbDs8riB?usp=drive_link).


### Training

- Train the **InkVAE** model with: `bash scripts/train_vae_ddp.sh`

- Then, train the **InkDiT** model with: `bash scripts/train_dit_ddp.sh`

- Finally, fine-tune the model on real data with: `bash scripts/tune_dit_ddp.sh`

### Inference

Run inference with: `CUDA_VISIBLE_DEVICES=0 python val_dit.py`

<!-- ## Copyright

This repository is provided for non-commercial research purposes only; for commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

For any issues encountered during use, please open an issue or contact Wei Pan (eewpan@mail.scut.edu.cn). -->

## Citation
If you find this work useful or use this code in your research, please consider citing the following paper.

```bibtex
@inproceedings{pan2026diffink,
  title={DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation},
  author={Wei Pan and Huiguo He and Hiuyi Cheng and Yilin Shi and Lianwen Jin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=XKOEQFKFdL}
}
```

### Acknowledgements

This work builds upon the open-source framework of [F5-TTS](https://github.com/swivid/f5-tts) and leverages the [CASIA-OLHWDB](https://nlpr.ia.ac.cn/databases/handwriting/home.html) dataset.