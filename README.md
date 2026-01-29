# DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation

## Updates
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

## Notification
```bibtex
@article{pan2025diffink,
  title={DiffInk: Glyph-and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation},
  author={Pan, Wei and He, Huiguo and Cheng, Hiuyi and Shi, Yilin and Jin, Lianwen},
  journal={arXiv preprint arXiv:2509.23624},
  year={2025}
}
