# DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation

## Updates
- [2025/10/1] Paper can be found at [arxiv](https://www.arxiv.org/pdf/2509.23624).


## Overview of TOHG
<div align="justify">

Text-to-online handwriting generation (TOHG) refers to the task of synthesizing realistic pen trajectories $(G_i)$ conditioned on textual $(T)$ content and style reference $(S_i)$.

</div>

![Overview of TOHG](/imgs/TOHG_overview.png)

## DiffInk vs. Character–Layout Decoupled Approaches
<div align="justify">

(a) A two-stage pipeline combining handwritten font generation with layout post-processing; (b) **DiffInk (Ours)**, which takes text and a style reference to directly output complete text lines. Unlike the two-stage pipeline, DiffInk generates more natural character connections rather than mechanically stitching bounding boxes.

</div>

![Comparison of Methods](/imgs/methods_compare.png)

## Notification
```bibtex
@inproceedings{pan2026diffink,
  title={DiffInk: Glyph-and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation},
  author={Pan, Wei and He, Huiguo and Cheng, Hiuyi and Shi, Yilin and Jin, Lianwen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=XKOEQFKFdL}
}
