# DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation

### Overview of TOHG
Text-to-online handwriting generation (TOHG) refers to the task of synthesizing realistic pen trajectories $(G_i)$ conditioned on textual $(T)$ content and style reference $(S_i)$.

<p align="center">
  <img src="/imgs/TOHG_overview.png" alt="Overview of TOHG" width="600px"/>
</p>

---

### DiffInk vs. Character–Layout Decoupled Approaches
(a) A two-stage pipeline combining handwritten font generation with layout post-processing;  
(b) **DiffInk (Ours)**, which takes text and a style reference to directly output complete text lines. Unlike the two-stage pipeline, DiffInk generates more natural character connections rather than mechanically stitching bounding boxes.

<p align="center">
  <img src="/imgs/methods_compare.png" alt="DiffInk Comparison" width="600px"/>
</p>


Paper can be found at https://www.arxiv.org/pdf/2509.23624
