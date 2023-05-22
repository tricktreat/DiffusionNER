# DiffusionNER

Code for [DiffusionNER: Boundary Diffusion for Named Entity Recognition](https://github.com/tricktreat/DiffusionNER). We will have all the code ready soon.

<p align="center"><img src="./assets/overview.jpg"></p>

In this paper, we propose DiffusionNER, which formulates the named entity recognition task as a boundary-denoising diffusion process and thus generates named entities from noisy spans. During training, DiffusionNER gradually adds noises to the golden entity boundaries by a fixed forward diffusion process and learns a reverse diffusion process to recover the entity boundaries. In inference, DiffusionNER first randomly samples some noisy spans from a standard Gaussian distribution and then generates the named entities by denoising them with the learned reverse diffusion process. The proposed boundary-denoising diffusion process allows progressive refinement and dynamic sampling of entities, empowering DiffusionNER with efficient and flexible entity generation capability. Experiments on multiple flat and nested NER datasets demonstrate that DiffusionNER achieves comparable or even better performance than previous state-of-the-art models.