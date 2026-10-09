# Third-Party Notices

`vendor/cifar_resnet.py` is retained from the FADRM-based codebase and preserves its original implementation. The FADRM MIT copyright notice is retained in `LICENSE`.

The remaining recovery, transport, topology, replay, evaluation, and experiment orchestration code is provided with this RETA++ release under the same MIT license.

Runtime dependencies retain their respective licenses:

- PyTorch and torchvision: model implementations, automatic differentiation, datasets, and image transforms.
- NumPy and SciPy: numerical arrays, linear assignment, and optimization.
- Pillow: image loading and export.
- GUDHI: persistent-homology pairings used by the PTA comparison.
- pytest: automated tests.

The framework image in `docs/assets/framework.png` comes from the RETA++ manuscript. Dataset images and pretrained checkpoints are supplied separately by their respective providers.

The README structure was inspired by the MTT repository at https://github.com/georgecazenavette/mtt-distillation.
