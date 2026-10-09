# Experiment Commands

All commands are run from the repository root. The main configuration is `configs/paper.json`; use `--set key=value` during recovery to override a field. The resolved configuration is saved to `run.json` and reused by the remaining stages.

## Study Launcher

```bash
python tools/experiments.py STUDY --data-root data --teachers checkpoints \
  --output-root outputs --seeds 42 43 44 45 46 --execute
```

| STUDY | Settings |
| --- | --- |
| `main` | CIFAR-100, Tiny-ImageNet, ImageNette, ImageWoof, ImageNet-1K; IPC 1/10/50; ResNet-18/50/101 |
| `high-ipc` | CIFAR-10 IPC 250/500/1000/1500; CIFAR-100 IPC 100/150; Tiny-ImageNet IPC 100 |
| `cross` | ImageNet-1K IPC 10; EfficientNet-B0, MobileNetV2, ShuffleNetV2, Swin-Tiny, Wide-ResNet-50-2, DenseNet-121/169/201 |
| `components` | Static residual, DRC, PTA, STC, HFTA, RETA, and paired combinations |
| `assignment` | Independent retrieval, balanced barycenters, capacity without structure, full STC |
| `packed` | Whole-image recovery, cell slots with random replay, cell-aligned replay, static-cell anchors, and cells without HFTA |
| `stc-sweep` | Five complexity weights times seven structure weights |
| `hfta-sweep` | Six topology weights, two datasets and two IPC budgets |
| `group-pool` | Group sizes 2/4/8 and candidate pools 16/32/64 |

Omit `--execute` to print commands. Each completed run produces per-architecture development-selection and test-evaluation files. Collect multiple seeds with:

```bash
python tools/collect_results.py --root outputs --output summary.json
```

The STC weight study uses the `development_accuracy` fields for hyperparameter selection, not the `test_accuracy` fields.

## A Single Ablation

For STC without HFTA:

```bash
python -u run.py all --dataset imagenette --ipc 10 --data data/imagenette \
  --teacher-checkpoint checkpoints/imagenette/resnet18.pth \
  --output outputs/imagenette_stc_only_seed42 --set topology=none
```

For the RETA comparison, use `--set assignment=independent --set topology=pta`. For balanced barycenters with HFTA, use `--set assignment=balanced`. For random-crop replay of recovered cell images, use `--ipc 1 --set cell_replay=false`; the random crop state is still replayed exactly.

## Diagnostics and Efficiency

```bash
python tools/diagnose.py --run outputs/imagenette_ipc10_seed42
python tools/efficiency.py --run outputs/imagenette_ipc10_seed42
```

Both commands write JSON under the run's `results` directory. Keep timing runs on one GPU and use the same precision, dataset, recovery schedule, and group/candidate sizes across comparisons.

## Common Corruptions

Arrange corrupted images as `CORRUPTION_ROOT/<type>/<severity>/<class>/*.JPEG`, with severity 1--5 and the same class names/order as the clean data. The evaluator expects the standard 15 corruption types listed in `tools/corruption.py`.

```bash
python tools/corruption.py --runs outputs/imagenette_ipc1_seed42 \
  outputs/imagenette_ipc1_seed43 outputs/imagenette_ipc1_seed44 \
  outputs/imagenette_ipc1_seed45 outputs/imagenette_ipc1_seed46 \
  --corruptions data/imagenette_c --output imagenette_corruption.json
```

The student checkpoint is chosen on the clean development split. Each run averages its 75 condition accuracies; aggregation then reports the mean and sample standard deviation across runs. Repeat for the other five ImageNet subsets.

## Continual Learning

First recover an ImageNet-1K set with at least ten stored images per class, enough for the largest buffer after the first 100-class task. Then run:

```bash
python -u tools/continual.py --run outputs/imagenet1k_ipc10_seed42 \
  --data data/imagenet1k --teacher-checkpoint checkpoints/imagenet1k/resnet18.pth \
  --capacity 500 --seed 42 --output outputs/continual_capacity500_seed42
```

The fixed class permutation uses seed 0. Ten disjoint tasks each introduce 100 classes. A single ResNet-18 continues across tasks. The total buffer is allocated evenly among old classes, with remainder slots assigned by the fixed class order. Some old classes receive no exemplar when capacity is smaller than the number of old classes. Capacities are 200, 500 and 1000 stored images.

The training loop uses each current-task minibatch and an equal-size sampled replay minibatch when memory is available. Current images use cross-entropy; replay images use teacher KL supervision on the identical augmented view. The defaults are 100 epochs per task, batch size 128, AdamW, and the four development-selection presets. These are exposed through the command line and recorded in `protocol.json`. The selected preset maximizes average incremental development accuracy. Its ten checkpoints are evaluated on all seen test classes, without task identity. The reported metric is the unweighted mean of the ten stage accuracies.

## Plug-in Integration

`retapp/plugin.py::RecoveryPlugin` supplies STC and HFTA hooks for an existing teacher-guided recovery loop. Keep the host recovery loss, image augmentation, and student protocol. Provide normalized real teacher features, patch complexity values, and a function returning raw RGB patches of a requested size.

```python
from retapp.config import Config
from retapp.plugin import RecoveryPlugin

config = Config(ipc=10)
plugin = RecoveryPlugin(teacher, real_features, complexity, fetch_patches,
                        slot_count=len(class_images), config=config)

# At a topology evaluation step; class_images contains every class slot.
plugin.refresh(class_images)
loss = host_recovery_loss + plugin.loss(active_images, active_indices)
loss.backward()
optimizer.step()

# At a residual block boundary, outside the pixel gradient update.
with torch.no_grad():
    active_images.copy_(plugin.connect(active_images))
```

For HFTA-only integration, omit `connect`. For STC-only integration, set `topology="none"`. For the combined plug-in, use both calls. Initialize one plug-in per class and evaluate it on groups of at most four images, retaining the host method's recovery schedule. Inputs and fetched patches use RGB values in [0,1]; `Teacher` performs normalization and resizing.
