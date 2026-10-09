# Method and Configuration

## Recovery

`retapp/recovery.py` alternates teacher-guided pixel optimization and stage-boundary anchor assignment. Pixel parameters use normalized image coordinates. The teacher is frozen, but its forward pass retains the derivatives with respect to synthetic pixels.

The recovery objective is teacher cross-entropy plus batch-normalization statistic matching, with HFTA added at its evaluation steps. The first BN layer has multiplier 10; other BN layers have multiplier one. Real images initialize the slots. Four recovery blocks contain three residual injections.

| Paper setting | Configuration field | Default |
| --- | --- | --- |
| Image budget K | `ipc` | 10 |
| Recovery budget B | `recovery_steps` | 300; 2000 for CIFAR-100/Tiny-ImageNet at IPC=1 |
| Pixel learning rate | `recovery_lr` | 0.25, cosine decay |
| Residual ratio alpha | `alpha` | 0.5 |
| BN weight | `bn_weight` | 0.01 |
| Maximum image-slot group | `group_size` | 4 |
| Candidate count M | `candidates` | 32 |
| Complexity coefficient | `complexity_weight` | 0.1 |
| Structure coefficient | `structure_weight` | 0.2 |
| Entropy zeta | `entropy` | 0.05 |
| Alternating updates R | `transport_updates` | 3 |
| HFTA weight | `hfta_weight` | 0.5 |
| Filtration scales | `hfta_scales` | 8 |
| Soft-edge temperature / scale spacing | `tau_fraction` | 0.5 |
| Spectral bandwidth / positive eigenvalue mean | `rho_fraction` | 0.05 |
| Class-cloud cap | `cloud_cap` | 32 |
| Real-reference subsamples Q | `reference_samples` | 20 |
| Topology/cache interval | `topology_every` | 10 |
| IPC=1 grid | `grid_side` | 2, giving four cell slots |

`recovery_steps=0` and `epochs=0` select the dataset defaults. The student default is 1000 epochs for CIFAR and for Tiny-ImageNet IPC=1, and 300 otherwise. The image resolution is 32 for CIFAR, 64 for Tiny-ImageNet, and 224 for ImageNet datasets. The large-image residual schedule uses resolutions 200, 224, 200, 224. Cell slots occupy half the corresponding image width and height. Pixel Adam uses betas (0.5, 0.9); student AdamW uses temperature 20 and weight decay 0.01.

## Structure Transport Connection

Implementation: `retapp/transport.py::assign`.

1. Compute the cosine fit between normalized teacher features of the active slots and the full same-class real pool. Retain the M candidates with the smallest mean group fit cost.
2. Compute the fit, smoothed-gradient complexity, and pairwise-distance structure costs. Min-max normalization is applied separately within the candidate set. A constant cost becomes zero.
3. Initialize the plan from fit plus weighted complexity. Give each real slot mass 1/M, each candidate mass 1/M, and the dummy row mass (M-K_g)/M. Omit the dummy row when M=K_g.
4. Perform R structure-cost updates, solving the entropic transport problem after each update. Log-domain Sinkhorn is used; slowly converging plans are refined on the same entropic dual to meet the marginal tolerance.
5. Recompute the final cost from the last coupling and apply SciPy's Hungarian solver to that cost. The selected patch indices are distinct within the active group. No barycentric pixel averaging is used in STC.
6. Mix each selected real patch into its slot with ratio alpha. Different groups can select the same patch; the assignment capacity is local to a group.

`assignment=balanced` implements the balanced barycentric comparison. `independent` implements fit-complexity retrieval without the capacity constraint. `static` keeps the initialization anchors, and `none` disables residual injection. These are explicit ablation choices; `stc` is the default.

## Hodge-Filtration Topology Alignment

Implementation: `retapp/topology.py::ClassTopology` and `response`.

The real reference has the same cardinality as the sampled synthetic class cloud. It is fixed before recovery and averages Q real subsamples. Filtration scales are the 0.1 through 0.8 quantiles of real squared pairwise distances. Tau is half the mean adjacent-scale spacing. Separate bandwidths for the graph and one-form spectra are calibrated from positive real-reference eigenvalues and remain fixed during optimization.

All vertex pairs and triangles are represented in oriented incidence matrices B1 and B2. With sigmoid edge weights w and geometric-mean triangle weights t, the operators are:

```text
B1_tilde = B1 diag(sqrt(w))
B2_tilde = B2 diag(sqrt(t))
L0 = B1_tilde B1_tilde^T
L1 = B1_tilde^T B1_tilde + B2_tilde B2_tilde^T
beta0 = tr exp(-L0 / rho0)
beta1 = tr exp(-L1 / rho1) - sum_edges(1 - w)
```

The missing-edge correction is retained, and finite-temperature first-order responses are not clipped. HFTA compares the concatenated zero- and first-order response curves with squared Euclidean distance. The symmetric heat-trace derivative is implemented directly so repeated eigenvalues do not cause unstable eigenvector derivatives. The main loss uses the full one-form spectrum, not an Euler approximation.

Every active group participates at topology-evaluation steps. Its current features remain differentiable. Other slots come from a detached class cache refreshed at the same interval. If the class exceeds the cap, inactive slots are sampled while retaining all active slots. For IPC=1, the class cloud contains the four cell slots.

`topology=pta` activates persistent-image alignment for RETA/component comparisons. It uses mutual-neighbor graphs and GUDHI pairings, with gradients through the corresponding birth/death distances. `topology=none` removes the topology loss.

## Cell Slots and Replay

Implementation: `retapp/replay.py`, `retapp/recovery.py`, and `retapp/training.py`.

At IPC=1, initialize four distinct high-confidence patches, recover their cell pixels, and assign distinct anchors at each residual boundary. Save the cells as a single 2-by-2 image at the dataset resolution. Relabeling selects one cell, records its index, bounding box and flip, and saves the teacher logits on that exact view. Training reads the same PNG and replays the recorded state. No additional independent cell images are stored in the distilled dataset.

## Selection and Data Splits

The split is stratified with `split_seed=42` and shared across experiment seeds. Recovery uses only the training portion. Four student presets cross learning rates 0.001/0.0005 with cosine periods one/two. All presets start from the same seeded initialization. Their final development accuracy determines the chosen checkpoint; the test set is only read by evaluation. Label-cache metadata checks the image checksum, crop geometry, batch size, and schedule.

## Diagnostics

Recovery saves class features at initialization and block boundaries, and anchor identities/features at each injection. `tools/diagnose.py` reports shared-anchor fraction, synthetic-to-real feature spread, anchor-direction cosine median, integrated hard-Betti deficit, topology-gradient coverage, and weak-image fraction. Hard Betti numbers use the hard graph/Hodge operators. Gradients are measured on the same topology evaluations used during recovery.

`tools/efficiency.py` sums class recovery durations, including pool/reference preparation, and reads peak CUDA allocation. Teacher training, relabeling and student training are outside that timing. Run the full recovery on one process/device for a single-run memory measurement.
