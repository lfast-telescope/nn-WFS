# nn_WFS

Neural Curvature Wavefront Sensing (CWFS) for the LFAST telescope. Predicts Zernike wavefront error coefficients ($Z_4$–$Z_{36}$, metres OPD) from defocused intra- and extra-focal PSF pairs ($\pm \Delta z$).

For complete mathematical derivations, optical parameters, and architecture diagrams, see [OVERVIEW.md](file:///home/u7/warrenbfoster/git/nn_WFS/OVERVIEW.md).

---

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
pip install hcipy  # required for synthetic data generation
```

### 2. Generate Synthetic Training Data

Uses `hcipy` Fraunhofer propagation with multi-layer atmospheric turbulence:

```bash
python make_training_data.py --config config/data_generation.yaml --output data/cwfs_synthetic.h5
```

*Dry run (validate config without writing data):*
```bash
python make_training_data.py --config config/data_generation.yaml --dry-run
```

### 3. Train a Model

```bash
# Cross-Batch Roddier CNN (recommended for turbulent sequences)
python train.py --config config/rodcnn.yaml --hdf5_path data/cwfs_synthetic.h5

# Vision Transformer with Cross-Attention Fusion
python train.py --config config/transformer.yaml --hdf5_path data/cwfs_synthetic.h5

# Siamese ResNet
python train.py --config config/cnn.yaml --hdf5_path data/cwfs_synthetic.h5

# Lightweight Perceptron Baseline
python train.py --config config/toy.yaml --hdf5_path data/cwfs_synthetic.h5
```

*Override any config parameter on the CLI via `--section.key=value`:*
```bash
python train.py --config config/rodcnn.yaml \
    --hdf5_path data/cwfs_synthetic.h5 \
    --training.epochs=100 --training.lr=1e-4 --data.batch_size=16
```

### 4. Evaluate & Compare

```bash
# Evaluate checkpoint on test split (reports per-mode RMS nm, total WFE nm, Strehl proxy)
python evaluate.py --ckpt checkpoints/rodcnn/best.pt --hdf5_path data/cwfs_synthetic.h5

# Compare Transformer vs. CNN side-by-side
python evaluate.py --compare \
    --ckpt_transformer checkpoints/transformer/best.pt \
    --ckpt_cnn checkpoints/cnn/best.pt \
    --hdf5_path data/cwfs_synthetic.h5

# Roddier channel ablation (test with r active vs. r = 0)
python evaluate.py --ckpt checkpoints/transformer/best.pt --ablate_roddier --hdf5_path data/cwfs_synthetic.h5

# On-sky fine-tuning (backbone frozen, fine-tune MLP head)
python evaluate.py --ckpt checkpoints/rodcnn/best.pt --onsky_dir /path/to/onsky/ --fine_tune --ft_epochs 20
```

---

## Model Architectures

| Model Type | Config | Architecture | Input Mode |
|---|---|---|---|
| `rodcnn` | [config/rodcnn.yaml](file:///home/u7/warrenbfoster/git/nn_WFS/config/rodcnn.yaml) | Single-stream ResNet on all $T^2$ Roddier pairs $(I_{1,i} - I_{2,j}) / (I_{1,i} + I_{2,j} + \varepsilon)$; mean-pooled over $T^2$ with gradient accumulation. | Per-example stacks |
| `transformer` | [config/transformer.yaml](file:///home/u7/warrenbfoster/git/nn_WFS/config/transformer.yaml) | ViT patch encoder ($16\times 16$) + 2-stage cross-attention fusion ($I_1 \to I_2 \to r$) + MLP head. | `two_stream`, `r_stack`, `pairs` |
| `cnn` | [config/cnn.yaml](file:///home/u7/warrenbfoster/git/nn_WFS/config/cnn.yaml) | Siamese 4-stage ResNet backbone + cross-attention fusion + MLP head. | `two_stream`, `r_stack`, `pairs` |
| `toy` | [config/toy.yaml](file:///home/u7/warrenbfoster/git/nn_WFS/config/toy.yaml) | PatchEmbed ($16\times 16$) + token mean pooling + single hidden layer MLP (150 units). | `two_stream`, `r_stack`, `pairs` |

---

## Key Features

- **$D_4$ Dihedral Symmetry Augmentation:** Analytical $8\times$ dataset multiplication via 4 rotations $\times$ 2 reflections with exact $2\times 2$ Zernike block transformations.
- **Arbitrary Noll Subsetting:** Train on arbitrary pairing-complete Noll index subsets (e.g. `trained_modes: [4, 5, ..., 15]`) from a single HDF5 dataset without re-generation.
- **Mixed Precision & Compilation:** Native support for `torch.amp` (FP16) and `torch.compile` (`reduce-overhead` / `default`).
- **Physical Metrics:** Checkpoints and evaluations report physical nanometres RMS wavefront error and Maréchal Strehl ratio proxy.

---

## Repository Layout

```
nn_WFS/
├── make_training_data.py   — Polychromatic synthetic PSF generation with hcipy
├── dataset.py              — HDF5 data loading, lazy evaluation, splits
├── train.py                — Unified training script with checkpoint management
├── evaluate.py             — Evaluation, model comparison, on-sky fine-tuning
├── models/                 — Model implementations (transformer, cnn, rodcnn, toy)
├── config/                 — YAML configs for data generation and models
├── utils/                  — D4 symmetry augmentation and physical WFE metrics
└── OVERVIEW.md             — In-depth technical reference and derivations
```

