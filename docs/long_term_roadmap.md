# Long-Term Optimization Roadmap

**Starting point**: Winner of the LR × weight_decay matrix sweep (Phase 1).

Each phase below builds on the previous winner. The axes are ordered by expected impact and by dependency — optimizer dynamics must be settled before architecture changes, because capacity and regularisation interact.

---

## Phase 2 — AdamW Betas

AdamW's $\beta_1$ and $\beta_2$ control the momentum and second-moment EMA decay. The current defaults `(0.9, 0.999)` are PyTorch's out-of-the-box values and have never been tuned for this domain. With cosine annealing dropping LR by 600×, the long $\beta_2 = 0.999$ memory may make the optimizer sluggish in adapting to the shifting gradient landscape in the late low-LR phase. Lowering $\beta_2$ (e.g. 0.99 or 0.995) makes the denominator more responsive.

**Requires**: A small code change to expose `betas` as a configurable YAML parameter passed to `AdamW`.

**Estimated scope**: 4–6 trials, ~5 GPU-hours.

---

## Phase 3 — Total Epochs

The bake-off ran 30 epochs. Val WFE was still improving at epoch 28–30 (best seed hit 32.56 nm at epoch 30), suggesting the model hadn't fully converged. The default config specifies 100 epochs. We need to determine where diminishing returns set in.

Since cosine period = total epochs, longer training means a *slower* annealing schedule (more time at high LR), not just extra epochs at the floor. This fundamentally changes the optimisation trajectory.

**Estimated scope**: 3 trials at [50, 75, 100] epochs, ~8–16 GPU-hours.

---

## Phase 4 — Loss Formulation: LogMSE × Z-Score Labels

The loss function is currently standard MSE on z-scored labels. Two alternative formulations interact in a 2×2 matrix:

| `z_score_labels` | `log_loss` | Character |
|:---|:---|:---|
| true | false | **Current**: balanced mode weighting, MSE in z-space |
| true | true | Aggressive equalisation: log-compressed z-space, very uniform mode attention |
| false | false | Physical: MSE in nm OPD, defocus dominates the loss |
| false | true | Hybrid: log-compressed physical, moderate mode rebalancing |

LogMSE ($\ln(\text{MSE})$) compresses the dynamic range, putting relatively more gradient weight on already-well-predicted modes vs. high-error modes. This could help the network avoid "giving up" on small-signal high-order Zernike modes (Z14, Z15) where the physical amplitude is much smaller than defocus (Z4).

**Estimated scope**: 4 trials (the 2×2 matrix), ~4 GPU-hours.

---

## Phase 5 — Dropout

Currently `model.dropout = 0.1`, applied in the MLPHead regression layer. The backbone (ResNet stages) has no dropout. Given the consistently negative overfitting gap from the bake-off (val WFE < train WFE by ~8 nm — driven by the 8× variance reduction from full-pair validation averaging), explicit dropout may be redundant or even counterproductive. The stochastic regularisation from `k_pairs_train=8` subsampling already injects substantial noise.

However, the right dropout level depends on the final model capacity (Phase 4 loss may shift effective capacity requirements), so this should come after loss formulation is settled.

**Estimated scope**: 4 trials sweeping `dropout ∈ [0.0, 0.05, 0.1, 0.2]`, ~4 GPU-hours.

---

## Phase 6 — Architecture: `base_ch` × `stage_blocks`

These change model capacity directly. With ~2.5M parameters at `base_ch=32, stage_blocks=2`, the model may be capacity-limited (consistent with the negative overfitting gap) or appropriately sized. This can't be assessed until the regularisation recipe (wd, betas, dropout) and loss formulation are finalised.

- `base_ch ∈ [24, 32, 48]` scales width (~1.4M, ~2.5M, ~5.6M parameters)
- `stage_blocks ∈ [2, 3]` scales depth

> [!NOTE]
> `base_ch=48` roughly doubles parameter count. May need `batch_size` reduction for VRAM at higher `k_pairs_train` values.

**Estimated scope**: 6 trials (3×2 matrix), ~8–12 GPU-hours.

---

## Phase 7 — Cosine Warm Restarts

Once the recipe is finalised, warm restarts (SGDR / `CosineAnnealingWarmRestarts`) allow the model to periodically reset LR back to peak, escaping local minima. With $T_\text{mult} > 1$, each successive cycle is longer, progressively refining. This is most effective with longer training (Phase 3) and should be tested last because it fundamentally changes the training trajectory.

**Requires**: Extending the scheduler engine to support `CosineAnnealingWarmRestarts` alongside the existing `CosineAnnealingLR`.

**Estimated scope**: 4 trials, ~4–6 GPU-hours.

---

## Summary

| Phase | Axis | Depends On | Est. GPU-hours |
|:---|:---|:---|:---|
| 1 | LR × Weight Decay | — | ~21h |
| 2 | AdamW Betas | Phase 1 winner | ~5h |
| 3 | Total Epochs | Phase 2 winner | ~8–16h |
| 4 | LogMSE × Z-Score Labels | Phase 3 winner | ~4h |
| 5 | Dropout | Phase 4 winner | ~4h |
| 6 | Architecture (base_ch × stage_blocks) | Phase 5 winner | ~8–12h |
| 7 | Cosine Warm Restarts | Phase 6 winner | ~4–6h |

**Total estimated long-term budget**: ~55–65 GPU-hours across all phases, run sequentially over multiple Slurm jobs.

> [!TIP]
> **Free win to implement at any time**: Test-time D4 augmentation (TTA) and Stochastic Weight Averaging (SWA) are inference/post-training enhancements that require no retraining. These can be implemented and evaluated in parallel with any phase above.

