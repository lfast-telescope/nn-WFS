from contextlib import contextmanager
from typing import Dict
import torch
import torch.nn as nn


class ModelSWA:
    """
    Stochastic Weight Averaging (SWA) accumulator for post-optimization model averaging.

    Maintains a uniform running arithmetic mean across epochs:
        θ_SWA = (n · θ_SWA + θ) / (n + 1)

    Tracks parameters and buffers (BatchNorm running statistics) from the model.
    Context manager `apply_shadow(model)` temporarily swaps SWA weights into
    the model for evaluation, validation, or checkpoint saving, safely restoring
    active training weights on exit.
    """

    def __init__(self, model: nn.Module):
        raw_model = getattr(model, '_orig_mod', model)
        self.n_models = 0
        self.params: Dict[str, torch.Tensor] = {
            name: p.clone().detach()
            for name, p in raw_model.named_parameters()
        }
        self.buffers: Dict[str, torch.Tensor] = {
            name: b.clone().detach()
            for name, b in raw_model.named_buffers()
        }

    @property
    def shadow_params(self) -> Dict[str, torch.Tensor]:
        return self.params

    def reset(self, model: nn.Module | None = None) -> None:
        """Reset the accumulator to zero models."""
        self.n_models = 0
        if model is not None:
            raw_model = getattr(model, '_orig_mod', model)
            self.params = {
                name: p.clone().detach()
                for name, p in raw_model.named_parameters()
            }
            self.buffers = {
                name: b.clone().detach()
                for name, b in raw_model.named_buffers()
            }

    def update(self, model: nn.Module) -> None:
        """
        Accumulate model parameters into the running SWA average.
        Buffers are copied from the latest model to preserve valid running stats.
        """
        raw_model = getattr(model, '_orig_mod', model)
        self.n_models += 1
        alpha = 1.0 / float(self.n_models)
        with torch.no_grad():
            for name, p in raw_model.named_parameters():
                if name in self.params:
                    # lerp_(target, weight) computes self = self + weight * (target - self)
                    self.params[name].lerp_(p.detach().to(self.params[name].device), alpha)
            for name, b in raw_model.named_buffers():
                if name in self.buffers:
                    self.buffers[name].copy_(b.detach().to(self.buffers[name].device))

    @contextmanager
    def apply_shadow(self, model: nn.Module):
        """
        Temporarily swaps SWA weights into `model` for evaluation or saving.
        Restores active training weights upon exiting the context.
        """
        if self.n_models == 0:
            yield
            return

        raw_model = getattr(model, '_orig_mod', model)
        backup_params = {name: p.clone() for name, p in raw_model.named_parameters()}
        backup_buffers = {name: b.clone() for name, b in raw_model.named_buffers()}
        try:
            with torch.no_grad():
                for name, p in raw_model.named_parameters():
                    if name in self.params:
                        p.copy_(self.params[name].to(p.device))
                for name, b in raw_model.named_buffers():
                    if name in self.buffers:
                        b.copy_(self.buffers[name].to(b.device))
            yield
        finally:
            with torch.no_grad():
                for name, p in raw_model.named_parameters():
                    if name in backup_params:
                        p.copy_(backup_params[name])
                for name, b in raw_model.named_buffers():
                    if name in backup_buffers:
                        b.copy_(backup_buffers[name])

    def state_dict(self) -> dict:
        return {
            'n_models': self.n_models,
            'params': {k: v.cpu() for k, v in self.params.items()},
            'buffers': {k: v.cpu() for k, v in self.buffers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_models = state.get('n_models', 0)
        if 'params' in state:
            for k, v in state['params'].items():
                if k in self.params:
                    self.params[k].copy_(v.to(self.params[k].device))
        if 'buffers' in state:
            for k, v in state['buffers'].items():
                if k in self.buffers:
                    self.buffers[k].copy_(v.to(self.buffers[k].device))
