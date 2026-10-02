"""
Unit tests for the RadialHarmonicFeatures azimuthal Fourier input feature,
its wiring into RODCNN, and its train.py config resolution.
"""

import sys
import unittest
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nn_WFS.models.common import RadialHarmonicFeatures
    from nn_WFS.models.cnn_cwfs import RODCNN
    from nn_WFS.train import _resolve_radial_profile, build_model, _load_config
    from nn_WFS.utils.augmentation import noll_to_nm
except ImportError:
    from models.common import RadialHarmonicFeatures
    from models.cnn_cwfs import RODCNN
    from train import _resolve_radial_profile, build_model, _load_config
    from utils.augmentation import noll_to_nm


class TestRadialHarmonicFeaturesModule(unittest.TestCase):
    def test_output_shape(self):
        feat = RadialHarmonicFeatures(harmonics=[0, 1, 2], n_radial_bins=8)
        x = torch.randn(4, 1, 32, 32)
        out = feat(x)
        self.assertEqual(out.shape, (4, 3, 32, 32))

    def test_constant_input_reconstructed_as_m0(self):
        # A constant field's only azimuthal content is m=0; reconstruction
        # should reproduce the constant inside the pupil, zero outside.
        feat = RadialHarmonicFeatures(harmonics=[0], n_radial_bins=8, pupil_radius_frac=1.0)
        x = torch.full((1, 1, 32, 32), 3.0)
        out = feat(x)[0, 0]

        H, W = 32, 32
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        )
        inside = (xx.pow(2) + yy.pow(2)).sqrt() <= 1.0
        torch.testing.assert_close(out[inside], torch.full_like(out[inside], 3.0), atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.all(out[~inside] == 0))

    def test_single_harmonic_reconstruction(self):
        # A pure cos(theta) field should be reconstructed as itself (m=1
        # channel), verifying the Fourier normalization factor is correct.
        H, W = 64, 64
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        )
        theta = torch.atan2(yy, xx)
        r = torch.sqrt(xx.pow(2) + yy.pow(2))
        mask = r <= 1.0
        x = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        feat = RadialHarmonicFeatures(harmonics=[1], n_radial_bins=16, pupil_radius_frac=1.0)
        out = feat(x)[0, 0]
        expected = torch.where(mask, torch.cos(theta), torch.zeros_like(theta))
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)

    def test_pupil_mask_zeroes_outside(self):
        feat = RadialHarmonicFeatures(harmonics=[0, 1], n_radial_bins=8, pupil_radius_frac=0.5)
        x = torch.randn(2, 1, 32, 32)
        out = feat(x)

        H, W = 32, 32
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        )
        outside = (xx.pow(2) + yy.pow(2)).sqrt() > 0.5
        self.assertTrue(torch.all(out[:, :, outside] == 0))

    def test_gradient_flow(self):
        feat = RadialHarmonicFeatures(harmonics=[0, 1, 2], n_radial_bins=8)
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = feat(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_multiple_shapes_cached_independently(self):
        feat = RadialHarmonicFeatures(harmonics=[0, 1], n_radial_bins=8)
        out_small = feat(torch.randn(1, 1, 32, 32))
        out_large = feat(torch.randn(1, 1, 256, 256))
        self.assertEqual(out_small.shape, (1, 2, 32, 32))
        self.assertEqual(out_large.shape, (1, 2, 256, 256))


class TestRODCNNRadialProfileWiring(unittest.TestCase):
    def test_disabled_default_backward_compatible(self):
        model = RODCNN(base_ch=16, stage_blocks=1, stem_stride=2, n_outputs=14)
        self.assertIsNone(model.radial_features)
        self.assertEqual(model.backbone.stem[0].in_channels, 1)

    def test_enabled_in_channels(self):
        model = RODCNN(
            base_ch=16, stage_blocks=1, stem_stride=2, n_outputs=14,
            radial_profile={'harmonics': [0, 1, 2, 3], 'n_radial_bins': 8},
        )
        self.assertEqual(model.backbone.stem[0].in_channels, 5)

    def test_forward_backward_enabled(self):
        B, T, H, W = 2, 4, 32, 32
        model = RODCNN(
            base_ch=16, stage_blocks=1, stem_stride=2, n_outputs=14,
            radial_profile={'harmonics': [0, 1, 2], 'n_radial_bins': 8},
        )
        I1 = torch.randn(B, T, 1, H, W, requires_grad=True)
        I2 = torch.randn(B, T, 1, H, W, requires_grad=True)
        out = model(I1, I2, k_pairs=4)
        self.assertEqual(out.shape, (B, 14))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(I1.grad)
        self.assertFalse(torch.isnan(I1.grad).any())


class TestResolveRadialProfile(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertIsNone(_resolve_radial_profile({}))
        cfg = {'model': {'radial_profile': {'enabled': False}}}
        self.assertIsNone(_resolve_radial_profile(cfg))

    def test_auto_derives_from_trained_modes(self):
        trained_modes = [4, 5, 6, 7, 8]  # defocus, astig x2, coma x2
        expected = sorted({abs(noll_to_nm(j)[1]) for j in trained_modes})
        cfg = {
            'model': {
                'trained_modes': trained_modes,
                'radial_profile': {'enabled': True, 'harmonics': 'auto'},
            }
        }
        resolved = _resolve_radial_profile(cfg)
        self.assertEqual(list(resolved['harmonics']), expected)

    def test_auto_falls_back_to_z4_z36_when_trained_modes_unresolved(self):
        expected = sorted({abs(noll_to_nm(j)[1]) for j in range(4, 37)})
        cfg = {'model': {'radial_profile': {'enabled': True}}}  # harmonics defaults to 'auto'
        resolved = _resolve_radial_profile(cfg)
        self.assertEqual(list(resolved['harmonics']), expected)

    def test_explicit_harmonics_list(self):
        cfg = {'model': {'radial_profile': {'enabled': True, 'harmonics': [0, 2, 4]}}}
        resolved = _resolve_radial_profile(cfg)
        self.assertEqual(resolved['harmonics'], (0, 2, 4))

    def test_override_string_harmonics_parsed(self):
        cfg = {'model': {'radial_profile': {'enabled': True, 'harmonics': '[0,1,2,3]'}}}
        resolved = _resolve_radial_profile(cfg)
        self.assertEqual(resolved['harmonics'], (0, 1, 2, 3))

    def test_defaults_for_bins_and_pupil_frac(self):
        cfg = {'model': {'radial_profile': {'enabled': True, 'harmonics': [0]}}}
        resolved = _resolve_radial_profile(cfg)
        self.assertEqual(resolved['n_radial_bins'], 16)
        self.assertEqual(resolved['pupil_radius_frac'], 1.0)

    def test_build_model_backward_compatibility(self):
        rodcnn_yaml = _REPO / 'config' / 'rodcnn.yaml'
        if rodcnn_yaml.exists():
            cfg = _load_config(str(rodcnn_yaml))
            model = build_model(cfg)
            # avoid isinstance: build_model imports RODCNN via a bare 'models.cnn_cwfs'
            # module path, which is a distinct module identity from 'nn_WFS.models.cnn_cwfs'
            self.assertEqual(model.__class__.__name__, 'RODCNN')
            self.assertIsNone(model.radial_features)
            self.assertEqual(model.backbone.stem[0].in_channels, 1)


if __name__ == '__main__':
    unittest.main()
