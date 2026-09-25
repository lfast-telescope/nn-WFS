"""
Unit tests for configurable stem_stride in ResNetBackbone, RODCNN, and SIAMCNN.
"""

import unittest
import torch
from nn_WFS.models.cnn_cwfs import ResNetBackbone, RODCNN, SIAMCNN
from nn_WFS.train import build_model


class TestStemStride(unittest.TestCase):
    def test_resnet_backbone_stride1(self):
        bb = ResNetBackbone(base_ch=32, stage_blocks=2, stem_stride=1)
        x = torch.randn(2, 1, 256, 256)
        out = bb(x)
        self.assertEqual(out.shape, (2, 256, 16, 16))

    def test_resnet_backbone_stride2(self):
        bb = ResNetBackbone(base_ch=32, stage_blocks=2, stem_stride=2)
        x = torch.randn(2, 1, 256, 256)
        out = bb(x)
        self.assertEqual(out.shape, (2, 256, 8, 8))

    def test_invalid_stem_stride(self):
        with self.assertRaises(ValueError):
            ResNetBackbone(base_ch=32, stage_blocks=2, stem_stride=3)

    def test_rodcnn_forward_subsampled(self):
        model = RODCNN(base_ch=32, stage_blocks=2, n_outputs=33, stem_stride=2)
        model.train()
        I1 = torch.randn(2, 8, 1, 256, 256)
        I2 = torch.randn(2, 8, 1, 256, 256)
        out = model(I1, I2, k_pairs=16)
        self.assertEqual(out.shape, (2, 33))

    def test_rodcnn_forward_full(self):
        model = RODCNN(base_ch=32, stage_blocks=2, n_outputs=33, stem_stride=2)
        model.eval()
        I1 = torch.randn(2, 8, 1, 256, 256)
        I2 = torch.randn(2, 8, 1, 256, 256)
        out = model(I1, I2, k_pairs=None)
        self.assertEqual(out.shape, (2, 33))

    def test_rodcnn_backward(self):
        model = RODCNN(base_ch=32, stage_blocks=2, n_outputs=33, stem_stride=2)
        model.train()
        I1 = torch.randn(2, 8, 1, 256, 256)
        I2 = torch.randn(2, 8, 1, 256, 256)
        target = torch.randn(2, 33)
        out = model(I1, I2, k_pairs=16)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        stem_weight = model.backbone.stem[0].weight
        self.assertIsNotNone(stem_weight.grad)
        self.assertTrue(torch.isfinite(stem_weight.grad).all())

    def test_siamcnn_forward(self):
        model = SIAMCNN(base_ch=32, stage_blocks=2, n_outputs=14, input_mode='two_stream', stem_stride=2)
        I1 = torch.randn(2, 1, 256, 256)
        I2 = torch.randn(2, 1, 256, 256)
        out = model(I1, I2)
        self.assertEqual(out.shape, (2, 14))

    def test_build_model_integration(self):
        cfg = {
            'model': {
                'type': 'rodcnn',
                'base_ch': 32,
                'stage_blocks': 2,
                'n_outputs': 33,
                'stem_stride': 2,
            }
        }
        model = build_model(cfg)
        self.assertEqual(model.__class__.__name__, 'RODCNN')
        self.assertEqual(model.backbone.stem_stride, 2)


if __name__ == '__main__':
    unittest.main()
