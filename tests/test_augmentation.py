import unittest
import numpy as np
import torch

from nn_WFS.utils.augmentation import (
    noll_to_nm,
    validate_trained_modes_pairing,
    build_d4_label_matrices,
    D4Augment,
    _apply_image_op,
)


class TestD4Augmentation(unittest.TestCase):
    def setUp(self):
        # Default active training set: Z4 to Z15 (12 modes)
        self.trained_modes = list(range(4, 16))
        self.n_modes = len(self.trained_modes)

    def test_noll_to_nm_canonical(self):
        """Verify Noll indexing matches standard optical conventions."""
        expected = {
            2: (1, 1),    # Tip (cos)
            3: (1, -1),   # Tilt (sin)
            4: (2, 0),    # Defocus (radial)
            5: (2, -2),   # Oblique astigmatism (sin)
            6: (2, 2),    # Vertical astigmatism (cos)
            7: (3, -1),   # Vertical coma (sin)
            8: (3, 1),    # Horizontal coma (cos)
            9: (3, -3),   # Trefoil (sin)
            10: (3, 3),   # Trefoil (cos)
            11: (4, 0),   # Primary spherical (radial)
            12: (4, 2),   # Secondary astigmatism (cos)
            13: (4, -2),  # Secondary astigmatism (sin)
            14: (4, 4),   # Tetrafoil (cos)
            15: (4, -4),  # Tetrafoil (sin)
        }
        for j, (n_exp, m_exp) in expected.items():
            n, m = noll_to_nm(j)
            self.assertEqual((n, m), (n_exp, m_exp), f"Mode Z{j}: got ({n}, {m}), expected ({n_exp}, {m_exp})")

    def test_pairing_validation(self):
        """Verify pairing completeness checks for Zernike mode subsets."""
        # 1. Complete sets must pass without error
        validate_trained_modes_pairing(list(range(4, 16)))
        validate_trained_modes_pairing(list(range(2, 16)))
        validate_trained_modes_pairing([4, 11])  # Only radial modes

        # 2. Incomplete pairs must raise ValueError
        # Missing Z6 (cos partner of Z5)
        with self.assertRaises(ValueError):
            validate_trained_modes_pairing([4, 5, 7, 8])
        # Missing Z7 (sin partner of Z8)
        with self.assertRaises(ValueError):
            validate_trained_modes_pairing([4, 6, 8])

    def test_group_algebraic_invariants(self):
        """Verify D4 matrices satisfy orthogonality, integer entries, and determinants."""
        matrices = build_d4_label_matrices(self.trained_modes)
        self.assertEqual(len(matrices), 8, "D4 must produce exactly 8 group transformation matrices")

        I = torch.eye(self.n_modes, dtype=torch.float32)

        for i, M in enumerate(matrices):
            # Orthogonality / Energy conservation: M^T @ M = I
            self.assertTrue(
                torch.allclose(M.T @ M, I, atol=1e-6),
                f"Matrix {i} is not orthogonal (WFE variance not conserved)",
            )

            # Integer representation in {-1, 0, 1}
            is_int_elem = torch.all(torch.isin(M, torch.tensor([-1.0, 0.0, 1.0])))
            self.assertTrue(is_int_elem, f"Matrix {i} contains non-integer elements")

            # Determinants: +1 for rotations (0..3), -1 for reflections (4..7)
            det = torch.linalg.det(M).item()
            expected_det = 1.0 if i < 4 else -1.0
            self.assertAlmostEqual(
                det, expected_det, places=4,
                msg=f"Matrix {i} has det={det}, expected {expected_det}",
            )

    def test_group_generators_and_closure(self):
        """Verify group generator presentation: R^4 = I, F^2 = I, and F R F = R^-1."""
        matrices = build_d4_label_matrices(self.trained_modes)
        I = torch.eye(self.n_modes, dtype=torch.float32)

        M_rot90  = matrices[1]
        M_rot180 = matrices[2]
        M_rot270 = matrices[3]
        M_flip   = matrices[4]

        # R^4 = I
        R4 = torch.linalg.matrix_power(M_rot90, 4)
        self.assertTrue(torch.allclose(R4, I, atol=1e-6), "R^4 != I")

        # F^2 = I
        F2 = torch.linalg.matrix_power(M_flip, 2)
        self.assertTrue(torch.allclose(F2, I, atol=1e-6), "F^2 != I")

        # Dihedral relation: F @ R @ F = R^-1 = R^3
        FRF = M_flip @ M_rot90 @ M_flip
        self.assertTrue(torch.allclose(FRF, M_rot270, atol=1e-6), "F R F != R^-1 (M_rot270)")

        # Group closure: product of any two matrices must equal another matrix in the set
        for i, M_a in enumerate(matrices):
            for j, M_b in enumerate(matrices):
                prod = M_a @ M_b
                found = any(torch.allclose(prod, M_k, atol=1e-6) for M_k in matrices)
                self.assertTrue(found, f"Product of matrices {i} and {j} not in D4 group")

    def test_physical_mode_transformations(self):
        """Verify exact physical behavior for radial, astigmatism, and coma modes."""
        matrices = build_d4_label_matrices(self.trained_modes)
        idx = {m: i for i, m in enumerate(self.trained_modes)}

        # 1. Radially symmetric modes (Z4 defocus, Z11 spherical) must be invariant under all 8 ops
        for i, M in enumerate(matrices):
            self.assertEqual(M[idx[4], idx[4]].item(), 1.0, f"Z4 altered in op {i}")
            self.assertEqual(M[idx[11], idx[11]].item(), 1.0, f"Z11 altered in op {i}")

        # 2. Astigmatism m=2 (Z5, Z6): 90° rotation doubles to 180°, so both negate
        M_rot90 = matrices[1]
        self.assertEqual(M_rot90[idx[5], idx[5]].item(), -1.0)
        self.assertEqual(M_rot90[idx[6], idx[6]].item(), -1.0)

        # 3. Coma m=1 (Z7 vertical/sin, Z8 horizontal/cos):
        # 90° CCW rotation: (x, y) -> (-y, x)
        # horizontal coma (cos) transforms to vertical coma (sin): Z8 -> Z7, Z7 -> -Z8
        # [Z6_cos, Z5_sin]: here [Z8_cos, Z7_sin]
        # M @ [Z8, Z7]^T: Z8_new = -Z7, Z7_new = Z8
        self.assertEqual(M_rot90[idx[7], idx[8]].item(), 1.0)
        self.assertEqual(M_rot90[idx[8], idx[7]].item(), -1.0)

    def test_curvature_image_commutation(self):
        """Verify image-space transform commutes with Roddier curvature calculation."""
        augment = D4Augment(self.trained_modes, p=1.0)

        torch.manual_seed(42)
        I1 = torch.rand(8, 256, 256, dtype=torch.float32) + 0.1
        I2 = torch.rand(8, 256, 256, dtype=torch.float32) + 0.1
        r = (I1 - I2) / (I1 + I2 + 1e-8)
        labels = torch.randn(self.n_modes, dtype=torch.float32)

        sample = {'I1': I1, 'I2': I2, 'r': r, 'labels': labels}

        # Run several stochastic augmentations
        for _ in range(10):
            out = augment(sample)
            # 1. Commutation: r(T(I1), T(I2)) == T(r(I1, I2))
            r_from_transformed_I = (out['I1'] - out['I2']) / (out['I1'] + out['I2'] + 1e-8)
            self.assertTrue(
                torch.allclose(out['r'], r_from_transformed_I, atol=1e-5),
                "Roddier curvature does not commute with spatial augmentation",
            )
            # 2. Label norm conservation
            self.assertAlmostEqual(
                out['labels'].norm().item(),
                labels.norm().item(),
                places=5,
                msg="Label L2 norm (WFE energy) changed after augmentation",
            )

    def test_round_trip_image_operations(self):
        """Verify 4 rotations or 2 flips restore the original image tensor bit-for-bit."""
        img = torch.randn(8, 256, 256)

        # 4 × 90° rotations = original image
        rot4 = _apply_image_op(img, flip=False, rot_k=0)
        for _ in range(4):
            rot4 = _apply_image_op(rot4, flip=False, rot_k=1)
        self.assertTrue(torch.allclose(rot4, img), "4x rot90 did not restore original tensor")

        # 2 × flips = original image
        flip2 = _apply_image_op(img, flip=True, rot_k=0)
        flip2 = _apply_image_op(flip2, flip=True, rot_k=0)
        self.assertTrue(torch.allclose(flip2, img), "2x flip did not restore original tensor")


if __name__ == '__main__':
    unittest.main()

