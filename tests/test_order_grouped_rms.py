import unittest
import math
import numpy as np
import torch

from nn_WFS.utils.metrics import (
    noll_radial_order,
    group_modes_by_radial_order,
    format_order_grouped_rms,
)


class TestOrderGroupedRMS(unittest.TestCase):
    def test_noll_radial_order_analytical(self):
        """Test noll_radial_order matches analytical values across orders 0 through 8."""
        for n in range(0, 9):
            start_j = n * (n + 1) // 2 + 1
            end_j = (n + 1) * (n + 2) // 2
            for j in range(start_j, end_j + 1):
                self.assertEqual(
                    noll_radial_order(j),
                    n,
                    f"Failed for j={j}, expected order {n}, got {noll_radial_order(j)}",
                )

    def test_noll_radial_order_invalid(self):
        """Test that invalid Noll index j < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            noll_radial_order(0)
        with self.assertRaises(ValueError):
            noll_radial_order(-1)

    def test_group_modes_by_radial_order_z4_to_z36(self):
        """Test grouping for full Z4–Z36 set (orders 2 to 7)."""
        trained_modes = list(range(4, 37))  # 33 modes
        # Assign mock RMS values in metres
        mode_rms = [1e-8 * j for j in trained_modes]  # e.g. 40nm, 50nm, ...

        grouped = group_modes_by_radial_order(trained_modes, mode_rms, scale=1e9)
        self.assertEqual(len(grouped), 6)  # orders 2, 3, 4, 5, 6, 7

        # Order 2: j = 4, 5, 6 -> 3 modes: 40, 50, 60 nm
        g2 = grouped[0]
        self.assertEqual(g2['order'], 2)
        self.assertEqual(g2['modes'], [4, 5, 6])
        self.assertEqual(g2['modes_str'], "Z4–Z6")
        self.assertEqual(g2['n_modes'], 3)
        self.assertAlmostEqual(g2['min'], 40.0)
        self.assertAlmostEqual(g2['max'], 60.0)
        self.assertAlmostEqual(g2['avg'], 50.0)

        # Order 7: j = 29..36 -> 8 modes: 290..360 nm
        g7 = grouped[5]
        self.assertEqual(g7['order'], 7)
        self.assertEqual(g7['modes'], list(range(29, 37)))
        self.assertEqual(g7['modes_str'], "Z29–Z36")
        self.assertEqual(g7['n_modes'], 8)
        self.assertAlmostEqual(g7['min'], 290.0)
        self.assertAlmostEqual(g7['max'], 360.0)
        self.assertAlmostEqual(g7['avg'], 325.0)

    def test_group_modes_by_radial_order_subset(self):
        """Test grouping on non-contiguous subset of modes."""
        trained_modes = [4, 6, 7, 10]
        # in metres: 10nm, 30nm, 20nm, 40nm
        mode_rms = [10e-9, 30e-9, 20e-9, 40e-9]

        grouped = group_modes_by_radial_order(trained_modes, mode_rms, scale=1e9)
        self.assertEqual(len(grouped), 2)

        # Order 2: modes 4 and 6 (non-contiguous)
        self.assertEqual(grouped[0]['order'], 2)
        self.assertEqual(grouped[0]['modes'], [4, 6])
        self.assertEqual(grouped[0]['modes_str'], "Z4, Z6")
        self.assertEqual(grouped[0]['n_modes'], 2)
        self.assertAlmostEqual(grouped[0]['min'], 10.0)
        self.assertAlmostEqual(grouped[0]['max'], 30.0)
        self.assertAlmostEqual(grouped[0]['avg'], 20.0)

        # Order 3: modes 7 and 10 (non-contiguous)
        self.assertEqual(grouped[1]['order'], 3)
        self.assertEqual(grouped[1]['modes'], [7, 10])
        self.assertEqual(grouped[1]['modes_str'], "Z7, Z10")
        self.assertEqual(grouped[1]['n_modes'], 2)
        self.assertAlmostEqual(grouped[1]['min'], 20.0)
        self.assertAlmostEqual(grouped[1]['max'], 40.0)
        self.assertAlmostEqual(grouped[1]['avg'], 30.0)

    def test_group_modes_tensor_input(self):
        """Test accepting torch.Tensor as mode_rms input."""
        trained_modes = [4, 5, 6]
        mode_rms = torch.tensor([12e-9, 15e-9, 18e-9])
        grouped = group_modes_by_radial_order(trained_modes, mode_rms, scale=1e9)
        self.assertEqual(len(grouped), 1)
        self.assertAlmostEqual(grouped[0]['avg'], 15.0, places=5)
        self.assertAlmostEqual(grouped[0]['min'], 12.0, places=5)
        self.assertAlmostEqual(grouped[0]['max'], 18.0, places=5)

    def test_length_mismatch_raises(self):
        """Test length mismatch raises ValueError."""
        with self.assertRaises(ValueError):
            group_modes_by_radial_order([4, 5], [10.0])

    def test_format_order_grouped_rms(self):
        """Test string formatting output contains avg, min, and max."""
        trained_modes = [4, 5, 6, 7, 8, 9, 10]
        mode_rms = [12.0e-9, 14.0e-9, 16.0e-9, 20.0e-9, 22.0e-9, 24.0e-9, 26.0e-9]

        lines = format_order_grouped_rms(trained_modes, mode_rms, scale=1e9)
        self.assertEqual(len(lines), 2)  # order 2 and order 3

        # Check order 2 line
        self.assertIn("Order n=2", lines[0])
        self.assertIn("Z4–Z6", lines[0])
        self.assertIn("3 modes", lines[0])
        self.assertIn("avg= 14.0", lines[0])
        self.assertIn("min= 12.0", lines[0])
        self.assertIn("max= 16.0", lines[0])

        # Check order 3 line
        self.assertIn("Order n=3", lines[1])
        self.assertIn("Z7–Z10", lines[1])
        self.assertIn("4 modes", lines[1])
        self.assertIn("avg= 23.0", lines[1])
        self.assertIn("min= 20.0", lines[1])
        self.assertIn("max= 26.0", lines[1])


if __name__ == "__main__":
    unittest.main()
