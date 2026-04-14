from __future__ import annotations

import unittest

from e3nn_core.irreps import Irrep, Irreps, MulIrrep


class IrrepsTest(unittest.TestCase):
    def test_parse_irrep(self) -> None:
        ir = Irrep.parse("2o")
        self.assertEqual(ir.l, 2)
        self.assertEqual(ir.p, -1)
        self.assertEqual(ir.dim, 5)

    def test_parse_irreps(self) -> None:
        irreps = Irreps("2x0e + 1o + 3x2e")
        self.assertEqual(str(irreps), "2x0e+1o+3x2e")
        self.assertEqual(irreps.dim, 2 + 3 + 15)
        self.assertEqual(irreps.num_irreps, 6)
        self.assertEqual(irreps.lmax, 2)

    def test_regroup_and_sort(self) -> None:
        irreps = Irreps([MulIrrep(1, Irrep(1, -1)), MulIrrep(2, Irrep(0, 1)), MulIrrep(3, Irrep(1, -1))])
        self.assertEqual(str(irreps.regroup()), "1o+2x0e+3x1o")
        self.assertEqual(str(irreps.sort()), "2x0e+4x1o")

    def test_slices(self) -> None:
        irreps = Irreps("0e + 2x1o")
        self.assertEqual(irreps.slices(), (slice(0, 1), slice(1, 7)))

    def test_selection_rule(self) -> None:
        out = Irrep.parse("1o") * Irrep.parse("1o")
        self.assertEqual(tuple(str(ir) for ir in out), ("0e", "1e", "2e"))


if __name__ == "__main__":
    unittest.main()
