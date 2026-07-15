from __future__ import annotations

import unittest

from e3nn_core.instructions import generate_tensor_product_instructions, make_tensor_product_instructions


class InstructionsTest(unittest.TestCase):
    def test_symbolic_tensor_product_generation(self) -> None:
        instructions = generate_tensor_product_instructions("1o", "1o")
        self.assertEqual(len(instructions), 3)
        self.assertEqual([str(inst.ir_out) for inst in instructions], ["0e", "1e", "2e"])
        self.assertEqual(instructions[0].path_shape, (1, 1, 1))

    def test_output_filtering(self) -> None:
        instructions = generate_tensor_product_instructions("2x0e + 1o", "1o", "0e + 2x1o + 2e")
        self.assertEqual([str(inst.ir_out) for inst in instructions], ["1o", "0e", "2e"])
        self.assertEqual(instructions[0].output_index, 1)
        self.assertEqual(instructions[-1].path_shape, (1, 1, 1))

    def test_output_multiplicity_is_reflected_in_path_shape(self) -> None:
        instructions = generate_tensor_product_instructions("1o", "1o", "2x0e + 1e + 2e")
        self.assertEqual([inst.path_shape for inst in instructions], [(1, 1, 2), (1, 1, 1), (1, 1, 1)])

    def test_duplicate_output_chunks_receive_distinct_paths(self) -> None:
        instructions = generate_tensor_product_instructions("1o", "1o", "0e+2x0e")
        self.assertEqual(tuple(instruction.output_index for instruction in instructions), (0, 1))
        self.assertEqual(tuple(instruction.path_shape for instruction in instructions), ((1, 1, 1), (1, 1, 2)))

    def test_explicit_instruction_preserves_weight_metadata(self) -> None:
        instruction = make_tensor_product_instructions(
            "1o", "1o", "0e", [(0, 0, 0, "uvw", True, 0.25)]
        )[0]
        self.assertTrue(instruction.has_weight)
        self.assertEqual(instruction.path_weight, 0.25)


if __name__ == "__main__":
    unittest.main()
