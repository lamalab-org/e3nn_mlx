# Installation

## Requirements

`e3nn-mlx` requires Python 3.11 or newer. Apple silicon Macs use MLX's Metal
backend. Linux installations use the MLX CPU package and are primarily useful
for testing and portability; generated Metal kernels are unavailable there.

## Install from PyPI

```bash
python -m pip install e3nn-mlx
```

Verify the installation:

```bash
python -c "import e3nn_mlx; print(e3nn_mlx.__version__)"
```

## Install for development

```bash
git clone https://github.com/lamalab-org/e3nn_mlx.git
cd e3nn_mlx
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,docs]'
python -m pytest
```

MLX tests that require a working runtime can otherwise skip. CI prevents an
accidental pass by setting `E3NN_MLX_REQUIRE_RUNTIME=1`.

## Select an execution path

Generated kernels are enabled by default on compatible Apple-silicon inputs.
Every public operation retains a general MLX path. For spherical harmonics and
scatter, pass `use_custom_kernel=False`; for tensor products, construct the
module with `use_custom_kernel=False` or call `differentiable_arrays` when a
forward-mode transform is needed.
