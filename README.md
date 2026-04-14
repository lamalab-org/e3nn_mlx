# e3nn-mlx

Incremental refactor of e3nn into a backend-neutral core plus an MLX-native runtime.

This repository currently contains steps 0-3 of the migration plan:

- porting map
- backend-agnostic `e3nn_core`
- thin backend protocol in `e3nn_backend`
- MLX backend skeleton in `e3nn_mlx`

The numerical MLX operators are intentionally left as TODOs until the static metadata
and dispatch seams are stable.
