# Reference fixture generation

Create an isolated environment, install `requirements.txt`, and run:

```bash
python tests/reference_generation/generate_e3nn_reference.py
```

The script writes versioned `.npz` data and a JSON manifest under
`tests/reference_data/`.  Fixture updates must be reviewed; never regenerate
them implicitly as part of the test suite.

