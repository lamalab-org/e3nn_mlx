# Releasing to PyPI

Releases use a version stored in `e3nn_mlx/_version.py`, a `v`-prefixed Git
tag, and PyPI Trusted Publishing. The tag and package versions must match.

## One-time PyPI setup

Create the `e3nn-mlx` project or a pending trusted publisher on PyPI with:

- GitHub owner: `lamalab-org`
- repository: `e3nn_mlx`
- workflow: `publish.yml`
- environment: `pypi`

Create a protected GitHub environment named `pypi`. No PyPI API token is
stored in GitHub.

## Prepare a release

1. Update `__version__` in `e3nn_mlx/_version.py`.
2. Add and commit the intended release notes.
3. Run the tests and package checks:

   ```bash
   python -m pip install -e '.[test,release]'
   python -m pytest
   python -m build
   python -m twine check dist/*
   check-wheel-contents dist/*.whl
   ```

4. Inspect the distributions and install the wheel in a clean environment.
5. Commit the release, merge it to `main`, and create a matching tag:

   ```bash
   git tag -a v0.1.0 -m "e3nn-mlx 0.1.0"
   git push origin v0.1.0
   ```

Pushing the tag starts `.github/workflows/publish.yml`. Its publish job uses
the protected `pypi` environment and OpenID Connect credentials. Do not upload
a distribution built from a dirty working tree or reuse a version already
present on PyPI.
