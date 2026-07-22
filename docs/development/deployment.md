# Building and deploying the documentation

## Build locally

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html
```

The equivalent short command is `make -C docs html`. Warnings are treated as
errors locally and in CI so broken API imports and references cannot silently
reach the website.

## Deploy with GitHub Pages

The repository includes `.github/workflows/docs.yml`. Pull requests build the
site as a required-quality check but do not deploy it. A push to `main` that
changes documentation, package source, or documentation configuration builds
and publishes the Pages artifact.

One repository setting must be enabled by an administrator:

1. Open **Settings → Pages** in `lamalab-org/e3nn_mlx`.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Merge this branch into `main`, or manually run the **Documentation** workflow
   from the Actions tab.
4. After the deploy job succeeds, the site is available at
   `https://lamalab-org.github.io/e3nn_mlx/`.

The workflow grants `pages: write` and `id-token: write` only to the deployment
job and uses GitHub's `github-pages` environment. Add a deployment protection
rule for `main` if the organization requires one.

## Alternative: Read the Docs

Read the Docs can also build the project with `python -m pip install -e
'.[docs]'` and `docs/conf.py`. GitHub Pages is the configured default because
it needs no additional service account and the project already uses GitHub
Actions.
