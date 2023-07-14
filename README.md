# dilated-attention-pytorch

## Install

[**NOTE**: Install `xformers` according to their instructions.]

```bash
pip install "dilated-attention-pytorch @ git+ssh://git@github.com/fkodom/dilated-attention-pytorch.git"

# Install all dev dependencies (tests etc.)
pip install "dilated-attention-pytorch[all] @ git+ssh://git@github.com/fkodom/dilated-attention-pytorch.git"

# Setup pre-commit hooks
pre-commit install
```


## Test

Tests run automatically through GitHub Actions.
* Fast tests run on each push.
* Slow tests (decorated with `@pytest.mark.slow`) run on each PR.

You can also run tests manually with `pytest`:
```bash
pytest dilated-attention-pytorch

# For all tests, including slow ones:
pytest --slow dilated-attention-pytorch
```


## Release

[Optional] Requires either PyPI or Docker GHA workflows to be enabled.

Just tag a new release in this repo, and GHA will automatically publish Python wheels and/or Docker images.
