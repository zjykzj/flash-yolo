# Contributing to Flash-YOLO

Thanks for your interest. Issues and pull requests are welcome.

For substantial changes, open an issue first so the approach can be discussed before you invest in
an implementation. Small fixes, docs and tests can go straight to a pull request.

## Development setup

```bash
git clone https://github.com/zjykzj/flash-yolo.git
cd flash-yolo
pip install -r requirements.txt        # runtime: torch + numpy + opencv + onnx tooling
pip install -r requirements-dev.txt    # adds pytest + ultralytics (dev-time reference only)
```

`ultralytics` is used only for development-time numeric comparison and checkpoint conversion; the
runtime never imports it. A Docker environment with CUDA 13 + TensorRT 11.4 is described in
[docs/docker.md](docs/docker.md).

## Tests

```bash
pytest tests/ -q    # CPU core suite, the same command CI runs
```

Some suites are gated by environment variables:

```bash
FLASH_YOLO_SMOKE=1 pytest tests/test_train_smoke.py -q   # end-to-end train/export smoke, CPU ~2-5 min
FLASH_YOLO_REPRO=1 pytest tests/test_train_repro.py -q   # bit-reproducible training (GPU recommended)
FLASH_YOLO_DDP=1   pytest tests/test_distributed.py -q   # 2-process CPU gloo DDP
```

CI (`.github/workflows/tests.yml`) runs the ungated core suite on every push to `main` and every
pull request.

## Code guidelines

- Keep every module independent and readable on its own; prefer explicit code over framework-style
  abstraction.
- Do not copy code from the upstream ultralytics repository (AGPL-3.0). Reimplement from the paper
  and checkpoint structure facts; an ultralytics checkout is a read-only reference for numeric
  comparison.
- Add or update tests for every behavior change. Bit-level parity and loss parity tests are the
  regression gates — run the relevant ones before opening a pull request.
- Printed output (logs, exceptions, argparse help) is English; code comments are Chinese-English
  mixed; documentation under `docs/` is Chinese.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes (Added / Changed / Fixed /
  Removed), following the existing entry format.

## Commit messages

Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `refactor:`, ...), with an
English subject and body. Keep the subject imperative and concise; put root cause, measurements and
trade-offs in the body. See `git log` for the established style.

## Pull requests

- Describe what changed and why; link the related issue when there is one.
- List the tests you ran and their result.
- Note any behavior, configuration or weight-compatibility changes.
- By contributing, you agree that your contributions are licensed under the repository's
  Apache-2.0 license.
