# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics `xview-yolov3` (AGPL-3.0) trains a YOLOv3 object detector on the [xView](https://challenge.xviewdataset.org/) satellite imagery dataset for the xView Detection Challenge. It is a self-contained Darknet-config PyTorch implementation that predates the `ultralytics` package: models are built from `cfg/*.cfg` files rather than YAMLs, and nothing here is importable as a library — every entry point is a script run from the repo root.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
pip3 install -U -r requirements.txt        # install (numpy, scipy, opencv-python, torch, matplotlib, tqdm, h5py)
python train.py                            # train; edit the train_path values near the top of train.py first
python train.py -resume 1                  # resume from weights/latest.pt
(cd weights && bash download_weights.sh)   # fetch pretrained xView weights
python detect.py -image_folder path/to/train_images
ruff format . && ruff check --fix .        # formatting is applied on PRs by Ultralytics Actions, not a repo config
```

- There is no test suite, no `pyproject.toml`, and no packaging. CI is limited to `format.yml` (Ultralytics Actions), `cla.yml`, and `stale.yml` — no build or test workflow runs on PRs.
- Both scripts use single-dash argparse flags (`-cfg`, `-img_size`, `-batch_size`), not the `--flag` convention of later Ultralytics repos.

## Architecture

- `models.py` — `parse_model_config()` reads a Darknet `cfg/*.cfg` file into module definitions, `create_modules()` builds them, and `Darknet` runs the forward pass with `YOLOLayer` heads. `cfg/c60_a30symmetric.cfg` (60 classes, 30 k-means anchors) is the default for both `train.py` and `detect.py`.
- `utils/datasets.py` — `ListDataset` iterates full-resolution `.tif` images, samples one augmented chip per image, and reads label boxes from the `utils/targets_c60.mat` MATLAB file; `ImageFolder` serves inference. Images are sampled with per-image weights, and classes with `xview_class_weights()`.
- `utils/utils.py` — the shared helper module: xView class index remapping, anchor/target building, NMS, mAP, and plotting.
- `utils/analysis.m` — the MATLAB script that generated the k-means anchors baked into the `cfg` files; it is not part of the Python pipeline.
- `scoring/` — the official xView Challenge scorer, vendored from the Defense Innovation Unit under Apache-2.0. Keep its license header intact and prefer not to modify it.

## Conventions

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Google-style docstrings; the Actions bot runs Ruff, docformatter, prettier (YAML/JSON/Markdown), and codespell on PRs and its output can differ from a bare local run — expect bot commits on the branch and `git pull --rebase` before pushing again.
- No version string and no release process: weights are hosted on Google Cloud Storage and fetched by `weights/download_weights.sh`, so there is nothing to publish or bump.
