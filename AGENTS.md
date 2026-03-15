# Repository Guidelines

## Project Structure & Module Organization
This repository contains two parallel RetinaNet implementations: `jittor-retinanet/` for the Jittor version and `pytorch-retinanet/` for the PyTorch reference. Core model code lives in each framework's `retinanet/` package. Training and evaluation entry points are top-level scripts such as `jittor-retinanet/train.py`, `jittor-retinanet/coco_validation.py`, `pytorch-retinanet/train.py`, and `pytorch-retinanet/coco_validation.py`. Utility scripts for dataset download and tiny-COCO generation live in `tools/`. Generated outputs are split by purpose: CSV logs in `logs/`, weights in `checkpoints/`, and COCO result JSON files in `results/`. Keep datasets outside git.

## Build, Test, and Development Commands
Install the documented Python dependencies first: `pip install jittor pandas pycocotools opencv-python requests`.

Run the main Jittor training flow from its subdirectory:
```bash
cd jittor-retinanet
python train.py --dataset coco --coco_path ./coco --depth 50 --epochs 5 --batch_size 2
```

Validate a saved checkpoint with:
```bash
python coco_validation.py --coco_path ./coco --model <checkpoint>
```

Download COCO 2017 data from the repository root with:
```bash
python tools/download_coco2017.py
```

Use the PyTorch commands from `pytorch-retinanet/` with the same argument pattern when comparing implementations.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, and small script-oriented modules. Preserve the current CLI style based on `argparse`. Keep framework-specific changes isolated to the corresponding subtree instead of mixing Jittor and PyTorch logic. No formatter or linter is configured here, so match nearby code closely and keep imports, argument names, and checkpoint file naming consistent with existing scripts.

## Testing Guidelines
There is no dedicated automated test suite yet. Treat training and validation scripts as the verification path. Before opening a PR, run at least one smoke test on a small dataset such as tiny COCO, confirm that logs are written to `logs/train_log.csv`, and verify that checkpoints and evaluation outputs are created in `checkpoints/` and `results/`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Update README.md` and `change the showing effect of photo`. Prefer concise present-tense messages under 72 characters, and scope each commit to one change. PRs should include: a short summary, affected framework (`jittor` or `pytorch`), dataset/setup used for validation, and screenshots or metric snippets when changing visualization or training behavior.

## Data & Configuration Tips
The scripts expect COCO-style directories like `coco/annotations/instances_train2017.json` and `coco/images/train2017/`. Avoid hard-coding machine-specific absolute paths; several helper scripts currently assume local paths and should be parameterized if you extend them.
