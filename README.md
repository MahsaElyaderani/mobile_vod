# Mobile VOD

PyTorch video object detection inspired by *Looking Fast and Slow: Memory-Guided
Mobile Video Object Detection*. The project combines pretrained image backbones,
recurrent memory, and SSD detection heads for the 30 ImageNet VID classes.
It supports timm MobileNetV2, EfficientNet-B0/TinyNet-D, and EdgeNeXt, with standalone
and joint models. All three families use a shared ConvLSTM and SSD detection heads in
joint mode. Backbone weights are ImageNet-pretrained; SSD heads are trained here.

| Model family | Small backbone (160×160) | Big backbone (320×320) |
| --- | --- | --- |
| MobileNetV2 | `mobilenetv2_050.lamb_in1k` | `mobilenetv2_100.ra_in1k`, `mobilenetv2_110d.ra_in1k`, or `mobilenetv2_120d.ra_in1k` |
| EfficientNet / TinyNet | `tinynet_d.in1k` | `efficientnet_b0.ra_in1k` |
| EdgeNeXt | `edgenext_xx_small.in1k` | `edgenext_small.usi_in1k` |

Backbones are implemented in `models/backbones.py`. EdgeNeXt uses a 304-channel
LSTM input and a small-branch 168→304 projection; other families retain 1280 input
channels. All use a 256-channel LSTM output and the same SSD head architecture.

## Setup

Create a separate environment (the launcher expects a Conda environment named `vision`):

```bash
git clone https://github.com/MahsaElyaderani/mobile_vod.git
cd mobile_vod
conda create -n vision python=3.11 -y
conda activate vision
python -m pip install -r requirement/requirements.txt \
  -r requirement/requirements_paired.txt -r requirement/requirements_timm.txt
```

The requirements pin PyTorch 2.6.0, TorchVision 0.21.0, and timm 1.0.22.
The default launcher requires CUDA; check that `torch.cuda.is_available()` is true.

Edit `path.py` for your machine, especially `DATASET_ROOT`. Training expects:

```text
ILSVRC2015/
  Annotations/imagenet_vid_train.json
  Annotations/imagenet_vid_val.json
  Data/VID/...
```

Annotations must be video-aware COCO-style JSON with `videos`, `images`,
`annotations`, and `categories`; raw ImageNet XML files are not loaded directly.
The included `splits/train_25pct.json` must match your training annotation file
(the loader checks its hash). Dataset files and trained checkpoints are not included.
Initial backbone training downloads pretrained weights through timm.

## Train and evaluate

Choose one family; each command trains its small standalone, big standalone,
and joint experiments, then evaluates their best detection checkpoints:

```bash
bash launchers/run_train_evaluate.sh mobilenetv2
bash launchers/run_train_evaluate.sh efficientnet
bash launchers/run_train_evaluate.sh edgenext
```

The optional second argument is `all` (default), `train`, or `eval`:

```bash
bash launchers/run_train_evaluate.sh edgenext train
bash launchers/run_train_evaluate.sh edgenext eval
```

Evaluation-only mode requires all three best detection checkpoints and never
runs training. Each evaluation creates a new results directory.

The launcher defaults to `TRAIN_SUBSET=0.25`. Set `0.5` for the fixed 50%
manifest; the run name is automatic:

```bash
TRAIN_SUBSET=0.5 \
  bash launchers/run_train_evaluate.sh edgenext all
# Later, evaluate the same run without training:
TRAIN_SUBSET=0.5 \
  bash launchers/run_train_evaluate.sh edgenext eval
```

Explicit JSON manifest paths are also accepted through `TRAIN_SUBSET`. These
fractions select the saved video subsets; they do not resample the dataset.

Optional `MAX_VAL_VIDEOS`, `VAL_FRAMES_PER_VIDEO`, and `SUBSET_SEED` environment
variables override 100, 20, and 7 for both training validation and final evaluation.
Use the same values across families for comparisons. When evaluating later, supply
any validation overrides again. Default run names include the family, subset filename,
and epoch budgets; nondefault validation settings add a suffix. Existing default
25% run names are preserved. Use `RUN_NAME` only to select an explicit folder or
start an independent repeat. Different manifests with the same filename need
separate `RUN_NAME` values; the trainer still rejects incompatible resumes. To evaluate
an existing run with different validation coverage, set `RUN_NAME` to that run explicitly. Continuing an existing
25% checkpoint on 50% with its optimizer/schedule is the separate
`launchers/run_joint_subset50.sh` workflow.

| Family | Small / big / joint experiments |
| --- | --- |
| `mobilenetv2` | `small_050`, `big_120d`, `joint_120d` |
| `efficientnet` | `small_tinynet_d`, `big_efficientnet_b0`, `joint_efficientnet_b0_tinynet_d` |
| `edgenext` | `small_edgenext_xx`, `big_edgenext_small`, `joint_edgenext` |

| Stage | Epochs | Batch size | Initial LR | Schedule |
| --- | ---: | ---: | ---: | --- |
| Classification | 5 | 8 images | 0.01 | Linear |
| Detection | 100 | 4 clips | 0.001 | Cosine |

This launcher selects `splits/train_25pct.json` by default, independently of
`MATCHED_TRAIN_SUBSET` in `path.py`. Classification repeats each image for three recurrent
steps; detection uses six-frame clips. Validation runs every 5,000 updates on up
to 100 videos with 20 frames each (subset seed 7, training seed 0). Final evaluation
uses the same validation selection, big-only mode for standalone models and
interval 8 for joint models. Adjust batch sizes to GPU memory before starting.

New runs use separate family folders:
`RUNS_DIR/shared_<family>_subset25_class5_linear_det100_cosine/seed_0/<experiment>/`
for 25%, or the equivalent `subset50` folder for 50%.
Previous experiment folders are not reused. Repeating the same command resumes
latest `classification.pt` and `detection.pt`, including mid-epoch progress.
Keep the saved training budget and settings unchanged when resuming.

Fresh detection starts from `classification_best.pt`. Best checkpoints use
highest classification micro AP or detection mAP@0.5; final evaluation loads
`detection_best.pt`. Evaluation JSON files go to a timestamped family folder under
`RESULTS_DIR`. All paths are configured in `path.py`.

For individual commands and options:

```bash
python train_matched.py --help
python -m evaluate.evaluate_paired_vod --help
```

Existing standalone detection stages remain available through `train.py`;
`train_joint_pretrain.py` and `train_paired_vod.py` provide separate joint entry points.
The shared launcher replaces the older family launchers.

MobileNetV1 is no longer supported by the active code. Its checkpoints require
the archived version; existing MobileNetV2 and EfficientNet checkpoint formats
are preserved. Local archives, datasets, and checkpoint weights are not published.

Local `docs/`, `results/`, `runs/`, `scripts/`, and `tests/` folders and monitoring
utilities are excluded from Git. Only training/evaluation Bash launchers are published.
