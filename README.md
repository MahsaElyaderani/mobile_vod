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

| Family | Small / big / joint experiments |
| --- | --- |
| `mobilenetv2` | `small_050`, `big_120d`, `joint_120d` |
| `efficientnet` | `small_tinynet_d`, `big_efficientnet_b0`, `joint_efficientnet_b0_tinynet_d` |
| `edgenext` | `small_edgenext_xx`, `big_edgenext_small`, `joint_edgenext` |

| Stage | Epochs | Batch size | Initial LR | Schedule |
| --- | ---: | ---: | ---: | --- |
| Classification | 5 | 8 images | 0.01 | Linear |
| Detection | 100 | 4 clips | 0.001 | Cosine |

Use `MATCHED_TRAIN_SUBSET = PROJECT_ROOT / "splits/train_25pct.json"` in `path.py`
for the shared 25% subset. Classification repeats each image for three recurrent
steps; detection uses six-frame clips. Validation runs every 5,000 updates on up
to 100 videos with 20 frames each (subset seed 7, training seed 0). Final evaluation
uses the same validation selection, big-only mode for standalone models and
interval 8 for joint models. Adjust batch sizes to GPU memory before starting.

New runs use separate family folders:
`RUNS_DIR/shared_<family>_subset25_class5_linear_det100_cosine/seed_0/<experiment>/`.
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
The older family launchers are retained locally but excluded from Git.

`launchers/run_detection_cosine_100epochs.sh` remains a separate MobileNetV2
workflow using previous best classification checkpoints. Each invocation creates
a fresh detection output directory; it does not automatically resume or run final evaluation.

MobileNetV1 is no longer supported by the active code. Its checkpoints require
the archived version; existing MobileNetV2 and EfficientNet checkpoint formats
are preserved. Local archives, datasets, and checkpoint weights are not published.

Local `docs/`, `results/`, `runs/`, `scripts/`, and `tests/` folders and monitoring
utilities are excluded from Git. Only training/evaluation Bash launchers are published.
