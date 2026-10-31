# Mobile VOD

PyTorch video object detection inspired by *Looking Fast and Slow: Memory-Guided
Mobile Video Object Detection*. The project combines MobileNet backbones,
recurrent memory, and SSD detection heads for the 30 ImageNet VID classes.
It supports MobileNetV1 and timm MobileNetV2, with standalone and joint models.

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
Initial MobileNetV2 training downloads pretrained weights through timm.

## Train and evaluate

Review the settings at the top of `run_matched.sh`, then run:

```bash
bash run_matched.sh
```

The current launcher runs `small_050`, `big_120d`, and `joint_120d` on the same
25% training subset. Small uses `mobilenetv2_050.lamb_in1k` at 160×160; big uses
`mobilenetv2_120d.ra_in1k` at 320×320. Joint combines both with shared memory
and detection heads.

Each experiment performs multi-label classification pretraining, then detection
training initialized from its best classification checkpoint, then detection
evaluation. Current defaults are **1 epoch per stage**, batch sizes **16 images / 8
clips**, linear learning-rate decay, and validation every 5,000 updates on up to
100 videos with 20 frames each. Classification repeats each image for three
recurrent steps; detection uses six-frame clips. Adjust batch sizes to GPU memory
before starting, and use a new output root for a different training budget.

Latest checkpoints (`classification.pt`, `detection.pt`) resume automatically,
including mid-epoch progress. Best checkpoints (`*_best.pt`) use highest
classification micro AP or detection mAP@0.5. Linear-budget resumes require the
saved budget to match. The launcher evaluates the best detector checkpoints.

Checkpoints go under `RUNS_DIR/matched_subset25_linear_1epoch/seed_0/`; evaluation
JSON files go under a timestamped directory in `RESULTS_DIR`. Paths are configured
in `path.py`. For individual commands and options, use:

```bash
python train_matched.py --help
python -m evaluate.evaluate_paired_vod --help
```

Existing standalone detection stages remain available through `train.py`;
`train_joint_pretrain.py` and `train_paired_vod.py` provide separate joint entry points.
