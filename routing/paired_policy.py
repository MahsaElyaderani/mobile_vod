"""Paired detector training schedules, evaluation decisions, and temporal context."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from paired_config import PolicyConfig
from datasets.paired_frames import big_frames


def frame_signature(frame):
    frame = big_frames(frame)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError('Expected normalized RGB [3,H,W].')
    return F.interpolate((frame.detach().cpu()[None] + 1) / 2, (16, 16), mode='area')[0]


def frame_change(signature, previous):
    return 0.0 if previous is None else float((signature - previous).abs().mean())


@dataclass
class VideoContext:
    previous_big: bool = False
    last_big_call: int | None = None
    last_big_contribution: int | None = None
    previous_signature: object = None

    def contribution_age(self, step):
        return None if self.last_big_contribution is None else step - self.last_big_contribution

    def observe_frame(self, frame):
        signature = frame_signature(frame)
        change = frame_change(signature, self.previous_signature)
        self.previous_signature = signature
        return change

    def scene_reset(self):
        self.previous_big = False
        self.last_big_contribution = None

    def record(self, used_big, step, contributed):
        self.previous_big = bool(used_big)
        if used_big:
            self.last_big_call = step
        if contributed:
            self.last_big_contribution = step


def forced_big(config: PolicyConfig, step, last_call, scene_change=False, precision=False):
    if step < 0 or (last_call is not None and last_call >= step):
        raise ValueError("Invalid video step or previous big-call index.")
    if config.mode == "small":
        if precision:
            raise ValueError("Small-only mode cannot honor big-model precision requests.")
        return []
    if config.mode == "big":
        return ["big_only"]
    reasons = []
    if step == 0:
        reasons.append("first_frame")
    if precision:
        reasons.append("precision_request")
    if scene_change:
        reasons.append("scene_change")
    if config.mode == "fixed":
        if step > 0 and step % config.interval == 0:
            reasons.append("fixed_interval")
    return reasons


def choose_big(config, step, last_call, scene_change=False, precision=False):
    reasons = forced_big(config, step, last_call, scene_change, precision)
    return bool(reasons), reasons


def random_training_schedule(batch, time, device):
    """Independent 50/50 choices, including the first frame; only big updates memory."""
    if min(batch, time) < 1:
        raise ValueError("Empty training schedule.")
    # Preserve RNG consumption of existing version-3 random-only checkpoints.
    # Previously one schedule-selection draw preceded each clip's branch draws.
    for _ in range(batch):
        torch.randint(1, ())
    big = torch.stack([torch.rand(time, device=device) < 0.5 for _ in range(batch)])
    return big, torch.zeros_like(big)


def validate_random_training_resume(payload):
    """Never silently change a saved experiment to random-only joint training."""
    definition = payload.get('run_definition', {})
    for arguments in (definition.get('arguments', {}), payload.get('resume_arguments', {})):
        if 'schedules' in arguments and arguments['schedules'] != ['random']:
            raise ValueError('This checkpoint used training schedules that have been removed. '
                             'Only random-only training checkpoints can resume; evaluation is unchanged.')
    if definition.get('random_schedule_version') != 3:
        raise ValueError('Checkpoint does not use the supported independent 50/50 random schedule '
                         '(version 3); refusing to change its training behavior on resume.')

    arguments = {**definition.get('arguments', {}), **payload.get('resume_arguments', {})}
    warmup_end = arguments.get('zero_memory_epochs', 0) + arguments.get('clean_memory_epochs', 0)
    if payload.get('epoch', 0) < warmup_end:
        raise ValueError('Checkpoint has unfinished small-model warmup, which has been removed. '
                         'Use prepare_joint_training.py to explicitly start a new joint-training run.')
