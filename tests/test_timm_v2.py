"""Offline regressions plus opt-in real checkpoints: TIMM_VERIFY_PRETRAINED=1."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ModelConfig, SSDConfig, model_config_dict
from datasets.transforms import EvalTransform, TrainTransform, _resize_and_normalize_image
from datasets.transforms import resize_detection_views
from datasets.paired_frames import PairedFrames
from datasets.dataloader import video_collate_fn
from models.timm_mobilenet_v2 import TimmMobileNetV2Backbone
from models.lstm_ssd import LSTMSSD
from models.ssd import MobileNetSSD
from models.paired_vod import PairedVOD
from paired_config import PairedConfig
from utils.checkpoints import transfer_stage1, model_from_checkpoint, FORMAT_VERSION
from utils.paired_checkpoints import paired_metadata, paired_from_payload, detector_fingerprint, initialize_pair
from train_paired_vod import parse_args


def config():
    return ModelConfig(backbone='timm_mobilenet_v2', mobilenet_output_channels=1280)


class TimmV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_active_defaults_and_archived_v1_rejection(self):
        from config import checkpoint_model_config
        self.assertEqual(model_config_dict(ModelConfig())['backbone'], 'timm_mobilenet_v2')
        self.assertEqual(ModelConfig().input_interpolation, 'bicubic')
        self.assertEqual(ModelConfig().mobilenet_output_channels, 1280)
        with self.assertRaises(ValueError):
            ModelConfig(backbone='mobilenet_v1')
        for values in ({}, {'backbone': 'mobilenet_v1'}):
            with self.assertRaisesRegex(ValueError, 'archived'):
                checkpoint_model_config(values)
        with self.assertRaises(ValueError):
            ModelConfig(mobilenet_output_channels=1024)
        with self.assertRaises(ValueError):
            PairedVOD(LSTMSSD(config()), PairedConfig(fast_width_mult=.35))

    def test_rgb_scaling_normalization_and_full_frame_geometry(self):
        rng = np.random.default_rng(4)
        image = Image.fromarray(rng.integers(0, 256, (197, 281, 3), dtype=np.uint8))
        target = {'boxes': torch.tensor([[0., 0., 281., 197.]]), 'labels': torch.tensor([1])}
        for width, size, name in [(1., 320, None), (.5, 160, None),
                                  (1., 320, 'efficientnet_b0.ra_in1k'), (1., 160, 'tinynet_d.in1k')]:
            backbone = TimmMobileNetV2Backbone(width, model_name=name)
            frames, targets = EvalTransform(size, interpolation='bicubic')([image], [target])
            reference = TF.to_tensor(TF.resize(image, [size, size], TF.InterpolationMode.BICUBIC, antialias=True))
            reference = TF.normalize(reference, backbone.data_config['mean'], backbone.data_config['std'])
            torch.testing.assert_close(backbone.normalize(frames)[0], reference, rtol=0, atol=6e-7)
            torch.testing.assert_close(targets[0]['boxes'], torch.tensor([[0., 0., 1., 1.]]))
            train, _ = TrainTransform(size, flip_prob=0, min_iou_options=(None,), interpolation='bicubic')([image], [target])
            self.assertTrue(torch.equal(train, frames))
            # Distinct RGB values detect swapped channels and double normalization.
            rgb = Image.new('RGB', (9, 7), (255, 128, 0))
            x = _resize_and_normalize_image(rgb, size, 'bicubic')[None]
            expected = (torch.tensor([1., 128/255., 0.]) - backbone.input_mean.flatten()) / backbone.input_std.flatten()
            torch.testing.assert_close(backbone.normalize(x)[0, :, 0, 0], expected)

    def test_shared_lstm_gradients_and_offline_checkpoint_roundtrip(self):
        pair = PairedVOD(LSTMSSD(config())).eval()
        self.assertIsInstance(pair.small_projection, torch.nn.Identity)
        source = Image.fromarray(np.random.default_rng(9).integers(0, 256, (241, 389, 3), dtype=np.uint8))
        x = resize_detection_views(source, 320, 'bicubic', paired=True)[None]
        for module, frame in [(pair.big.backbone, x.big), (pair.fast_backbone, x.small)]:
            self.assertEqual(module(frame).shape, (1, 1280, 10, 10))
        with self.assertRaisesRegex(ValueError, 'PairedFrames'):
            pair.forward_small(x.big)
        state = None
        for forward in [pair.forward_big, pair.forward_small]:
            pair.zero_grad(set_to_none=True)
            out, state = forward(x, state)
            (out['class_logits'].square().mean() + out['box_regression'].square().mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in pair.parameters() if p.grad is not None))
            self.assertIsNotNone(pair.big.lstm.parameters().__next__().grad)
            state = tuple(t.detach() for t in state)
        payload = {**paired_metadata(pair), 'stage': 'detector', 'model': pair.state_dict()}
        import timm
        original = timm.create_model
        def offline(*args, **kwargs):
            self.assertFalse(kwargs.get('pretrained', False))
            return original(*args, **kwargs)
        with patch('timm.create_model', side_effect=offline):
            restored = paired_from_payload(payload, 'detector')
        self.assertEqual(detector_fingerprint(pair), detector_fingerprint(restored))

    def test_independent_resizing_shared_augmentation_and_batching(self):
        source = Image.fromarray(np.random.default_rng(19).integers(0, 256, (197, 281, 3), dtype=np.uint8))
        target = {'boxes': torch.tensor([[0., 0., 281., 197.]]), 'labels': torch.tensor([1])}
        frames, targets = EvalTransform(320, 'bicubic', paired=True)([source], [target])
        self.assertTrue(torch.equal(frames.big[0], _resize_and_normalize_image(source, 320, 'bicubic')))
        self.assertTrue(torch.equal(frames.small[0], _resize_and_normalize_image(source, 160, 'bicubic')))
        two_hop = torch.nn.functional.interpolate(frames.big, (160, 160), mode='bicubic',
                                                   align_corners=False, antialias=True).clamp(-1, 1)
        self.assertFalse(torch.equal(frames.small, two_hop))
        # Crop selection is shared, then each resolution is resized from that same crop.
        transform = TrainTransform(320, flip_prob=1, interpolation='bicubic', paired=True)
        with patch.object(transform, '_sample_crop', return_value=(20/281, 10/197, 260/281, 180/197)):
            augmented, _ = transform([source], [target])
        flipped = TF.hflip(source.crop((20, 10, 260, 180)))
        self.assertTrue(torch.equal(augmented.big[0], _resize_and_normalize_image(flipped, 320, 'bicubic')))
        self.assertTrue(torch.equal(augmented.small[0], _resize_and_normalize_image(flipped, 160, 'bicubic')))
        batch, _ = video_collate_fn([(frames, targets), (augmented, targets)])
        self.assertEqual(batch.big.shape, (2, 1, 3, 320, 320))
        self.assertEqual(batch.small.shape, (2, 1, 3, 160, 160))
        self.assertTrue(torch.equal(batch.to('cpu')[:, 0].small[1], augmented.small[0]))

    def test_stage_transfer_and_cli_defaults(self):
        stage1 = MobileNetSSD(config())
        payload = {'format_version': FORMAT_VERSION, 'stage': 'stage1', 'model': stage1.state_dict(),
                   'model_config': model_config_dict(config()), 'ssd_config': vars(SSDConfig())}
        report = transfer_stage1(LSTMSSD(config()), payload)
        self.assertEqual(len(report['reinitialized_shape_changes']), 5)
        model_from_checkpoint(payload)
        with patch('utils.runtime.load_checkpoint', return_value=payload):
            args = parse_args(['--big-checkpoint=x', '--checkpoint-out=y'])
        self.assertEqual((args.fast_init, args.fast_width_mult), ('timm', .5))
        args = parse_args(['--resume=x', '--checkpoint-out=y'], resume_payload={
            'paired_config': vars(PairedConfig(fast_width_mult=.5)),
            'run_definition': {'arguments': {'fast_init': 'timm'}}})
        self.assertEqual((args.fast_init, args.fast_width_mult), ('timm', .5))


@unittest.skipUnless(os.environ.get('TIMM_VERIFY_PRETRAINED') == '1', 'Opt-in checkpoint download/verification')
class PretrainedTests(unittest.TestCase):
    def test_official_weights_and_independent_reference_for_each_branch(self):
        import timm
        from path import TIMM_CACHE, RESULTS_DIR
        from utils.compute import profile_conv_linear
        import json
        torch.set_num_threads(2)
        reports = []
        for width, size, stride, name in [(1., 320, 32, None), (.5, 160, 16, None),
                (1.1, 320, 32, None), (1.2, 320, 32, None),
                (1., 160, 16, 'tinynet_d.in1k'), (1., 320, 32, 'efficientnet_b0.ra_in1k')]:
            wrapper = TimmMobileNetV2Backbone(width, 32, model_name=name).eval()
            provenance = wrapper.load_pretrained()
            reference = timm.create_model(wrapper.model_name, pretrained=True, cache_dir=str(TIMM_CACHE)).eval()
            cfg = timm.data.resolve_model_data_config(reference)
            reference.reset_classifier(0)
            self.assertEqual(wrapper.data_config, cfg)
            for key, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, wrapper.network.state_dict()[key]), key)
            x = torch.rand(1, 3, size, size) * 2 - 1
            normalized = TF.normalize((x + 1) / 2, cfg['mean'], cfg['std'])
            with torch.no_grad():
                torch.testing.assert_close(wrapper(x), reference.forward_features(normalized), rtol=0, atol=0)
            adapted = TimmMobileNetV2Backbone(width, stride, model_name=name).eval()
            provenance = adapted.load_pretrained()
            if stride == 16:
                reference.blocks[5][0].conv_dw.stride = (1, 1)
            for key, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, adapted.network.state_dict()[key]), key)
            with torch.no_grad():
                output = adapted(x)
                torch.testing.assert_close(output, reference.forward_features(normalized), rtol=0, atol=0)
            self.assertEqual(tuple(output.shape), (1, 1280, 10, 10))
            adapted(x).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapted.parameters()))
            reports.append({**provenance, 'output_shape': list(output.shape),
                            'reference_max_abs_error': 0., 'all_feature_tensors_equal': True,
                            'finite_backward': True, 'parameters': sum(p.numel() for p in adapted.parameters()),
                            'conv_linear_macs': profile_conv_linear(adapted, lambda: adapted(x))})
        destination = RESULTS_DIR / 'timm_v2_verification'
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'pretrained_report.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
