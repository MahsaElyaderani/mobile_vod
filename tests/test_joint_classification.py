"""Architecture matrix, multi-label targets, recurrent gradients, and transfer safety."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ModelConfig, model_config_dict
from paired_config import PairedConfig
from models.lstm_ssd import LSTMSSD
from models.paired_vod import PairedVOD
from models.joint_classification import JointClassifier
from datasets.paired_frames import PairedFrames
from datasets.classification import VIDClassificationDataset, ClassificationTransform
from utils.joint_initialization import make_joint_config, classifier_from_payload
from train import resolve_model_config


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_width_pairs_and_classification_transfer(self):
        for family, widths in [('timm_mobilenet_v2', (.5, 1.)), ('efficientnet_b0', (1.,))]:
            for big in widths:
                for small in widths:
                    with self.subTest(family=family, big=big, small=small):
                        cfg = make_joint_config(family, big)
                        paired = PairedConfig(fast_width_mult=small)
                        model = JointClassifier(cfg, paired, 30).eval()
                        x = torch.rand(1, 3, 320, 320)*2-1
                        if cfg.uses_timm:
                            x = PairedFrames(x, torch.rand(1, 3, 160, 160)*2-1)
                        logits = model(x, use_big=torch.tensor([[True, False, True]]))
                        self.assertEqual(logits.shape, (1, 3, 30))
                        target = torch.zeros_like(logits); target[:, :, [0, 8]] = 1
                        torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
                        for module in (model.big_backbone, model.small_backbone, model.lstm, model.classifier):
                            grads = [p.grad for p in module.parameters() if p.grad is not None]
                            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
                        self.assertFalse(any('predictor' in k or 'feature_maps' in k for k in model.state_dict()))
                        detector = PairedVOD(LSTMSSD(cfg), paired).eval()
                        head = {k: v.clone() for k, v in detector.big.predictor.state_dict().items()}
                        model.transfer_to_detector(detector)
                        for k, v in model.lstm.state_dict().items():
                            self.assertTrue(torch.equal(v, detector.big.lstm.state_dict()[k]))
                        self.assertTrue(all(torch.equal(v, detector.big.predictor.state_dict()[k]) for k, v in head.items()))
                        with torch.no_grad():
                            output, _ = detector.forward_selected(x, None, torch.tensor([False]))
                        self.assertTrue(torch.isfinite(output['class_logits']).all())
                        if cfg.uses_timm:
                            self.assertIsInstance(detector.small_projection, torch.nn.Identity)

    def test_standalone_sizes_and_legacy_resolution(self):
        for family, widths in [('timm_mobilenet_v2', (.5, 1.)), ('efficientnet_b0', (1.,)), ('tinynet_d', (1.,))]:
            for width in widths:
                for role, size in [('small', 160), ('big', 320)]:
                    cfg, _ = resolve_model_config(width, backbone=family, model_size=role)
                    self.assertEqual(cfg.image_size, size)
                    model = LSTMSSD(cfg).eval()
                    with torch.no_grad():
                        output, state = model.forward_step(torch.zeros(1, 3, size, size))
                    self.assertEqual(state[0].shape[-2:], (size//32, size//32))
                    self.assertTrue(torch.isfinite(output['class_logits']).all())
        self.assertEqual(resolve_model_config(.5)[0].image_size, 160)
        for family, width in (('timm_mobilenet_v2', 1.), ('timm_mobilenet_v2', .5)):
            for size in (160, 320):
                cfg, _ = resolve_model_config(width, image_size=size, backbone=family)
                self.assertEqual(cfg.image_size, size)
        with self.assertRaises(ValueError):
            resolve_model_config(1., image_size=320, model_size='small')
        from config import SSDConfig
        legacy = {'model_config': model_config_dict(ModelConfig(width_mult=.5)),
                  'ssd_config': asdict(SSDConfig())}
        self.assertEqual(resolve_model_config(None, checkpoint=legacy)[0].image_size, 320)
        with self.assertRaises(ValueError):
            resolve_model_config(None, checkpoint=legacy, model_size='small')

    def test_multi_label_and_full_frame_preprocessing(self):
        class Base:
            clip_length, num_classes = 1, 30
            def __len__(self): return 1
            def __getitem__(self, index):
                return self.transform([Image.new('RGB', (44, 32), (255, 0, 0))],
                                      [{'labels': torch.tensor([1, 9, 9])}])
        dataset = VIDClassificationDataset(Base(), 'timm_mobilenet_v2', training=False)
        frame, labels = dataset[0]
        self.assertEqual(labels.sum(), 2)
        self.assertEqual(labels.nonzero().flatten().tolist(), [0, 8])
        self.assertEqual(frame.big.shape, (3, 320, 320))
        self.assertEqual(frame.small.shape, (3, 160, 160))
        self.assertTrue(torch.equal(frame.big[:, 0, 0], torch.tensor([1., -1., -1.])))

    def test_classification_checkpoint_format_mapping_and_shapes(self):
        cfg = make_joint_config('timm_mobilenet_v2')
        model = JointClassifier(cfg, PairedConfig(), 30)
        payload = {'classification_format_version': 1, 'stage': 'joint_pretrain', 'objective': 'vid_multilabel_bce',
                   'class_to_idx': {str(k): k-1 for k in range(1,31)}, 'model_config': model_config_dict(cfg),
                   'paired_config': asdict(model.paired_config), 'model': model.state_dict()}
        restored = classifier_from_payload(payload)
        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()))
        with self.assertRaises(ValueError): classifier_from_payload({**payload, 'stage': 'stage2'})
        with self.assertRaises(ValueError): classifier_from_payload({**payload, 'class_to_idx': {'x': 0}})
        with self.assertRaises(ValueError):
            PairedVOD(LSTMSSD(cfg), PairedConfig(fast_width_mult=.35))
        with self.assertRaises(ValueError):
            model.transfer_to_detector(PairedVOD(LSTMSSD(make_joint_config('timm_mobilenet_v2', .5))))


class MatchedTests(unittest.TestCase):
    def test_reused_features_match_repeated_forward_and_update(self):
        from copy import deepcopy
        from dataclasses import replace
        from train_matched import build_model
        torch.set_num_threads(2)
        torch.manual_seed(23)
        frames = PairedFrames(torch.rand(1, 3, 320, 320)*2-1,
                              torch.rand(1, 3, 160, 160)*2-1)
        for experiment in ('small_050', 'big_120d', 'joint_120d',
                           'small_tinynet_d', 'big_efficientnet_b0', 'joint_efficientnet_b0_tinynet_d'):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, 'classification', 7).train()
                reference = deepcopy(model)
                # Disable reuse only: retain the already-frozen BN module states.
                # This executes the original three-backbone-forward algorithm.
                reference.paired_config = replace(reference.paired_config, freeze_bn_stats=False)
                before = {k: v.clone() for k, v in model.named_buffers()}
                outputs, gradients, states = [], [], []
                for current, calls in ((reference, 3), (model, 1)):
                    memory = []
                    hook = current.lstm.register_forward_hook(
                        lambda module, args, result: memory.append(tuple(v.detach().clone() for v in result[1])))
                    optimizer = torch.optim.RMSprop(current.parameters(), lr=.01, alpha=.9, momentum=.9, eps=1.)
                    torch.manual_seed(42)
                    with patch.object(current.big_backbone, 'forward', wraps=current.big_backbone.forward) as big, \
                         patch.object(current.small_backbone, 'forward', wraps=current.small_backbone.forward) as small:
                        logits = current(frames, steps=3, use_big=torch.tensor([[True, False, True]]))
                    hook.remove()
                    self.assertEqual(big.call_count, calls)
                    self.assertEqual(small.call_count, 0 if current.standalone else calls)
                    target = torch.zeros_like(logits); target[:, :, [0, 8]] = 1
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
                    gradients.append({k: v.grad.clone() for k, v in current.named_parameters()})
                    optimizer.step()
                    states.append(optimizer.state_dict())
                    outputs.append(logits.detach())
                    if current is reference:
                        expected_memory, rng = memory, torch.get_rng_state()
                    else:
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                # Sharing a graph changes float32 gradient accumulation order.
                torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-4, atol=1e-7)
                torch.testing.assert_close(states[0], states[1], rtol=1e-4, atol=1e-7)
                torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=1e-5, atol=1e-8)
                self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.named_buffers()))

    def test_reuse_guard_preserves_other_configurations(self):
        torch.set_num_threads(2)
        for family, frozen in (('timm_mobilenet_v2', False), ('efficientnet_b0', False)):
            with self.subTest(family=family):
                model = JointClassifier(make_joint_config(family),
                    PairedConfig(fast_width_mult=1. if family == 'efficientnet_b0' else .5,
                                 freeze_bn_stats=frozen), 30).train()
                frames = torch.zeros(1, 3, 320, 320)
                frames = PairedFrames(frames, torch.zeros(1, 3, 160, 160))
                bn = next(m for m in model.big_backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
                before = int(bn.num_batches_tracked)
                with torch.no_grad(), patch.object(model.big_backbone, 'forward', wraps=model.big_backbone.forward) as big:
                    model(frames, steps=3)
                self.assertEqual(big.call_count, 3)
                self.assertEqual(int(bn.num_batches_tracked) - before, 0 if frozen else 3)

    def test_micro_ap_and_coverage(self):
        from engine.evaluator import classification_metrics, compute_map50
        scores = torch.tensor([[.9, .8, .8], [.1, .2, .3]])
        targets = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        metrics = classification_metrics(scores, targets)
        self.assertAlmostEqual(metrics['micro_ap'], (1 + 2/3 + 3/5) / 3)
        self.assertEqual(metrics['present_classes'], [1, 2, 3])
        self.assertEqual(classification_metrics(scores, targets * 0)['micro_ap'], 0.)
        self.assertEqual(classification_metrics(scores * 0, targets)['micro_ap'], .5)
        empty = compute_map50([], [], num_classes=3)
        self.assertEqual(empty['absent_classes'], [1, 2, 3])
        self.assertEqual(empty['mAP@0.5'], 0.)
        with self.assertRaises(ValueError):
            classification_metrics(scores * float('nan'), targets)

    def test_latest_and_best_saving(self):
        import tempfile
        from train_matched import save_matched_checkpoints
        from utils.runtime import load_checkpoint
        for stage, key, scores in [('classification', 'bce_loss', [.4, .3, .5, .3]),
                                   ('classification', 'micro_ap', [.4, .5, .3, .5]),
                                   ('detection', 'mAP@0.5', [.4, .5, .3, .5])]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / (stage + '.pt')
                history = []
                for epoch, score in enumerate(scores, 1):
                    history.append({'epoch': epoch, 'validation': {key: score}})
                    payload = {'stage': 'matched_' + stage, 'epoch': epoch, 'history': history, 'selection_metric': key}
                    save_matched_checkpoints(payload, destination)
                    self.assertEqual(load_checkpoint(destination)['epoch'], epoch)
                    self.assertEqual(load_checkpoint(destination.with_stem(stage + '_best'))['epoch'], min(epoch, 2))
                    history = load_checkpoint(destination)['history']  # Resume retains the best score.
                self.assertEqual(len(list(Path(directory).glob('*.pt'))), 2)
                history.append({'epoch': 5, 'validation': {key: float('nan')}})
                with self.assertRaises(ValueError):
                    save_matched_checkpoints({**payload, 'history': history}, destination)
                self.assertEqual(load_checkpoint(destination)['epoch'], 4)

    def test_seven_models_shared_initialization_gradients_and_memory(self):
        from train_matched import EXPERIMENTS, build_model, seed_training
        from utils.checkpoints import capture_rng
        torch.set_num_threads(2)
        frames = PairedFrames(torch.rand(1, 3, 320, 320), torch.rand(1, 3, 160, 160))
        shared, heads, rng = None, None, None
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                classifier = build_model(experiment, 'classification', 7).train()
                weights = {k: v for k, v in classifier.state_dict().items()
                           if k.startswith(('lstm.', 'classifier.'))}
                if shared is None:
                    shared = {k: v.clone() for k, v in weights.items()}
                self.assertTrue(all(torch.equal(v, shared[k]) for k, v in weights.items()))
                before_bn = {k: v.clone() for k, v in classifier.named_buffers()}
                seed_training(10)
                logits = classifier(frames, steps=2)
                state = capture_rng()
                if rng is None:
                    rng = state
                self.assertTrue(torch.equal(state['torch'], rng['torch']))
                self.assertEqual(state['python'], rng['python'])
                logits.square().mean().backward()
                self.assertTrue(all(p.requires_grad for p in classifier.parameters()))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in classifier.parameters()))
                self.assertTrue(all(torch.equal(v, before_bn[k]) for k, v in classifier.named_buffers()))
                detector = build_model(experiment, 'detection', 7).train()
                current = detector.big.predictor.state_dict()
                if heads is None:
                    heads = {k: v.clone() for k, v in current.items()}
                self.assertTrue(all(torch.equal(v, heads[k]) for k, v in current.items()))
                classifier.transfer_to_detector(detector)
                self.assertTrue(all(torch.equal(v, detector.big.lstm.state_dict()[k]) for k, v in classifier.lstm.state_dict().items()))
                previous = detector.initial_state(frames)
                output, memory = detector.forward_selected(frames, previous, torch.tensor([False]))
                self.assertEqual(memory[0].shape, (1, 256, 10, 10))
                if detector.standalone:
                    self.assertIsInstance(detector.fast_backbone, torch.nn.Identity)
                    direct, expected = detector.forward_big(frames, previous)
                    torch.testing.assert_close(output['class_logits'], direct['class_logits'], rtol=0, atol=0)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(memory, expected)))
                else:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(memory, previous)))
                output['class_logits'].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in detector.big.backbone.parameters()))

    def test_matched_training_transfer_resume_and_evaluation(self):
        import contextlib
        import io
        import json
        import os
        import tempfile
        import shutil
        from train_matched import EXPERIMENTS, parse_args, train, model_from_matched
        from models.timm_mobilenet_v2 import TimmMobileNetV2Backbone
        from utils.runtime import load_checkpoint
        from train import _same_value
        from evaluate.evaluate_paired_vod import main as evaluate
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                folder = root / split
                folder.mkdir()
                images, annotations = [], []
                for index in range(4):
                    Image.new('RGB', (64, 64), (40 + index*20, 90, 180)).save(folder / f'{index}.png')
                    images.append(dict(id=index, video_id=1, frame_id=index, width=64, height=64, file_name=f'{split}/{index}.png'))
                    annotations.append(dict(id=index, image_id=index, category_id=1, bbox=[8, 8, 40, 40]))
                (root / f'{split}.json').write_text(json.dumps(dict(videos=[{'id': 1}], images=images,
                    annotations=annotations, categories=[{'id': i, 'name': str(i)} for i in range(1, 31)])))
            common = ['--annotations', str(root/'train.json'), '--frames-root', str(root),
                      '--val-annotations', str(root/'val.json'), '--val-frames-root', str(root),
                      '--epochs', '1', '--batch-size', '1', '--max-batches', '1', '--num-workers', '0',
                      '--clip-length', '2', '--clip-stride', '2', '--steps', '2',
                      '--val-frames-per-video', '2', '--device', 'cpu', '--no-progress']
            weights = (contextlib.nullcontext() if os.environ.get('TIMM_VERIFY_PRETRAINED') == '1' else
                       patch.object(TimmMobileNetV2Backbone, 'load_pretrained', return_value={'test': 'random'}))
            with contextlib.redirect_stdout(io.StringIO()), weights:
                for experiment in EXPERIMENTS:
                    with self.subTest(experiment=experiment):
                        classification = root / experiment / 'classification.pt'
                        detection = root / experiment / 'detection.pt'
                        args = parse_args(common + ['--experiment', experiment, '--task', 'classification', '--checkpoint-out', str(classification)])
                        trained = train(args)
                        self.assertFalse(list(classification.parent.glob('*_epoch_*.pt')))
                        snapshot = classification.with_name('classification_snapshot.pt')
                        shutil.copyfile(classification, snapshot)
                        self.assertEqual(trained['history'][0]['validation']['samples'], 2)
                        restored = model_from_matched(load_checkpoint(classification))
                        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in trained['model'].items()))
                        detection_args = parse_args(common + ['--experiment', experiment, '--task', 'detection',
                            '--init-checkpoint', str(snapshot), '--checkpoint-out', str(detection)])
                        detected = train(detection_args)
                        self.assertFalse(list(detection.parent.glob('*_epoch_*.pt')))
                        self.assertEqual(detected['history'][0]['validation']['num_frames'], 2)
                        with self.assertRaises(FileExistsError):
                            train(detection_args)
                        mode = 'big' if not experiment.startswith('joint') else 'fixed'
                        results = evaluate(['--checkpoint', str(detection), '--annotations', str(root/'val.json'),
                            '--frames-root', str(root), '--frames-per-video', '2', '--mode', mode, '--device', 'cpu',
                            '--warmup-frames', '0', '--no-visuals', '--no-progress', '--output-dir', str(root/experiment/'evaluation')])
                        self.assertEqual(next(iter(results.values()))['num_frames'], 2)
                        if experiment in ('small_050', 'joint_100'):
                            for original_args, checkpoint in ((args, classification), (detection_args, detection)):
                                resumed_args = parse_args(common + ['--experiment', experiment, '--task', original_args.task,
                                    '--resume', str(checkpoint), '--checkpoint-out', str(checkpoint), '--epochs', '2'])
                                if experiment == 'small_050':
                                    from utils.runtime import atomic_save_checkpoint
                                    best_path = checkpoint.with_stem(checkpoint.stem + '_best')
                                    saved_best = load_checkpoint(best_path)
                                    different_weights = {k: v + 1 for k, v in saved_best['model'].items()}
                                    for changes in ({'run_id': 'unrelated'}, {'epoch': 2}, {'model': different_weights}):
                                        atomic_save_checkpoint({**saved_best, **changes}, best_path)
                                        with self.assertRaises(FileExistsError):
                                            train(resumed_args)
                                    atomic_save_checkpoint(saved_best, best_path)
                                resumed = train(resumed_args)
                                original_args.epochs = 2
                                original_args.checkpoint_out = str(root/experiment/(original_args.task + '_full.pt'))
                                full = train(original_args)
                                self.assertTrue(all(torch.equal(v, full['model'][k]) for k, v in resumed['model'].items()), original_args.task)
                                self.assertTrue(_same_value(resumed['optimizer'], full['optimizer']))
                                with self.assertRaises(ValueError):
                                    model_from_matched({**resumed, 'class_to_idx': {}})
                                with self.assertRaises(ValueError):
                                    model_from_matched({**resumed, 'stage': 'stage2'})


if __name__ == '__main__':
    unittest.main()
