"""Read-only real-VID loader/reference checks; no model training or configuration edits."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from path import VAL_ANNOTATIONS, VAL_FRAMES, TIMM_CACHE
from datasets.imagenet_vid import ImageNetVIDDataset
from datasets.classification import VIDClassificationDataset, classification_collate
from datasets.dataloader import budget_loader, build_dataloader
from datasets.transforms import EvalTransform
from models.timm_mobilenet_v2 import TimmMobileNetV2Backbone

torch.set_num_threads(2)
torch.manual_seed(7)
random.seed(7)
report = {'status': 'running', 'torch': str(torch.__version__), 'timm': timm.__version__,
          'device': 'cpu', 'annotations': str(VAL_ANNOTATIONS), 'frames_root': str(VAL_FRAMES),
          'tolerances': {'normalized_input_atol': 6e-7, 'features_atol': 1e-4, 'features_rtol': 1e-4}}
output = Path(__file__).with_name('report.json')
output.write_text(json.dumps(report, indent=2))
raw = json.loads(Path(VAL_ANNOTATIONS).read_text())
classes = sorted(raw['categories'], key=lambda x: int(x['id']))
label_index = {int(c['id']): i for i, c in enumerate(classes)}
annotations = defaultdict(list)
for ann in raw['annotations']:
    if (not ann.get('iscrowd', False) and int(ann['category_id']) in label_index
            and float(ann['bbox'][2]) > 0 and float(ann['bbox'][3]) > 0):
        annotations[int(ann['image_id'])].append(ann)
records = {int(r['id']): r for r in raw['images']}
order = list(records)
random.Random(7).shuffle(order)
chosen = set()
for category in label_index:
    videos = set()
    for image in order:
        record = records[image]
        if record['video_id'] not in videos and any(int(a['category_id']) == category for a in annotations[image]):
            chosen.add(image); videos.add(record['video_id'])
            if len(videos) == 2:
                break
    assert len(videos) == 2, ('Insufficient validation videos', category)
chosen = sorted(chosen)
base = ImageNetVIDDataset(VAL_ANNOTATIONS, VAL_FRAMES, clip_length=1, clip_stride=1)
index = {int(base.images_by_video[v][s]['id']): i for i, (v,s) in enumerate(base.clip_index)}
classification = Subset(VIDClassificationDataset(base, 'timm_mobilenet_v2', training=False), [index[i] for i in chosen])
loader = DataLoader(classification, batch_size=4, shuffle=False, collate_fn=classification_collate)
big, small, targets = [], [], []
for frames, labels in loader:
    big.append(frames.big); small.append(frames.small); targets.append(labels)
views = {'big': torch.cat(big), 'small': torch.cat(small)}
targets = torch.cat(targets)
expected = torch.zeros_like(targets)
references = {'big': [], 'small': []}
for n, image_id in enumerate(chosen):
    record = records[image_id]
    with Image.open(Path(VAL_FRAMES) / record['file_name']) as source:
        assert source.size == (int(record['width']), int(record['height']))
        rgb = source.convert('RGB')
        for branch, size in [('big',320), ('small',160)]:
            pixels = np.array(rgb.resize((size,size), resample=Image.Resampling.BICUBIC), dtype=np.float32) / 255.
            references[branch].append(torch.from_numpy(pixels).permute(2,0,1))
    for a in annotations[image_id]:
        expected[n,label_index[int(a['category_id'])]] = 1
assert torch.equal(targets, expected), 'Classification labels differ from raw annotations'
for key in references:
    references[key] = torch.stack(references[key])
    torch.testing.assert_close(views[key], references[key]*2-1, rtol=0, atol=0)
args = SimpleNamespace(seed=7, batch_size=4, num_workers=0, prefetch_factor=1,
                       sharing_strategy=None, device='cpu', task='classification')
shuffled = list(budget_loader(classification, args, epoch=1, start_batch=0, batches=len(classification)//4))
permutation = torch.randperm(len(classification), generator=torch.Generator().manual_seed(8))[:len(shuffled)*4]
assert torch.equal(torch.cat([x[0].big for x in shuffled]), views['big'][permutation])
assert torch.equal(torch.cat([x[0].small for x in shuffled]), views['small'][permutation])
assert torch.equal(torch.cat([x[1] for x in shuffled]), expected[permutation])
report.update(classification_frames=len(chosen), classification_videos=len({records[i]['video_id'] for i in chosen}),
              class_coverage=int((targets.sum(0)>0).sum()), sample_image_ids=chosen,
              label_check='exact', raw_resize_check='exact', budget_loader_check='exact', branches=[])
print(f'PASS: {len(chosen)} real classification frames, all 30 classes, labels and both dataloaders.', flush=True)

for branch, width in [('big',1.2), ('small',.5)]:
    wrapper = TimmMobileNetV2Backbone(width,32).eval()
    provenance = wrapper.load_pretrained()
    reference = timm.create_model(wrapper.model_name, pretrained=True, cache_dir=str(TIMM_CACHE)).eval()
    cfg = timm.data.resolve_model_data_config(reference)
    assert cfg == wrapper.data_config
    reference.reset_classifier(0)
    x = views[branch]
    direct = (references[branch] - torch.tensor(cfg['mean'])[None,:,None,None]) / torch.tensor(cfg['std'])[None,:,None,None]
    normalized = wrapper.normalize(x)
    torch.testing.assert_close(normalized, direct, rtol=0, atol=6e-7)
    phases = [('original_stride32',wrapper)]
    if branch == 'small':
        adapted = TimmMobileNetV2Backbone(width,16).eval()
        adapted.load_pretrained()
        phases.append(('adapted_stride16',adapted))
    for phase, model in phases:
        if phase == 'adapted_stride16':
            reference.blocks[5][0].conv_dw.stride=(1,1)
        assert all(torch.equal(w, reference.state_dict()[k]) for k,w in model.network.state_dict().items())
        max_reference, max_single, max_equal_input = 0.,0.,0.
        with torch.no_grad():
            for start in range(0,len(x),4):
                batch=x[start:start+4]
                actual=model(batch)
                independent=reference.forward_features(direct[start:start+4])
                same_input=reference.forward_features(model.normalize(batch))
                single=torch.cat([model(frame[None]) for frame in batch])
                torch.testing.assert_close(actual,same_input,rtol=0,atol=0)
                torch.testing.assert_close(actual,independent,rtol=1e-4,atol=1e-4)
                torch.testing.assert_close(actual,single,rtol=1e-4,atol=1e-4)
                assert torch.isfinite(actual).all()
                side=5 if branch=='small' and phase=='original_stride32' else 10
                assert actual.shape[1:] == (1280,side,side)
                max_reference=max(max_reference,float((actual-independent).abs().max()))
                max_single=max(max_single,float((actual-single).abs().max()))
        row=dict(branch=branch,model_name=model.model_name,phase=phase,pretrained_tensors_equal=True,
                 normalized_input_max_abs_error=float((normalized-direct).abs().max()),
                 reference_feature_max_abs_error=max_reference,batch_vs_single_max_abs_error=max_single,
                 identical_input_feature_max_abs_error=max_equal_input,output_per_frame=[1280,side,side],
                 data_config=cfg)
        report['branches'].append(row)
        print('PASS:',row,flush=True)

# Actual six-frame detection dataset and video collator, with random augmentation disabled.
detection=ImageNetVIDDataset(VAL_ANNOTATIONS, VAL_FRAMES,
    transform=EvalTransform(320, interpolation='bicubic', paired=True),clip_length=6,clip_stride=6)
indices=[]; videos=set()
for i,(video,start) in enumerate(detection.clip_index):
    if video in {records[k]['video_id'] for k in chosen} and video not in videos:
        indices.append(i);videos.add(video)
        if len(indices)==8: break
count=0
for batch, target_batch in build_dataloader(Subset(detection,indices),batch_size=2,shuffle=False):
    for b, clip_targets in enumerate(target_batch):
        for t, target in enumerate(clip_targets):
            image_id=int(target['image_id']); record=records[image_id]
            with Image.open(Path(VAL_FRAMES)/record['file_name']) as im:
                for key,size in [('big',320),('small',160)]:
                    pixels=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/255.
                    reference_frame=torch.from_numpy(pixels).permute(2,0,1)*2-1
                    torch.testing.assert_close(getattr(batch,key)[b,t],reference_frame,rtol=0,atol=0)
            expected_labels=torch.tensor([label_index[int(a['category_id'])]+1 for a in annotations[image_id]],dtype=torch.long)
            assert torch.equal(target['labels'],expected_labels)
            boxes=[]
            for a in annotations[image_id]:
                x,y,w,h=map(float,a['bbox'])
                boxes.append([x/record['width'],y/record['height'],(x+w)/record['width'],(y+h)/record['height']])
            torch.testing.assert_close(target['boxes'],torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),rtol=0,atol=2e-7)
            count+=1
report.update(status='passed',detection_clips=len(indices),detection_frames=count,
              detection_labels_boxes_and_resize='passed',
              limitations=['CPU float32 only; no CUDA validation.', 'No random crop/flip augmentation in this diagnostic.',
                           'Checks pretrained backbones, not server-trained checkpoints or classification/detection accuracy.'])
output.write_text(json.dumps(report,indent=2)+'\n')
print('PASS: detection loader checks;',count,'frames. Report:',output,flush=True)
