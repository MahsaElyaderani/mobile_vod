"""Check candidate manifests/counts against the project's actual dataset loader."""
import gc
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]))
from path import TRAIN_ANNOTATIONS
from routing.data import selected_dataset

source_hash = hashlib.sha256(Path(TRAIN_ANNOTATIONS).read_bytes()).hexdigest()
previous = set()
for percent in (10, 25, 50):
    file = root / f'train_{percent}pct.json'
    manifest = json.loads(file.read_text())
    report = manifest['report']
    ids = manifest['video_ids']
    assert manifest['annotations_sha256'] == source_hash
    assert len(set(ids)) == len(ids) and previous.issubset(ids)
    previous = set(ids)
    args = SimpleNamespace(annotations=str(TRAIN_ANNOTATIONS), frames_root=str(root),
                           video_ids=str(file), max_videos=None, subset_seed=7)
    # Index/count verification reads annotations only; no images or training are needed.
    dataset = selected_dataset(args, clip_length=6, clip_stride=6)
    assert len(dataset) == report['clips']
    assert len(dataset.video_ids) == report['videos']
    assert len(dataset.routing_records) == report['frames']
    labels = {}
    fs, cs, vc = Counter(), Counter(), Counter()
    for image in dataset.routing_records:
        labels[image] = {int(a['category_id']) for a in dataset.anns_by_image.get(image, [])
                        if int(a['category_id']) in dataset.category_id_to_label
                        and float(a['bbox'][2]) > 0 and float(a['bbox'][3]) > 0}
        fs.update(labels[image])
    video_classes = {i: set() for i in ids}
    for vid, start in dataset.clip_index:
        present = set()
        for record in dataset.images_by_video[vid][start:start+6]:
            present.update(labels[int(record['id'])])
        cs.update(present); video_classes[vid].update(present)
    for present in video_classes.values():
        vc.update(present)
    for row in report['classes']:
        k = row['category_id']
        assert (fs[k], cs[k], vc[k]) == (row['positive_frames'], row['positive_clips'], row['clip_videos'])
    del dataset, labels
    gc.collect()
    dataset = selected_dataset(args, clip_length=1, clip_stride=1)
    assert len(dataset) == report['frames']
    del dataset
    gc.collect()
    print(f'{percent}%: verified loader compatibility, counts, coverage, hash and nesting.', flush=True)
files = [root/f'train_{p}pct.json' for p in (10,25,50)] + [root/'comparison.json']
before = [p.read_bytes() for p in files]
subprocess.run([sys.executable, '-B', str(root/'select_candidates.py')], check=True, stdout=subprocess.DEVNULL)
assert before == [p.read_bytes() for p in files]
print('Regeneration is byte-for-byte identical. No training or validation configuration changed.')
