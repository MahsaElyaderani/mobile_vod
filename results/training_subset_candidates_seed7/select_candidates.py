"""Reproduce nested whole-video candidates from the adjacent annotation audit."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
raw = json.loads((root / 'training_audit.json').read_text())
rows, classes = raw['per_video'], raw['classes']
frames = np.array([r['frames'] for r in rows], dtype=np.int64)
clips = np.array([r['clips'] for r in rows], dtype=np.int64)
f = np.array([[r['positive_frames'].get(str(c['category_id']), 0) for c in classes] for r in rows], dtype=np.float64)
c = np.array([[r['positive_clips'].get(str(k['category_id']), 0) for k in classes] for r in rows], dtype=np.float64)
v = (c > 0).astype(np.float64)
order = np.random.default_rng(7).permutation(len(rows))


def describe(indices):
    indices = np.asarray(indices, dtype=int)
    fs, cs, vs = f[indices].sum(0), c[indices].sum(0), v[indices].sum(0)
    nf, nc = int(frames[indices].sum()), int(clips[indices].sum())
    report = dict(videos=len(indices), frames=nf, frame_fraction=nf/raw['frames'], clips=nc,
                  classification_updates=nf//16, detection_updates=nc//8,
                  minimum_clip_videos_per_class=int(vs.min()),
                  positive_frame_max_min_ratio=float(fs.max()/fs.min()) if fs.min() else None,
                  positive_clip_max_min_ratio=float(cs.max()/cs.min()) if cs.min() else None,
                  positive_frame_cv=float(fs.std()/fs.mean()), positive_clip_cv=float(cs.std()/cs.mean()),
                  largest_video_frame_share=float(frames[indices].max()/nf),
                  worst_class_single_video_share=float((f[indices].max(0)/np.maximum(fs, 1)).max()))
    report['classes'] = [dict(name=k['name'], category_id=k['category_id'], positive_frames=int(fs[j]),
        positive_clips=int(cs[j]), clip_videos=int(vs[j]),
        largest_video_positive_frame_share=float(f[indices,j].max()/fs[j]) if fs[j] else None)
        for j, k in enumerate(classes)]
    return report


selected = []
used = np.zeros(len(rows), dtype=bool)
fs, cs, vs = np.zeros(30), np.zeros(30), np.zeros(30)
consumed = 0
reports = {}
for percent in (10, 25, 50):
    target = raw['frames'] * percent // 100
    fq, cq = f.sum() * percent / 100 / 30, c.sum() * percent / 100 / 30
    while True:
        eligible = order[(~used[order]) & (frames[order] <= target-consumed)]
        if not len(eligible):
            break
        # Reduce squared deviations from equal-class targets per added frame.
        # A small extra reward favors distinct videos containing each class.
        gain = (((2*(fq-fs)*f[eligible] - f[eligible]**2)/fq**2).sum(1)
                + ((2*(cq-cs)*c[eligible] - c[eligible]**2)/cq**2).sum(1)
                + .05*np.log1p(v[eligible]/(vs+1)).sum(1)) / frames[eligible]
        index = eligible[np.argmax(gain)]
        selected.append(int(index)); used[index] = True
        fs += f[index]; cs += c[index]; vs += v[index]; consumed += int(frames[index])
    report = describe(selected)
    assert report['minimum_clip_videos_per_class'] >= 10
    assert len(set(selected)) == len(selected) and consumed <= target
    manifest = dict(annotations_sha256=raw['annotations_sha256'],
                    video_ids=sorted(rows[i]['video_id'] for i in selected),
                    selection=dict(seed=7, target_frame_percent=percent, target_frames=target,
                        nested=True, minimum_verified_clip_videos_per_class=10,
                        method='Greedy reduction in squared deviations from equal-class positive-frame/clip targets per added frame, plus 0.05*sum-log clip-video coverage reward. Targets use full-data positive-count totals times the requested frame fraction divided by 30.',
                        clip_length=6, clip_stride=6), report=report)
    (root / f'train_{percent}pct.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Same seed, whole videos and frame cap, without class-aware scoring.
    random_indices, random_frames = [], 0
    for i in order:
        if random_frames + frames[i] <= target:
            random_indices.append(int(i)); random_frames += int(frames[i])
    reports[str(percent)] = dict(balanced=report, random_baseline=describe(random_indices))
summary = dict(annotations_sha256=raw['annotations_sha256'], full=describe(range(len(rows))), candidates=reports,
               limitations=['Heuristic selection, not an optimal-balance guarantee.',
                            'Video IDs are a diversity proxy, not verified scene/appearance diversity.',
                            'Counts are before augmentation and final incomplete-batch dropping.',
                            'A positive clip contains the class in at least one of its six frames.'])
(root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
for percent, item in reports.items():
    print(percent, {k:v for k,v in item['balanced'].items() if k!='classes'})
    print('random baseline', {k:v for k,v in item['random_baseline'].items() if k in ('videos','positive_frame_cv','positive_clip_cv','minimum_clip_videos_per_class')})
