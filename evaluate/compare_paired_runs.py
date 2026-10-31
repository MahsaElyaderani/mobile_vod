import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import parse_paths
import argparse
import csv
from itertools import zip_longest
import json

import cv2
import numpy as np

if __package__ in (None, ""):
    from compare_visual_runs import COMPARISON_KEYS, row_tensors, validate_runs
else:
    from .compare_visual_runs import COMPARISON_KEYS, row_tensors, validate_runs
from engine.visualizer import render_frame
from utils.paired_checkpoints import write_json
from utils.runtime import tqdm


def compare_runs(run_paths, frames_root, output_dir, labels=None, threshold=.3, max_side=480,
                 playback_fps=30., save_frames_every=30, progress=True, model_only_titles=False):
    import math
    paths = [Path(path) for path in run_paths]
    labels = labels or [path.name for path in paths]
    if (len(paths) < 2 or len(labels) != len(paths) or len(set(labels)) != len(labels) or
            not 0 <= threshold <= 1 or max_side < 1 or save_frames_every < 0 or
            not math.isfinite(playback_fps) or playback_fps <= 0):
        raise ValueError("Invalid comparison configuration.")
    metrics = [json.loads((path / "metrics.json").read_text()) for path in paths]
    for path in paths[1:]:
        validate_runs(paths[0], path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    if save_frames_every:
        (output / "frames").mkdir()
    names = {int(key): value for key, value in metrics[0]["class_names"].items()}
    streams, writer, current, indices, seen, shape = [], None, None, [], [], None
    write_json({"status": "running"}, output / "run_status.json")
    try:
        streams = [(path / "detections.jsonl").open() for path in paths]
        for lines in tqdm(zip_longest(*streams), total=metrics[0]["num_frames"], desc="Compare saved modes",
                          unit="frame", disable=not progress):
            if any(line is None for line in lines):
                raise ValueError("Source detection files have different lengths.")
            rows = [json.loads(line) for line in lines]
            for row in rows[1:]:
                for key in ("image_id", "video_id", "frame_id", "file_name", "original_size", "ground_truth"):
                    if row[key] != rows[0][key]:
                        raise ValueError(f"Source frames differ at {key}.")
            image = cv2.imread(str(Path(frames_root) / rows[0]["file_name"]))
            if image is None or [image.shape[1], image.shape[0]] != rows[0]["original_size"]:
                raise ValueError("Comparison frame missing or dimensions differ.")
            panels = []
            for index, (row, label) in enumerate(zip(rows, labels)):
                target, prediction = row_tensors(row)
                route = row.get("route", {})
                title = label if model_only_titles else f"{label} {route.get('selected_model', '')} age={route.get('previous_contribution_age', '-')}"
                canvas = render_frame(image, target, prediction, names, title, threshold, "side-by-side", max_side,
                                      model_only_titles=model_only_titles)
                half = canvas.shape[1] // 2
                if index == 0:
                    panels.append(canvas[:, :half])
                panels.append(canvas[:, half:])
            canvas = np.concatenate(panels, axis=1)
            canvas = cv2.copyMakeBorder(canvas, 0, canvas.shape[0] % 2, 0, canvas.shape[1] % 2,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if current != rows[0]["video_id"]:
                if writer:
                    writer.release()
                current, shape = rows[0]["video_id"], canvas.shape[:2]
                filename = f"video_{current}_comparison.mp4"
                writer = cv2.VideoWriter(str(output / "videos" / filename), cv2.VideoWriter_fourcc(*"mp4v"),
                                         playback_fps, (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not open comparison video writer.")
                indices.append({"video_id": current, "path": "videos/" + filename, "num_frames": 0})
            if canvas.shape[:2] != shape:
                raise ValueError("Frame dimensions changed within a video.")
            writer.write(canvas)
            if save_frames_every and indices[-1]["num_frames"] % save_frames_every == 0:
                frame_path = output / "frames" / f"video_{current}_image_{rows[0]['image_id']}.jpg"
                if not cv2.imwrite(str(frame_path), canvas):
                    raise IOError("Could not save comparison frame.")
            indices[-1]["num_frames"] += 1
            seen.append(rows[0]["image_id"])
        if seen != metrics[0]["evaluated_image_ids"]:
            raise ValueError("Detection rows do not match evaluated frame IDs.")
        with (output / "metrics.csv").open("x", newline="") as stream:
            table = csv.writer(stream)
            table.writerow(["mode", "AP50_percent", "FPS", "conv_linear_GFLOPs_per_frame", "big_call_frequency"])
            for label, item in zip(labels, metrics):
                table.writerow([label, 100*item["mAP@0.5"], item["FPS"], item["conv_linear_FLOPs_per_frame"]/1e9,
                                item["big_call_frequency"]])
        with (output / "per_class.csv").open("x", newline="") as stream:
            table = csv.writer(stream)
            table.writerow(["class"] + [label + "_AP50_percent" for label in labels])
            for key, name in sorted(names.items()):
                values = [item["AP@0.5_per_class"].get(str(key)) for item in metrics]
                table.writerow([name] + ["NA" if value is None else 100*value for value in values])
        write_json({"videos": indices}, output / "videos_index.json")
        write_json({"labels": labels, "verified_comparison_fields": COMPARISON_KEYS,
                    "source_runs": [str(path.resolve()) for path in paths], "num_frames": len(seen),
                    "layout": "ground truth | " + " | ".join(labels),
                    "model_only_titles": model_only_titles}, output / "comparison.json")
        write_json({"status": "complete", "num_frames": len(seen)}, output / "run_status.json")
    except BaseException as error:
        write_json({"status": "failed", "error": str(error)}, output / "run_status.json")
        raise
    finally:
        for stream in streams:
            stream.close()
        if writer:
            writer.release()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw ground truth and matched predictions from saved paired runs.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--draw-score-threshold", type=float, default=.3)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--playback-fps", type=float, default=30.)
    parser.add_argument("--save-frames-every", type=int, default=30)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--model-only-titles", action="store_true",
                        help="Use only model labels in panel headers; retain box labels and scores.")
    a = parse_paths(parser, "compare_paired_runs", argv)
    compare_runs(a.runs, a.frames_root, a.output_dir, a.labels, a.draw_score_threshold, a.max_side,
                 a.playback_fps, a.save_frames_every, not a.no_progress, a.model_only_titles)


if __name__ == "__main__":
    main()
