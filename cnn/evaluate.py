"""Evaluate total-field prediction on the separate RIS test set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration import DEFAULT_CONFIG, load_config, load_evaluation_config, sample_counts
from fields import build_sample_tensors, discover_groups, sample_files
from model import UNetAttention


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_CONFIG = SCRIPT_DIR / "evaluation_config.json"


def evaluation_groups(data_dir, selected_groups, counts=None):
    data_dir = Path(data_dir).resolve()
    if any(data_dir.glob("sample_*.fld")):
        # A prepared size/angle folder can be selected directly.
        groups = [group for group in discover_groups(data_dir.parent) if group[2].resolve() == data_dir]
        if not groups:
            raise ValueError(f"Test folder must use the size/angle naming convention: {data_dir.name}")
    else:
        groups = discover_groups(data_dir)
    if selected_groups is not None:
        available = {folder.name for _, _, folder in groups}
        missing = set(selected_groups) - available
        if missing:
            raise ValueError(f"Dataset groups not found: {', '.join(sorted(missing))}")
        groups = [group for group in groups if group[2].name in selected_groups]
    if counts is not None:
        available = {(size, angle) for size, angle, _ in groups}
        missing = {key for key, count in counts.items() if count > 0} - available
        if missing:
            raise ValueError(f"Test size/angle groups not found: {sorted(missing)}")
        groups = [group for group in groups if counts.get(group[:2], 0) > 0]
    if not groups:
        raise ValueError("Select at least one test group with a positive sample count")
    return groups


def test_cases(folder, num_cases, seed):
    files = sample_files(folder)
    if num_cases is None:
        return files
    if num_cases > len(files):
        raise ValueError(f"{folder.name}: requested {num_cases} test cases but only {len(files)} are available")
    if seed is None:
        raise ValueError("A configured seed is required when selecting test cases")
    return random.Random(f"{seed}:evaluation:{folder.name}").sample(files, num_cases)


def check_test_data(data_dir, magnetic_scale, groups=None, samples_per_group=None, seed=None):
    counts = None if samples_per_group is None else sample_counts(samples_per_group)
    total = 0
    for size, angle, folder in evaluation_groups(data_dir, groups, counts):
        count = None if counts is None else counts[size, angle]
        files = test_cases(folder, count, seed)
        for path in files:
            build_sample_tensors(path, size, magnetic_scale)
        total += len(files)
        print(f"Test {folder.name}: checked {len(files)} samples", flush=True)
    print(f"Total test samples: {total}", flush=True)


def evaluate_model(model, device, data_dir, batch_size, magnetic_scale, groups=None, samples_per_group=None, seed=None, selected_paths=None):
    counts = None if samples_per_group is None else sample_counts(samples_per_group)
    model.eval()
    records = []
    with torch.no_grad():
        for size, angle, folder in evaluation_groups(data_dir, groups, counts):
            count = None if counts is None else counts[size, angle]
            files = test_cases(folder, count, seed)
            if selected_paths is not None:
                selected_paths[folder.name] = [path.relative_to(data_dir).as_posix() for path in files]
            samples = [build_sample_tensors(path, size, magnetic_scale) for path in files]
            inputs, targets = zip(*samples)
            loader = DataLoader(TensorDataset(torch.stack(inputs), torch.stack(targets)), batch_size=batch_size)
            total_loss = 0.0
            for inputs, targets in loader:
                outputs = model(inputs.to(device))
                loss = torch.nn.functional.mse_loss(outputs, targets.to(device))
                total_loss += loss.item() * len(inputs)
            test_loss = total_loss / len(samples)
            records.append({"folder": folder.name, "size": size, "incident_angle_deg": angle,
                            "samples": len(samples), "test_loss": test_loss})
            print(f"Test {folder.name}: {len(samples)} samples, total-field MSE {test_loss:.6f}", flush=True)
    overall = sum(record["test_loss"] * record["samples"] for record in records) / sum(record["samples"] for record in records)
    print(f"Sample-weighted overall total-field MSE: {overall:.6f}", flush=True)
    return records


def save_evaluation_results(records, selected_paths, evaluation, model_path,
                            device, batch_size, started_at, elapsed):
    ended_at = datetime.now().astimezone()
    output_dir = model_path.parent / "evaluation"
    output_dir.mkdir(exist_ok=True)
    with (output_dir / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["folder", "size", "incident_angle_deg", "samples", "test_loss"])
        writer.writeheader()
        writer.writerows(records)
    total_samples = sum(record["samples"] for record in records)
    overall = sum(record["test_loss"] * record["samples"] for record in records) / total_samples
    try:
        dataset_path = Path(os.path.relpath(evaluation["data"]["dataset"], output_dir)).as_posix()
    except ValueError:
        # Windows cannot express a relative path between different drives.
        dataset_path = str(evaluation["data"]["dataset"])
    effective_config = {
        "checkpoint": "../" + model_path.name,
        "batch_size": batch_size,
        "device": str(device),
        "data": {**evaluation["data"], "dataset": dataset_path},
    }
    (output_dir / "evaluation_config.json").write_text(json.dumps(effective_config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "samples.json").write_text(json.dumps(selected_paths, indent=2) + "\n", encoding="utf-8")
    summary = [
        "RIS CNN evaluation summary",
        f"Start time: {started_at.isoformat(timespec='seconds')}",
        f"End time: {ended_at.isoformat(timespec='seconds')}",
        f"Elapsed time: {timedelta(seconds=round(elapsed))} ({elapsed:.3f} seconds)",
        f"Checkpoint: ../{model_path.name}",
        "Training configuration: ../config.json",
        f"Dataset: {dataset_path}",
        f"Device: {device}",
        f"Batch size: {batch_size}",
        f"Evaluated groups: {len(records)}",
        f"Evaluated samples: {total_samples}",
        f"Sample-weighted overall total-field MSE: {overall:.10g}",
        "",
        "Per-group total-field MSE:",
    ]
    summary.extend(f"{record['folder']}: {record['samples']} samples, MSE {record['test_loss']:.10g}" for record in records)
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_dir


def latest_checkpoint(results_dir):
    checkpoints = [path for path in results_dir.glob("run_*/model.pth") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(
            f"No saved model found in {results_dir}. "
            "Run python train.py; the best model is saved automatically when training finishes. "
            "Alternatively, use --model-path to select a checkpoint."
        )
    return max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-config", type=Path, default=EVALUATION_CONFIG,
                        help="Evaluation JSON file; defaults to evaluation_config.json beside this script.")
    parser.add_argument("--check-data", action="store_true", help="Check test files without loading a model.")
    parser.add_argument("--model-path", type=Path, help="Override the checkpoint in the evaluation config.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    try:
        evaluation = load_evaluation_config(args.evaluation_config)
        if args.check_data:
            config_path = DEFAULT_CONFIG
        else:
            args.model_path = args.model_path or evaluation["checkpoint"] or latest_checkpoint(SCRIPT_DIR / "results")
            args.model_path = args.model_path.resolve()
            if args.model_path.is_dir():
                args.model_path = args.model_path / "model.pth"
            config_path = args.model_path.parent / "config.json"
        config = load_config(config_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.batch_size is None:
        args.batch_size = evaluation["batch_size"] or config["training"]["batch_size"]
    if args.device is None:
        args.device = evaluation["device"] or config["training"]["device"]
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    magnetic_scale = config["data"]["magnetic_scale"]
    if args.check_data:
        check_test_data(evaluation["data"]["dataset"], magnetic_scale, evaluation["data"]["groups"],
                        evaluation["data"]["samples_per_group"], config["training"]["seed"])
        return
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    print(f"Evaluating checkpoint: {args.model_path.resolve()}", flush=True)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if "magnetic_scale" not in checkpoint:
        raise ValueError("Checkpoint is missing magnetic_scale metadata; its training scale must be established before evaluation")
    if checkpoint["magnetic_scale"] != magnetic_scale:
        raise ValueError(
            f"Magnetic scale mismatch: config uses {magnetic_scale}, "
            f"but the checkpoint was trained with {checkpoint['magnetic_scale']}. "
            "The config.json beside the checkpoint must match its training settings."
        )
    print(f"Dataset: {evaluation['data']['dataset']}", flush=True)
    model = UNetAttention(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_paths = {}
    records = evaluate_model(model, device, evaluation["data"]["dataset"], args.batch_size,
                             magnetic_scale, evaluation["data"]["groups"], evaluation["data"]["samples_per_group"],
                             config["training"]["seed"], selected_paths=selected_paths)
    output_dir = save_evaluation_results(records, selected_paths, evaluation, args.model_path,
                                         device, args.batch_size, started_at, time.perf_counter() - start_time)
    print(f"Saved evaluation results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
