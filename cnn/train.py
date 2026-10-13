"""Train the RIS CNN to predict total electric and magnetic fields."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from configuration import DEFAULT_CONFIG, load_config, sample_counts as configured_samples, validate_config
from model import UNetAttention
from fields import build_sample_tensors, discover_groups, sample_files


SCRIPT_DIR = Path(__file__).resolve().parent
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Experiment JSON file; defaults to config.json beside this script.")
    parser.add_argument("--all-data", action=argparse.BooleanOptionalAction, default=None,
                        help="Use every training sample instead of the configured random sample counts.")
    parser.add_argument("--check-data", action="store_true",
                        help="Load and validate the selected data without creating or training a model.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda:0, etc.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for key in ("seed", "epochs", "batch_size", "patience", "device"):
            value = getattr(args, key)
            if value is not None:
                config["training"][key] = value
        if args.all_data is not None:
            config["data"]["all_data"] = args.all_data
        validate_config(config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args, config


def selected_samples(folder, count, seed, size, angle_deg):
    available = sample_files(folder)
    if count > len(available):
        raise ValueError(f"{folder.name}: requested {count} samples but only {len(available)} are available")
    # Independent group seeds keep a group's selection stable when another budget changes.
    generator = random.Random(f"{seed}:{size}:{angle_deg}")
    return generator.sample(available, count)


def load_samples(sample_counts, seed, all_data=False, check_only=False, *, magnetic_scale):
    groups = discover_groups(SCRIPT_DIR / "data" / "training")
    group_paths = {(size, angle): folder for size, angle, folder in groups}
    if len(group_paths) != len(groups):
        raise ValueError("Training folders must uniquely identify each size and angle")
    if all_data:
        sample_counts = {key: len(sample_files(folder)) for key, folder in group_paths.items()}
    tensors_by_size, sample_paths = {}, {}
    for (size, angle), count in sorted(sample_counts.items()):
        if count == 0:
            continue
        if (size, angle) not in group_paths:
            raise FileNotFoundError(f"No training folder for size {size}, angle {angle} degrees")
        files = selected_samples(group_paths[size, angle], count, seed, size, angle)
        tensors = tensors_by_size.setdefault(size, {"inputs": [], "targets": []})
        paths = sample_paths.setdefault(size, [])
        for path in files:
            inputs, targets = build_sample_tensors(path, size, magnetic_scale)
            if not check_only:
                tensors["inputs"].append(inputs)
                tensors["targets"].append(targets)
            paths.append(path.relative_to(SCRIPT_DIR).as_posix())
        print(f"{size}x{size}, {angle} degrees: randomly selected {len(files)} samples", flush=True)
    if not sample_paths:
        raise ValueError("Select at least one training group with a positive sample count")
    datasets = {}
    if not check_only:
        for size, tensors in tensors_by_size.items():
            if len(tensors["inputs"]) < 2:
                raise ValueError(f"Size {size} requires at least two samples for training and validation")
            datasets[size] = TensorDataset(torch.stack(tensors["inputs"]), torch.stack(tensors["targets"]))
    print(f"Total selected: {sum(len(paths) for paths in sample_paths.values())} samples", flush=True)
    return datasets, sample_paths


def save_checkpoint(model_state_dict, config, epoch, validation_loss, path):
    """Replace the best checkpoint only after a complete file has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({
            "model_config": config["model"],
            "magnetic_scale": config["data"]["magnetic_scale"],
            "model_state_dict": model_state_dict,
            "best_validation_loss": validation_loss,
            "epoch": epoch,
        }, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args, config = parse_args()
    training, data = config["training"], config["data"]
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    seed = training["seed"]
    magnetic_scale = data["magnetic_scale"]
    sample_counts = {} if data["all_data"] else configured_samples(data["samples_per_group"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Checking data does not allocate the large CNN or run any optimization.
    if args.check_data:
        load_samples(sample_counts, seed, data["all_data"], check_only=True, magnetic_scale=magnetic_scale)
        print("Data check passed.")
        return

    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if training["device"] == "auto" else training["device"]
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}", flush=True)
    model = UNetAttention(config["model"]).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, config["optimizer"]["name"])(
        model.parameters(), **config["optimizer"]["parameters"]
    )
    datasets, sample_paths = load_samples(sample_counts, seed, data["all_data"], magnetic_scale=magnetic_scale)

    loaders, splits = {}, {}
    for size, dataset in datasets.items():
        train_size = int(data["training_fraction"] * len(dataset))
        if not 0 < train_size < len(dataset):
            raise ValueError(f"Size {size}: training_fraction must leave nonempty training and validation sets")
        train, validation = random_split(
            dataset, [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(seed),
        )
        loaders[size] = {
            "train": DataLoader(train, batch_size=training["batch_size"], shuffle=True),
            "validation": DataLoader(validation, batch_size=training["batch_size"], shuffle=False),
        }
        splits[size] = {
            "train": [sample_paths[size][index] for index in train.indices],
            "validation": [sample_paths[size][index] for index in validation.indices],
        }

    results_dir = SCRIPT_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    results_dir.mkdir(parents=True, exist_ok=False)
    (results_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (results_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    checkpoint_path = results_dir / "model.pth"
    print(f"Best model will be saved to: {checkpoint_path}", flush=True)
    best_val_loss = float("inf")
    best_weights, best_epoch = None, None
    epochs_no_improve = 0
    stopped_early = False
    with (results_dir / "losses.csv").open("w", newline="", encoding="utf-8") as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(["epoch", "training_loss", "validation_loss"])
        for epoch in range(training["epochs"]):
            model.train()
            train_losses, validation_losses = [], []
            for size in sorted(loaders):
                for inputs, targets in loaders[size]["train"]:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                for size in sorted(loaders):
                    for inputs, targets in loaders[size]["validation"]:
                        inputs, targets = inputs.to(device), targets.to(device)
                        validation_losses.append(criterion(model(inputs), targets).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(validation_losses))
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise RuntimeError("Training produced non-finite losses")
            writer.writerow([epoch + 1, train_loss, val_loss])
            loss_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print(f"Epoch [{epoch + 1}/{training['epochs']}], Training Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}, Best Loss: {best_val_loss:.6f}", flush=True)
            if epochs_no_improve >= training["patience"]:
                stopped_early = True
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    save_checkpoint(best_weights, config, best_epoch, best_val_loss, checkpoint_path)
    print(f"Saved best model from epoch {best_epoch}: {checkpoint_path}", flush=True)
    ended_at = datetime.now().astimezone()
    elapsed = time.perf_counter() - start_time
    summary = [
        "RIS CNN training summary",
        f"Start time: {started_at.isoformat(timespec='seconds')}",
        f"End time: {ended_at.isoformat(timespec='seconds')}",
        f"Elapsed time: {timedelta(seconds=round(elapsed))} ({elapsed:.3f} seconds)",
        f"Completion: {'early stopping' if stopped_early else 'maximum epochs reached'}",
        f"Completed epochs: {epoch + 1} / {training['epochs']}",
        f"Best epoch: {best_epoch}",
        f"Best validation MSE: {best_val_loss:.10g}",
        f"Final training MSE: {train_loss:.10g}",
        f"Final validation MSE: {val_loss:.10g}",
        f"Selected samples: {sum(len(paths) for paths in sample_paths.values())}",
        f"Training samples: {sum(len(parts['train']) for parts in splits.values())}",
        f"Validation samples: {sum(len(parts['validation']) for parts in splits.values())}",
        f"Device: {device}",
        "Checkpoint: model.pth (best validation weights)",
        "Training configuration: config.json",
    ]
    (results_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Total running time: {elapsed:.1f} seconds")
    print(f"Saved training records to: {results_dir}")


if __name__ == "__main__":
    main()
