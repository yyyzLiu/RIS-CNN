"""Load and validate the shared CNN experiment configuration."""

import json
import math
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    expected = {
        "model": {"in_channels", "out_channels", "dropout_rate", "features"},
        "data": {"samples_per_group", "training_fraction", "magnetic_scale", "all_data"},
        "training": {"seed", "epochs", "batch_size", "patience", "device"},
        "optimizer": {"name", "parameters"},
    }
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError("Config must contain model, data, training, and optimizer sections")
    for section, keys in expected.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(keys))}")

    def integer(value, label, minimum):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")

    def number(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")

    model, data, training = config["model"], config["data"], config["training"]
    for key, channels in (("in_channels", 5), ("out_channels", 8)):
        integer(model[key], f"model.{key}", 1)
        if model[key] != channels:
            raise ValueError(f"The supplied total-field data require model.{key} = {channels}")
    if not isinstance(model["features"], list) or not model["features"]:
        raise ValueError("model.features must be a nonempty list")
    for width in model["features"]:
        integer(width, "Feature width", 16)
    number(model["dropout_rate"], "model.dropout_rate")
    if not 0 <= model["dropout_rate"] < 1:
        raise ValueError("model.dropout_rate must be >= 0 and < 1")
    number(data["training_fraction"], "data.training_fraction")
    if not 0 < data["training_fraction"] < 1:
        raise ValueError("data.training_fraction must be > 0 and < 1")
    number(data["magnetic_scale"], "data.magnetic_scale")
    if data["magnetic_scale"] <= 0:
        raise ValueError("data.magnetic_scale must be positive")
    sample_counts(data["samples_per_group"])
    integer(training["seed"], "training.seed", 0)
    if training["seed"] >= 2 ** 32:
        raise ValueError("training.seed must be < 2**32")
    for key in ("epochs", "batch_size", "patience"):
        integer(training[key], f"training.{key}", 1)
    if not isinstance(training["device"], str) or not training["device"].strip():
        raise ValueError("training.device must be a nonempty string")
    if type(data["all_data"]) is not bool:
        raise ValueError("all_data must be true or false")
    optimizer = config["optimizer"]
    if optimizer["name"] not in ("Adamax", "Adam", "AdamW", "SGD"):
        raise ValueError("optimizer.name must be Adamax, Adam, AdamW, or SGD")
    if not isinstance(optimizer["parameters"], dict) or "lr" not in optimizer["parameters"]:
        raise ValueError("optimizer.parameters must contain lr")
    number(optimizer["parameters"]["lr"], "optimizer.parameters.lr")
    if optimizer["parameters"]["lr"] <= 0:
        raise ValueError("optimizer.parameters.lr must be positive")


def sample_counts(settings):
    if not isinstance(settings, dict):
        raise ValueError("data.samples_per_group must map sizes to angle/count mappings")
    counts = {}
    for size, angles in settings.items():
        if not isinstance(angles, dict) or not isinstance(size, str) or not size.isdecimal() or int(size) <= 0:
            raise ValueError(f"Invalid sample group size: {size}")
        for angle, count in angles.items():
            if not isinstance(angle, str) or not angle.lstrip("-").isdecimal():
                raise ValueError(f"Invalid incident angle: {angle}")
            if type(count) is not int or count < 0:
                raise ValueError(f"Sample count for size {size}, angle {angle} must be a nonnegative integer")
            key = (int(size), int(angle))
            if key in counts:
                raise ValueError(f"Duplicate size/angle group: {key}")
            counts[key] = count
    return counts


def load_evaluation_config(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    keys = {"checkpoint", "data", "batch_size", "device"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError(f"Evaluation config must contain exactly: {', '.join(sorted(keys))}")
    data = settings["data"]
    if not isinstance(data, dict) or set(data) != {"dataset", "groups", "samples_per_group"}:
        raise ValueError("Evaluation data must contain dataset, groups, and samples_per_group")
    for section, key in ((settings, "checkpoint"), (data, "dataset")):
        value = section[key]
        if key == "checkpoint" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Evaluation {key} must be a nonempty path string")
        section[key] = (path.parent / value).resolve()
    groups = data["groups"]
    if groups is not None:
        if not isinstance(groups, list) or not groups or any(
            not isinstance(group, str) or not group or group in (".", "..")
            or "/" in group or "\\" in group for group in groups
        ):
            raise ValueError("Evaluation groups must be null or a nonempty list of dataset folder names")
        if len(set(groups)) != len(groups):
            raise ValueError("Evaluation groups must not contain duplicates")
    if data["samples_per_group"] is not None:
        counts = sample_counts(data["samples_per_group"])
        if not any(counts.values()):
            raise ValueError("Select at least one test size/angle with a positive sample count")
    batch_size = settings["batch_size"]
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError("Evaluation batch_size must be null or a positive integer")
    device = settings["device"]
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("Evaluation device must be null or a nonempty string")
    return settings
