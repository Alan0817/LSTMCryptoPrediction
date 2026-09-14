"""Versioned, state-dict based persistence for LSTM training artifacts."""

from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    DROPOUT,
    FEATURE_COLUMNS,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LOWER_SIGNAL_THRESHOLD,
    MODEL_PATH,
    NUM_LAYERS,
    SEQUENCE_LENGTH,
    UPPER_SIGNAL_THRESHOLD,
)
from model import LSTMClassifier


@dataclass
class LoadedModelArtifact:
    model: LSTMClassifier
    scaler: object
    feature_columns: tuple
    sequence_length: int
    upper_signal_threshold: float
    lower_signal_threshold: float
    model_metadata: dict
    training_metadata: dict


def default_model_metadata():
    return {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }


def _serialize_scaler(scaler):
    """Store fitted scaler numbers, not a pickleable sklearn object."""
    required = ("mean_", "scale_", "var_", "n_features_in_")
    missing = [name for name in required if not hasattr(scaler, name)]
    if missing:
        raise ValueError("Scaler has not been fitted; missing attributes: {}".format(missing))
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


def _deserialize_scaler(state):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=float)
    scaler.scale_ = np.asarray(state["scale"], dtype=float)
    scaler.var_ = np.asarray(state["var"], dtype=float)
    scaler.n_features_in_ = int(state["n_features_in"])
    scaler.n_samples_seen_ = int(state["n_samples_seen"])
    return scaler


def save_model_artifact(
    model,
    scaler,
    training_metadata,
    output_path=MODEL_PATH,
    feature_columns=FEATURE_COLUMNS,
    sequence_length=SEQUENCE_LENGTH,
    upper_signal_threshold=UPPER_SIGNAL_THRESHOLD,
    lower_signal_threshold=LOWER_SIGNAL_THRESHOLD,
    model_metadata=None,
):
    """Persist state and preprocessing metadata; never serialize the model object itself."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "scaler_state": _serialize_scaler(scaler),
        "feature_columns": list(feature_columns),
        "sequence_length": sequence_length,
        "upper_signal_threshold": upper_signal_threshold,
        "lower_signal_threshold": lower_signal_threshold,
        "model_metadata": model_metadata or default_model_metadata(),
        "training_metadata": training_metadata,
    }
    torch.save(artifact, output_path)
    return output_path


def load_model_artifact(artifact_path=MODEL_PATH, device=None):
    """Load a trusted state-dict artifact and reconstruct the fixed LSTM architecture."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError("Model artifact not found: {}".format(artifact_path))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        artifact = torch.load(artifact_path, map_location=device, weights_only=True)
    except TypeError:  # Supports PyTorch releases before ``weights_only`` existed.
        artifact = torch.load(artifact_path, map_location=device)
    required = {
        "model_state_dict", "scaler_state", "feature_columns", "sequence_length",
        "upper_signal_threshold", "lower_signal_threshold", "model_metadata",
        "training_metadata",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError("Invalid model artifact; missing keys: {}".format(sorted(missing)))

    metadata = artifact["model_metadata"]
    model = LSTMClassifier(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"],
    ).to(device)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return LoadedModelArtifact(
        model=model,
        scaler=_deserialize_scaler(artifact["scaler_state"]),
        feature_columns=tuple(artifact["feature_columns"]),
        sequence_length=artifact["sequence_length"],
        upper_signal_threshold=artifact["upper_signal_threshold"],
        lower_signal_threshold=artifact["lower_signal_threshold"],
        model_metadata=metadata,
        training_metadata=artifact["training_metadata"],
    )
