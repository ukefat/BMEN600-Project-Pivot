"""ds004504 adapter using the course project's copied, validated preprocessing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .data import Normalizer
from .eeg_course.download import sha256
from .eeg_course.features import extract
from .eeg_course.shared import (CHANNELS, iter_epochs, original_sample_indices,
                               participant_metadata, prepare, read_recording, select_intervals)
from .experiment import save_json

PREPROCESS = {"epoch_seconds": 8.0, "sampling_hz": None, "reference": "preserve",
              "normalization": "none", "welch_seconds": 2.0, "power_floor_uv2": 1e-12,
              "flat_ptp_uv": 1e-6, "amplitude_warning_uv": 1000.0, "source": "derivatives"}


def prepare_cache(data_dir, cache_dir, max_epochs=24):
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    root, cache = Path(data_dir), Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((root/"download_manifest.json").read_text())
    expected = {r["path"]: r for r in provenance["files"]}
    for relative, record in expected.items():
        if not relative.endswith(".set") and (root/relative).exists():
            if sha256(root/relative) != record["sha256"]:
                raise ValueError(f"Changed dataset metadata: {relative}")
    people, definitions = participant_metadata(root)
    specification = {"preprocess": PREPROCESS, "max_epochs": max_epochs,
                     "adapter_sha256": sha256(Path(__file__)),
                     "selection": "evenly spaced usable epoch indices, including endpoints",
                     "dataset": provenance["dataset"], "version": provenance["version"],
                     "git_commit": provenance["git_commit"],
                     "code": {p.name: sha256(p) for p in (Path(__file__).parent/"eeg_course").glob("*.py")}}
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    rows = []
    for person in people.to_dict("records"):
        pid = person["participant_id"]
        relative = f"derivatives/{pid}/eeg/{pid}_task-eyesclosed_eeg.set"
        path = root/relative
        digest = sha256(path)
        if digest != expected[relative]["sha256"]:
            raise ValueError(f"Changed recording: {relative}")
        filename = cache/f"{pid}-{identity[:12]}-{digest[:12]}.npz"
        if not filename.exists():
            eeg, x, fs, _, _ = read_recording(path, root/pid/"eeg"/f"{pid}_task-eyesclosed_channels.tsv",
                                               root/pid/"eeg"/f"{pid}_task-eyesclosed_eeg.json")
            original, _, removed = original_sample_indices(eeg, x.shape[1], "derivatives")
            x = prepare(x, fs, PREPROCESS)
            intervals = select_intervals(x, fs, original, PREPROCESS)
            usable = intervals.loc[intervals.usable]
            if not len(usable):
                rows.append({**person, "excluded": True, "reason": "no usable contiguous epochs"})
                continue
            keep = np.linspace(0, len(usable)-1, min(max_epochs, len(usable)), dtype=int)
            chosen = usable.iloc[keep]
            epochs, spectral = [], []
            for _, epoch, target_fs in iter_epochs(x, fs, chosen, PREPROCESS):
                epochs.append(epoch.T.astype(np.float32))
                spectral.append(extract(epoch[:, :epoch.shape[1]//2], target_fs, PREPROCESS)[0])
            np.savez(filename, x=np.stack(epochs), spectral=np.stack(spectral), sampling_hz=fs,
                     source_start=chosen.source_start_sample.to_numpy(),
                     original_start=chosen.original_start_seconds.to_numpy(), usable_epochs=len(usable),
                     rejected_epochs=int((~intervals.usable).sum()), removed_seconds=removed/fs,
                     history=str(eeg.get("history", "")), identity=identity)
        with np.load(filename) as stored:
            if str(stored["identity"]) != identity:
                raise ValueError("EEG cache identity mismatch")
            rows.append({**person, "excluded": False, "cache": str(filename.resolve()),
                         "epochs": len(stored["x"]), "usable_epochs": int(stored["usable_epochs"]),
                         "sampling_hz": float(stored["sampling_hz"]), "samples": stored["x"].shape[1],
                         "rejected_epochs": int(stored["rejected_epochs"]), "source_sha256": digest})
        print(f"Prepared {pid}: {rows[-1]['epochs']} / {rows[-1]['usable_epochs']} usable epochs", flush=True)
    manifest = {"specification": specification, "identity": identity, "channels": CHANNELS,
                "participants": rows, "definitions": definitions}
    save_json(cache/"manifest.json", manifest)
    return manifest


def subject_splits(subjects, seed=600):
    """60/20/20 participant split, stratified by diagnosis; never split epoch rows."""
    people = pd.DataFrame(subjects).sort_values("participant_id").reset_index(drop=True)
    if not people.participant_id.is_unique:
        raise ValueError("Participant IDs must be unique")
    remaining, test = train_test_split(people, test_size=0.2, random_state=seed, stratify=people.group)
    train, validation = train_test_split(remaining, test_size=0.25, random_state=seed+1, stratify=remaining.group)
    return {"train": train.sort_values("participant_id").to_dict("records"),
            "validation": validation.sort_values("participant_id").to_dict("records"),
            "test": test.sort_values("participant_id").to_dict("records")}


class EEGDataset(Dataset):
    """Independent participant epochs, optional synthetic fields, fixed [T,C] tensors."""
    def __init__(self, subjects, normalizer=None, epoch_limit=None):
        self.normalizer, self.subjects = normalizer, subjects
        self.x, self.spectral, self.metadata = [], [], []
        for person in subjects:
            with np.load(person["cache"]) as stored:
                count = len(stored["x"]) if epoch_limit is None else min(epoch_limit, len(stored["x"]))
                self.x.extend(stored["x"][:count])
                self.spectral.extend(stored["spectral"][:count])
                for i in range(count):
                    self.metadata.append({"dataset": "ds004504", "participant_id": person["participant_id"],
                                          "group": person["group"], "epoch_index": i,
                                          "source_start_sample": int(stored["source_start"][i])})
        self.x = np.stack(self.x)
        self.spectral = np.stack(self.spectral)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index] if self.normalizer is None else self.normalizer.apply(self.x[index])
        return {"x": torch.from_numpy(x), "latents": None, "parameters": None,
                "metadata": self.metadata[index]}


def fit_normalizer(dataset):
    # Streaming over epochs avoids concatenating another copy of the complete dataset.
    count, total, squares = 0, np.zeros(dataset.x.shape[-1]), np.zeros(dataset.x.shape[-1])
    for epoch in dataset.x:
        x = epoch.astype(np.float64)
        count += len(x)
        total += x.sum(0)
        squares += (x*x).sum(0)
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 1e-12))
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))


def aggregate_participants(features, metadata):
    ids = sorted({r["participant_id"] for r in metadata})
    vectors, labels = [], []
    for pid in ids:
        indices = [i for i, r in enumerate(metadata) if r["participant_id"] == pid]
        groups = {metadata[i]["group"] for i in indices}
        if len(groups) != 1:
            raise ValueError("Conflicting labels for a participant")
        vectors.append(np.mean(features[indices], axis=0))
        labels.append(groups.pop())
    return ids, np.stack(vectors), np.array(labels)
