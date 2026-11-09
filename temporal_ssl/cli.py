"""JSON configurations and a deliberately small explicit (non-Cartesian) sweep matrix."""
import argparse
import copy
import json
from pathlib import Path

from .data import make_splits, save_dataset
from .experiment import run_experiment, save_json
from .plots import analyze, plot_trajectory


def load_config(path):
    return json.loads(Path(path).read_text())


def override(cfg, changes):
    cfg = copy.deepcopy(cfg)
    for key, value in changes.items():
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "generate", "sweep"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/default.json")
        p.add_argument("--output", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                       help='Dotted override, e.g. --set model.objective=\'"masked"\'')
        if command in {"run", "sweep"}:
            p.add_argument("--resume", action="store_true")
        if command == "sweep":
            p.add_argument("--matrix", default="configs/matrix.json")
    p = sub.add_parser("analyze")
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        print(analyze(args.root, args.output))
        return
    cfg = load_config(args.config)
    changes = {}
    for item in args.set:
        key, value = item.split("=", 1)
        changes[key] = json.loads(value)
    cfg = override(cfg, changes)
    if args.command == "run":
        run_experiment(cfg, args.output, args.resume)
    elif args.command == "generate":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        save_json(output/"config.json", cfg)
        splits, normalizer = make_splits(cfg["data"])
        if normalizer:
            import numpy as np
            np.savez(output/"normalization.npz", mean=normalizer.mean, std=normalizer.std)
        save_json(output/"manifest.json", {k: [r["metadata"] for r in v.records] for k, v in splits.items()})
        for name, dataset in splits.items():
            save_dataset(output/f"{name}.npz", dataset)
            plot_trajectory(dataset.records[0], output/f"{name}.png")
    else:
        entries = load_config(args.matrix)
        names = [entry["name"] for entry in entries]
        if len(set(names)) != len(names) or any(Path(n).name != n or n in {".", "..", ""} for n in names):
            raise ValueError("Matrix names must be unique directory names")
        for entry in entries:
            run_experiment(override(cfg, entry["overrides"]), Path(args.output)/entry["name"], args.resume)
        analyze(args.output)


if __name__ == "__main__":
    main()
