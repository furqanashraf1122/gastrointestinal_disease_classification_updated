#!/usr/bin/env python3
"""Run independent training seeds on one fixed data manifest and summarize results."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_metric_files(run_dirs, filename, sections):
    per_seed = {}
    for run_dir in run_dirs:
        data = load_json(run_dir / filename)
        seed = str(data["run_seed"])
        per_seed[seed] = data
    summary = {}
    for section in sections:
        metric_names = sorted({
            key for d in per_seed.values()
            for key, value in d.get(section, {}).items()
            if isinstance(value, (int, float))
        })
        summary[section] = {}
        for metric in metric_names:
            values = [d[section][metric] for d in per_seed.values()
                      if metric in d.get(section, {})
                      and isinstance(d[section][metric], (int, float))]
            summary[section][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_runs": len(values),
            }
    return per_seed, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Dataset class-folder root")
    p.add_argument("--manifest", required=True, help="One fixed train/val/test manifest")
    p.add_argument("--config", default="config.json")
    p.add_argument("--out", required=True, help="Output root for all seeds")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--paper-weights", action="store_true",
                   help="Use reported manuscript weights rather than reselecting on validation")
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        p.error("Provide at least two unique seeds")
    root, manifest, config, out = map(Path, (args.root, args.manifest, args.config, args.out))
    if not manifest.is_file():
        p.error(f"Manifest not found: {manifest}")
    if not config.is_file():
        p.error(f"Configuration not found: {config}")
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    script = Path(__file__).with_name("gi_ensemble.py")
    for seed in seeds:
        base = out / f"seed_{seed}"
        models, features, results = base / "models", base / "features", base / "results"
        commands = [
            [sys.executable, str(script), "train-all", "--root", str(root), "--manifest", str(manifest),
             "--out", str(models), "--config", str(config), "--run-seed", str(seed)],
            [sys.executable, str(script), "extract-all", "--root", str(root), "--manifest", str(manifest),
             "--models", str(models), "--out", str(features), "--config", str(config), "--run-seed", str(seed)],
            [sys.executable, str(script), "ensemble", "--features", str(features), "--out", str(results),
             "--config", str(config), "--run-seed", str(seed)],
        ]
        if args.pretrained:
            commands[0].append("--pretrained")
        if args.paper_weights:
            commands[2].append("--paper-weights")
        for command in commands:
            print("RUN:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
        run_dirs.append(results)

    # A0 is optional; summarize only metrics actually present in every run.
    ablations, ablation_summary = aggregate_metric_files(
        run_dirs, "ablation_metrics.json", ["A0", "A1", "A2", "A3", "A4", "A5"])
    table10_runs = {str(load_json(d / "table10_fusion_metrics.json")["run_seed"]):
                    load_json(d / "table10_fusion_metrics.json") for d in run_dirs}
    table10_summary = {}
    for strategy in ("feature_averaging", "feature_concatenation"):
        table10_summary[strategy] = {}
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            values = [d[strategy][metric] for d in table10_runs.values()]
            table10_summary[strategy][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "n_runs": len(values),
            }
    summary = {
        "protocol": "fixed manifest; independent model/projection/ensemble seeds",
        "seeds": seeds,
        "manifest_sha256": sha256(manifest),
        "config_sha256": sha256(config),
        "ablation_summary": ablation_summary,
        "table10_summary": table10_summary,
        "per_seed_ablation_results": ablations,
        "per_seed_table10_results": table10_runs,
        "note": "Sample standard deviation uses ddof=1. This script does not reconstruct missing historical results.",
    }
    (out / "repeated_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out / "repeated_run_summary.json"),
                      "seeds": seeds, "manifest_sha256": summary["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
