from pathlib import Path
import random

import yaml


def load_runtime_config(project_root):
    config_path = Path(project_root) / "configs" / "evaluation.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_run_paths(project_root, cfg, create=False):
    project_root = Path(project_root)
    baseline_root_rel = cfg["paths"]["baseline_root"]
    baseline_root = project_root / baseline_root_rel
    runs_root = baseline_root / "runs"
    run_name = str(cfg["run"]["name"]).strip()
    run_dir = runs_root / run_name

    paths = {
        "baseline_root": baseline_root,
        "runs_root": runs_root,
        "run_dir": run_dir,
        "tuning_dir": run_dir / "tuning",
        "diagnostics_dir": run_dir / "diagnostics",
        "model_output_dir": run_dir / "model_output",
        "plots_dir": run_dir / "plots",
        "latest_run_file": baseline_root / "latest_run.txt",
    }

    if create:
        for p in [
            paths["baseline_root"],
            paths["runs_root"],
            paths["run_dir"],
            paths["tuning_dir"],
            paths["diagnostics_dir"],
            paths["model_output_dir"],
            paths["plots_dir"],
        ]:
            p.mkdir(parents=True, exist_ok=True)

    return paths


def update_latest_run_pointer(paths):
    latest_run_file = Path(paths["latest_run_file"])
    latest_run_file.parent.mkdir(parents=True, exist_ok=True)
    latest_run_file.write_text(Path(paths["run_dir"]).name + "\n", encoding="utf-8")


def write_run_metadata(paths, cfg, stage):
    run_dir = Path(paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.yaml"

    metadata = {
        "run": {
            "name": cfg["run"]["name"],
            "description": cfg["run"]["description"],
            "seed": int(cfg["run"]["seed"]),
            "sample_pct": int(cfg["run"]["sample_pct"]),
        },
        "data": {
            "split": cfg["data"]["split"],
            "context_filter_tuning": cfg["data"]["context_filter_tuning"],
            "context_filter_evaluation": cfg["data"]["context_filter_evaluation"],
        },
        "sampling": {
            "tuning_validation_pct": int(get_sampling_value(cfg, "tuning_validation_pct")),
            "evaluation_pct": int(get_sampling_value(cfg, "evaluation_pct")),
        },
        "stage": stage,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def get_newest_run_dir(baseline_root):
    runs_root = Path(baseline_root) / "runs"
    if not runs_root.exists():
        return None

    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def get_sampling_value(cfg, key):
    return int(cfg["sampling"][key])


def subsample_items(items, sample_pct, seed):
    sample_pct = int(sample_pct)
    if sample_pct >= 100:
        return list(items)

    items = list(items)
    if not items:
        return []

    n_total = len(items)
    n_keep = max(1, int(round((sample_pct / 100.0) * n_total)))
    rng = random.Random(int(seed))
    selected_indices = sorted(rng.sample(range(n_total), n_keep))
    return [items[i] for i in selected_indices]
