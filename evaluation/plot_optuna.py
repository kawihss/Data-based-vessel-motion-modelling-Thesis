from pathlib import Path
import optuna
import optuna.visualization as vis


def save_study_plots(study, out_dir, model_key):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    varying_params = []
    for name in sorted(study.best_trial.params.keys()):
        values = {t.params.get(name) for t in completed_trials if name in t.params}
        if len(values) >= 2:
            varying_params.append(name)

    has_intermediate = any(t.intermediate_values for t in study.trials)

    plots = {
        "optimization_history": lambda: vis.plot_optimization_history(study),
        "parallel_coordinate": lambda: vis.plot_parallel_coordinate(study, params=varying_params) if varying_params else None,
        "slice": lambda: vis.plot_slice(study, params=varying_params) if varying_params else None,
        "contour": lambda: vis.plot_contour(study, params=varying_params[:4]) if len(varying_params) >= 2 else None,
        "param_importances": lambda: vis.plot_param_importances(study),
        "intermediate_values": lambda: vis.plot_intermediate_values(study) if has_intermediate else None,
    }

    for name, make_fig in plots.items():
        try:
            fig = make_fig()
            if fig is None:
                print(f"Skipped {name} for {model_key}: not enough informative data")
                continue
            path = out_dir / f"{model_key}_{name}.html"
            fig.write_html(str(path))
            print(f"Saved optuna plot: {path.name}")
        except Exception as e:
            print(f"Skipped {name} for {model_key}: {e}")
