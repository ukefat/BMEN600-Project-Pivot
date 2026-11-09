import json

from temporal_ssl.cli import load_config
from temporal_ssl.study import plan_runs, run_study, validation_trend


def test_paired_plan_and_validation_only_budget_selection(tmp_path, monkeypatch):
    cfg = load_config("configs/research.json")
    cfg["study"].update(pilot_epochs=4, maximum_epochs=8, window=2)
    cfg["study"]["seeds"] = [{"data": 7, "model": 11}]
    matrix = [{"name": objective, "overrides": {"model.objective": objective}} for objective in ["causal", "masked"]]
    planned = plan_runs(cfg, matrix)
    assert planned[0][2]["data"]["seed"] == planned[1][2]["data"]["seed"]
    assert planned[0][2]["training"]["seed"] == planned[1][2]["training"]["seed"]
    calls = []

    def fake_run(config, path, resume=False, evaluate=True, allow_epoch_extension=False):
        calls.append((config["training"]["epochs"], evaluate))
        path.mkdir(exist_ok=True)
        (path/"config.json").write_text(json.dumps(config))
        if evaluate:
            (path/"metrics.json").write_text("{}")
        return [{"epoch": i+1, "validation_loss": loss} for i, loss in enumerate([1.0, 0.9, 0.8, 0.7])]

    monkeypatch.setattr("temporal_ssl.study.run_experiment", fake_run)
    monkeypatch.setattr("temporal_ssl.study.report_study", lambda root: None)
    run_study(cfg, matrix, tmp_path)
    assert calls == [(4, False)]*2 + [(8, False)]*2 + [(8, True)]*2
    decision = json.loads((tmp_path/"budget_decision.json").read_text())
    assert decision["epochs"] == 8
    assert validation_trend([{ "epoch": i+1, "validation_loss": 1.0} for i in range(4)], 2)["relative_improvement"] == 0
