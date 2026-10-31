"""Gate tests for Module 14: End-to-End Evaluation Suite and Final Results."""

from pathlib import Path
import pytest
import yaml

from citysense.config.loader import load_citysense_config
from citysense.report.evaluation import SuiteEvaluator


CONFIG_DIR = Path("config")
PROJECT_PATH = CONFIG_DIR / "project.yaml"
ENGINE_PATH = CONFIG_DIR / "engine.yaml"
NETWORK_PATH = CONFIG_DIR / "network.yaml"
DEMAND_PATH = CONFIG_DIR / "demand.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"
OBSERVATION_PATH = CONFIG_DIR / "observation.yaml"
DIAGNOSIS_PATH = CONFIG_DIR / "diagnosis.yaml"
RL_PATH = CONFIG_DIR / "rl.yaml"
API_PATH = CONFIG_DIR / "api.yaml"
TOLERANCES_PATH = Path("tests/tolerances.yaml")


@pytest.fixture(scope="module")
def tolerances():
    with open(TOLERANCES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: v["value"] for k, v in data.items()}


@pytest.fixture(scope="module")
def full_config():
    cfg, cfg_hash = load_citysense_config(
        project_path=PROJECT_PATH,
        engine_path=ENGINE_PATH,
        network_path=NETWORK_PATH,
        demand_path=DEMAND_PATH,
        scenarios_dir=SCENARIOS_DIR,
        observation_path=OBSERVATION_PATH,
        diagnosis_path=DIAGNOSIS_PATH,
        rl_path=RL_PATH,
        api_path=API_PATH,
    )
    return cfg, cfg_hash


def test_suite_evaluator_end_to_end(full_config, tolerances, tmp_path):
    """Verify that the end-to-end evaluation suite runs across all scenarios and seeds within tolerances."""
    cfg, cfg_hash = full_config
    evaluator = SuiteEvaluator(cfg, cfg_hash)

    result = evaluator.run_suite(reports_dir=tmp_path)

    # 1. Verify all scenarios are present
    assert len(result.scenario_metrics) == len(cfg.scenarios) * cfg.diagnosis.evaluation_k_seeds
    for sc_name in cfg.scenarios.keys():
        assert sc_name in result.mean_recall
        assert sc_name in result.mean_precision
        assert sc_name in result.top_cause_accuracy

    # 2. Verify detection recall & precision meet tolerances for jam-producing scenarios
    for sc_name in ("recurring_capacity_drop", "creeping_capacity_drop", "downstream_spillback"):
        assert result.mean_recall[sc_name] >= tolerances["jam_recall_min"], (
            f"Recall {result.mean_recall[sc_name]} for {sc_name} below tolerance {tolerances['jam_recall_min']}"
        )
        assert result.mean_precision[sc_name] >= tolerances["jam_precision_min"], (
            f"Precision {result.mean_precision[sc_name]} for {sc_name} below tolerance {tolerances['jam_precision_min']}"
        )

    # Control scenario false alarm check
    assert result.mean_recall["control"] == 1.0
    assert result.mean_precision["control"] == 1.0

    # 3. Verify origin localization error tolerance for bottleneck scenarios
    for sc_name in ("recurring_capacity_drop", "creeping_capacity_drop", "downstream_spillback"):
        err = result.mean_origin_error[sc_name]
        assert err is not None
        assert err <= tolerances["origin_cell_tolerance"], (
            f"Origin cell error {err} for {sc_name} exceeds tolerance {tolerances['origin_cell_tolerance']}"
        )

    # 4. Verify capacity ratio error tolerance
    for sc_name in ("recurring_capacity_drop", "creeping_capacity_drop", "downstream_spillback"):
        cap_err = result.mean_capacity_error[sc_name]
        assert cap_err is not None
        assert cap_err <= tolerances["capacity_ratio_tolerance"], (
            f"Capacity ratio error {cap_err} for {sc_name} exceeds tolerance {tolerances['capacity_ratio_tolerance']}"
        )

    # 5. Verify 100% correct top-ranked cause rate
    for sc_name in cfg.scenarios.keys():
        assert result.top_cause_accuracy[sc_name] == 1.0, (
            f"Top cause accuracy {result.top_cause_accuracy[sc_name]} for {sc_name} below 1.0"
        )

    # 6. Verify RL policy evaluation results exist
    assert "DoubleDQN" in result.policy_results
    assert "fixed-time" in result.policy_results
    assert "longest-queue" in result.policy_results
    assert "max-pressure" in result.policy_results
    for pol_name, pres in result.policy_results.items():
        assert pres.mean_queue >= 0.0
        assert pres.ci_lower <= pres.ci_upper

    # 7. Verify markdown report is written and formatted correctly
    assert result.report_file_path.exists()
    assert result.report_file_path.stat().st_size > 0
    report_text = result.report_file_path.read_text(encoding="utf-8")
    assert cfg.diagnosis.simulated_banner_text in report_text
    assert "Executive Summary" in report_text
    assert "Benchmark Metrics by Planted Scenario" in report_text
    assert "RL vs Classical Baseline Signal Controllers" in report_text
    assert "Failure Modes, Edge Cases, and Limitations" in report_text
