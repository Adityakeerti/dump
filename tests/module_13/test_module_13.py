"""Tests for Module 13: Static visual artifacts and HTML summary."""

from pathlib import Path
import pytest

from citysense.config.loader import load_citysense_config
from citysense.counterfactual.types import (
    CounterfactualLadderResult,
    CounterfactualRungResult,
)
from citysense.diagnosis.capacity import CapacityFitResult
from citysense.diagnosis.shockwave import ShockwaveOrigin
from citysense.diagnosis.types import JamCluster
from citysense.observation.types import ObservationRecord
from citysense.situations.builder import build_situation
from citysense.report.generator import ReportGenerator
from citysense.rl.eval import PolicyEvaluationResult


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


def test_module_13_report_generator_artifacts(full_config, tmp_path):
    cfg, cfg_hash = full_config
    diag_cfg = cfg.diagnosis

    # Synthetic observations
    obs = [
        ObservationRecord(
            cell_id=c,
            timestamp_sec=float(t),
            speed_mps=3.5 if c == 15 and 300 <= t <= 400 else 13.89,
            flow_vps=0.3,
            density_vpm=0.08,
            source="SIMULATED",
            simulated=True,
        )
        for c in range(10, 20)
        for t in range(250, 450, 10)
    ]

    cluster = JamCluster(
        cluster_id=0,
        cell_ids=[13, 14, 15],
        timestamps_sec=[300.0, 350.0, 400.0],
        tail_cell=13,
        head_cell=15,
        start_time_sec=300.0,
        end_time_sec=400.0,
        extent_m=60.0,
        duration_sec=100.0,
        mean_speed_mps=3.5,
        recurring=True,
        matched_days=[1, 2, 3],
    )

    trail = [
        {
            "cell_id": 15,
            "timestamp_sec": 300.0,
            "mean_speed_mps": 3.5,
            "mean_flow_vps": 0.2,
            "is_discharge_origin": True,
            "description": diag_cfg.action_template_capacity,
        }
    ]

    shockwave = ShockwaveOrigin(
        origin_cell=15,
        estimated_wave_speed_mps=5.56,
        earliest_time_sec=300.0,
        evidence_trail=trail,
        jam_cluster_id=0,
    )

    cap_fit = CapacityFitResult(
        cell_id=15,
        time_window_sec=(300.0, 350.0),
        nominal_capacity=0.5,
        estimated_capacity=0.2,
        capacity_ratio=0.4,
        confidence_interval=(0.35, 0.45),
        change_points=[300.0],
        significant_drop_detected=True,
    )

    rungs = [
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_capacity,
            mean_recovery_vhr=0.45,
            ci_lower_vhr=0.40,
            ci_upper_vhr=0.50,
            replicate_recoveries=[0.45],
            rank=1,
        ),
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_demand,
            mean_recovery_vhr=0.10,
            ci_lower_vhr=0.08,
            ci_upper_vhr=0.12,
            replicate_recoveries=[0.10],
            rank=2,
        ),
        CounterfactualRungResult(
            rung_name=diag_cfg.top_ranked_name_signal,
            mean_recovery_vhr=0.02,
            ci_lower_vhr=0.01,
            ci_upper_vhr=0.03,
            replicate_recoveries=[0.02],
            rank=3,
        ),
    ]

    ladder = CounterfactualLadderResult(
        scenario_name="recurring_capacity_drop",
        rungs=rungs,
        top_ranked_rung=diag_cfg.top_ranked_name_capacity,
        baseline_total_vhr=0.80,
        noise_floor_vhr=diag_cfg.counterfactual_noise_floor_vhr,
        explains_jam=True,
    )

    situation = build_situation(
        situation_id="SIT-REPORT-001",
        corridor_name="Arterial Corridor Alpha",
        observations=obs,
        jam_clusters=[cluster],
        shockwave_origin=shockwave,
        capacity_fit=cap_fit,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash=cfg_hash,
    )

    report_gen = ReportGenerator(diag_cfg)

    heatmap_path = tmp_path / diag_cfg.heatmap_png_filename
    ranking_path = tmp_path / diag_cfg.ladder_png_filename
    html_path = tmp_path / diag_cfg.report_html_filename

    # 1. Generate speed heatmap
    out_heat = report_gen.generate_speed_heatmap(
        observations=obs,
        situation=situation,
        origin_cell=situation.ranked_hypotheses[0].target_cell,
        output_path=heatmap_path,
    )
    assert out_heat.exists(), f"Heatmap file does not exist at {heatmap_path}"
    assert out_heat.stat().st_size > 0

    # 2. Generate ranking bar chart
    out_rank = report_gen.generate_ranking_barchart(
        situation=situation,
        output_path=ranking_path,
    )
    assert out_rank.exists(), f"Ranking barchart does not exist at {ranking_path}"
    assert out_rank.stat().st_size > 0

    # 3. Generate HTML report
    policy_results = {
        "DoubleDQN": PolicyEvaluationResult(
            policy_name="DoubleDQN",
            mean_queue=1.5,
            ci_lower=1.2,
            ci_upper=1.8,
            episodes=[1.5, 1.4, 1.6],
        ),
        "FixedTime": PolicyEvaluationResult(
            policy_name="FixedTime",
            mean_queue=3.5,
            ci_lower=3.1,
            ci_upper=3.9,
            episodes=[3.5, 3.4, 3.6],
        ),
    }

    out_html = report_gen.generate_html_report(
        situation=situation,
        heatmap_png_path=heatmap_path,
        ranking_png_path=ranking_path,
        policy_eval_results=policy_results,
        output_path=html_path,
    )
    assert out_html.exists(), f"HTML report does not exist at {html_path}"
    assert out_html.stat().st_size > 0

    html_text = out_html.read_text(encoding="utf-8")

    # Verify banner text appears verbatim
    assert diag_cfg.simulated_banner_text in html_text, "Simulated banner text missing in HTML"
    assert "ALL TWIN FIGURES AND OUTPUTS ARE SIMULATED" in html_text

    # Verify situation summary content
    assert situation.situation_id in html_text
    assert situation.corridor_name in html_text
    assert situation.config_hash in html_text

    # Verify hypotheses table content
    for h in situation.ranked_hypotheses:
        assert h.hypothesis_type in html_text
        assert h.recommended_action in html_text
        assert h.responsible_party in html_text

    # Verify policy evaluation table content
    assert "DoubleDQN" in html_text
    assert "FixedTime" in html_text
    assert "Mean Queue (veh)" in html_text


def test_module_13_report_generator_without_origin_and_policy(full_config, tmp_path):
    cfg, cfg_hash = full_config
    diag_cfg = cfg.diagnosis

    obs = [
        ObservationRecord(
            cell_id=c,
            timestamp_sec=float(t),
            speed_mps=13.89,
            flow_vps=0.3,
            density_vpm=0.08,
            source="SIMULATED",
            simulated=True,
        )
        for c in range(10, 15)
        for t in range(250, 300, 10)
    ]

    ladder = CounterfactualLadderResult(
        scenario_name="control",
        rungs=[],
        top_ranked_rung="NONE",
        baseline_total_vhr=0.0,
        noise_floor_vhr=diag_cfg.counterfactual_noise_floor_vhr,
        explains_jam=False,
    )

    situation = build_situation(
        situation_id="SIT-REPORT-002",
        corridor_name="Arterial Corridor Beta",
        observations=obs,
        jam_clusters=[],
        shockwave_origin=None,
        capacity_fit=None,
        ladder_result=ladder,
        diag_cfg=diag_cfg,
        config_hash=cfg_hash,
    )

    report_gen = ReportGenerator(diag_cfg)

    heatmap_path = tmp_path / "speed_heatmap_no_origin.png"
    ranking_path = tmp_path / "ranking_empty.png"
    html_path = tmp_path / "report_empty.html"

    out_heat = report_gen.generate_speed_heatmap(
        observations=obs,
        situation=situation,
        origin_cell=None,
        output_path=heatmap_path,
    )
    assert out_heat.exists()
    assert out_heat.stat().st_size > 0

    out_rank = report_gen.generate_ranking_barchart(
        situation=situation,
        output_path=ranking_path,
    )
    assert out_rank.exists()
    assert out_rank.stat().st_size > 0

    out_html = report_gen.generate_html_report(
        situation=situation,
        heatmap_png_path=heatmap_path,
        ranking_png_path=ranking_path,
        policy_eval_results=None,
        output_path=html_path,
    )
    assert out_html.exists()
    assert out_html.stat().st_size > 0
    html_text = out_html.read_text(encoding="utf-8")
    assert diag_cfg.simulated_banner_text in html_text
    assert situation.situation_id in html_text
