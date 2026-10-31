"""Gate tests for Module 9: Reinforcement Learning Signal Control."""

from pathlib import Path
import numpy as np
import pytest
import torch

from citysense.config.loader import (
    compute_config_hash,
    load_citysense_config,
    load_demand_config,
    load_engine_config,
    load_network_config,
    load_rl_config,
)
from citysense.engine.ctm import NativeCTMEngine
from citysense.network.corridor import build_corridor_network
from citysense.network.demand import CorridorDemand
from citysense.rl.dqn import DoubleDQNAgent
from citysense.rl.env import CorridorSignalEnv
from citysense.rl.eval import evaluate_policy_suite
from citysense.rl.shield import SignalSafetyShield


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


def setup_env():
    eng_cfg = load_engine_config(ENGINE_PATH)
    net_cfg = load_network_config(NETWORK_PATH)
    dem_cfg = load_demand_config(DEMAND_PATH)
    rl_cfg = load_rl_config(RL_PATH)

    net = build_corridor_network(net_cfg, seed=rl_cfg.seed)
    demand = CorridorDemand(dem_cfg)
    engine = NativeCTMEngine(
        config=eng_cfg,
        n_cells=net.n_cells,
        upstream_cells=net.upstream_cells,
        downstream_cells=net.downstream_cells,
    )
    env = CorridorSignalEnv(
        engine=engine,
        network=net,
        demand=demand,
        rl_config=rl_cfg,
        engine_config=eng_cfg,
    )
    return env, net_cfg, rl_cfg, eng_cfg


def test_observation_and_action_shapes_match_config():
    """Verify observation and action dimensions match configuration."""
    env, net_cfg, rl_cfg, _ = setup_env()
    obs = env.reset(seed=rl_cfg.seed)

    expected_dim = (
        2 * rl_cfg.state_feature_dim_per_approach
        + rl_cfg.state_extra_dim
        + 2
    )
    assert obs.shape == (net_cfg.num_signalized_junctions, expected_dim)
    assert env.action_space_size == rl_cfg.action_space_size


def test_safety_shield_property_invariant():
    """Property test: safety shield is never violated over 1000 randomized actions."""
    rl_cfg = load_rl_config(RL_PATH)
    eng_cfg = load_engine_config(ENGINE_PATH)
    num_j = 4
    shield = SignalSafetyShield(rl_cfg, num_j)
    shield.reset()

    rng = np.random.default_rng(rl_cfg.seed)
    min_green = rl_cfg.shield_min_green_sec
    max_cycle = rl_cfg.shield_max_cycle_sec
    dt = eng_cfg.dt

    for _ in range(1000):
        for j in range(num_j):
            time_before = shield.states[j].time_in_phase
            phase_before = shield.states[j].current_phase

            proposed_action = int(rng.choice([rl_cfg.action_keep, rl_cfg.action_switch]))
            effective_phase, _ = shield.step(j, proposed_action, dt)

            # Invariant 1: If time was < min_green, phase MUST NOT have switched
            if time_before < min_green:
                assert effective_phase == phase_before, (
                    f"Junction {j} switched before min_green! Time was {time_before:.1f}s"
                )

            # Invariant 2: If time exceeded max_cycle, phase MUST switch
            if time_before >= max_cycle:
                assert effective_phase != phase_before, (
                    f"Junction {j} failed to switch at max_cycle! Time was {time_before:.1f}s"
                )


def test_smoke_train_checkpoint_save_and_reload(tmp_path: Path):
    """Verify tiny smoke training completes, saves checkpoint, and reloads reproducing exact actions."""
    env, _, rl_cfg, _ = setup_env()
    agent = DoubleDQNAgent(rl_cfg, env.obs_dim_per_junction)

    obs = env.reset(seed=rl_cfg.seed)
    smoke_steps = rl_cfg.smoke_train_steps

    # Collect experience and train
    for step in range(smoke_steps):
        actions = [agent.select_action(obs[j], epsilon=0.5) for j in range(env.num_junctions)]
        next_obs, reward, done, _ = env.step(actions)
        for j in range(env.num_junctions):
            agent.buffer.push(obs[j], actions[j], reward, next_obs[j], done)
        agent.update()
        obs = next_obs

    # Compute full config hash
    _, cfg_hash = load_citysense_config(
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

    ckpt_file = tmp_path / rl_cfg.checkpoint_filename
    saved_path = agent.save_checkpoint(ckpt_file, cfg_hash, rl_cfg.seed, {"loss": 0.01})
    assert saved_path.exists()

    # Create reloaded agent
    reloaded_agent = DoubleDQNAgent(rl_cfg, env.obs_dim_per_junction)
    metadata = reloaded_agent.load_checkpoint(saved_path)
    assert metadata["config_hash"] == cfg_hash
    assert metadata["seed"] == rl_cfg.seed

    # Test identical action reproduction on 10 deterministic states
    test_states = np.random.default_rng(999).normal(0.0, 1.0, size=(10, env.obs_dim_per_junction)).astype(np.float32)
    for s in test_states:
        a_orig = agent.select_action(s, epsilon=0.0)
        a_reload = reloaded_agent.select_action(s, epsilon=0.0)
        assert a_orig == a_reload


def test_comparative_evaluation_report():
    """Verify evaluation suite runs and outputs measured numbers across RL and all baselines."""
    env, net_cfg, rl_cfg, _ = setup_env()
    agent = DoubleDQNAgent(rl_cfg, env.obs_dim_per_junction)

    # Fast evaluation
    eval_episodes = 2
    eval_steps = 25
    results = evaluate_policy_suite(
        env=env,
        rl_agent=agent,
        net_config=net_cfg,
        rl_config=rl_cfg,
        eval_episodes=eval_episodes,
        eval_steps=eval_steps,
        seed=rl_cfg.seed,
    )

    required_policies = ["DoubleDQN", "fixed-time", "longest-queue", "max-pressure"]
    for pol in required_policies:
        assert pol in results
        res = results[pol]
        assert res.mean_queue >= 0.0
        assert res.ci_lower <= res.mean_queue <= res.ci_upper
        # Print measured numbers per Hard Rule 8
        print(f"Policy {pol:15s}: mean_queue={res.mean_queue:.2f} (95% CI: [{res.ci_lower:.2f}, {res.ci_upper:.2f}])")
