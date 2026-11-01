"""Pretrain the live twin's Double DQN headlessly (same network, demand, scenarios and exploration schedule).

    .venv/bin/python scripts/pretrain_live_dqn.py --episodes 12

Stop the live twin service first: both write the checkpoint in config/live.yaml (checkpoint_path, episodes_path).
Episodes rotate through all scenarios so the policy is not tuned to one disruption.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from citysense.config.loader import load_citysense_config  # noqa: E402
from citysense.live import LiveTwin, load_live_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(1)
    c = ROOT / "config"
    cfg, h = load_citysense_config(c / "project.yaml", c / "engine.yaml", c / "network.yaml", c / "demand.yaml", c / "scenarios",
                                   c / "observation.yaml", c / "diagnosis.yaml", c / "rl.yaml", c / "api.yaml")
    live = load_live_config(c / "live.yaml").model_copy(update={"autoplay": False})
    twin = LiveTwin(cfg, h, live)
    scenarios = sorted(cfg.scenarios)
    steps = int(cfg.demand.simulation_duration_sec / cfg.engine.dt)
    start = twin.episodes_done
    for k in range(args.episodes):
        twin.control(None, None, scenarios[(start + k) % len(scenarios)], None, True)
        t0 = time.time()
        twin.advance(steps)  # finishes the episode: metrics recorded, checkpoint saved
        ep = twin.episodes[-1]
        means = {n: round(v["mean_system_vehicles"], 1) for n, v in ep["policies"].items()}
        print(f"episode {ep['episode'] + 1} {ep['scenario']:24s} eps {ep['epsilon']:.2f} {time.time() - t0:5.1f}s  {means}", flush=True)


if __name__ == "__main__":
    main()
