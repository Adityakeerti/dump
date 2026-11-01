"""Run the live digital twin: `uv run python scripts/serve_twin.py` (or the repo's .venv python).

Environment overrides: TWIN_HOST, TWIN_PORT. Paths are relative to the repo root.
"""

import os
import sys
from pathlib import Path

import torch
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from citysense.config.loader import load_citysense_config  # noqa: E402
from citysense.live import create_live_app, load_live_config  # noqa: E402

CONFIG = ROOT / "config"


def build_app():
    cfg, cfg_hash = load_citysense_config(
        project_path=CONFIG / "project.yaml",
        engine_path=CONFIG / "engine.yaml",
        network_path=CONFIG / "network.yaml",
        demand_path=CONFIG / "demand.yaml",
        scenarios_dir=CONFIG / "scenarios",
        observation_path=CONFIG / "observation.yaml",
        diagnosis_path=CONFIG / "diagnosis.yaml",
        rl_path=CONFIG / "rl.yaml",
        api_path=CONFIG / "api.yaml",
    )
    live = load_live_config(CONFIG / "live.yaml")
    return create_live_app(cfg, cfg_hash, live), live


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TWIN_TORCH_THREADS", "1")))  # tiny MLPs: 1 thread is fastest
    app, live = build_app()
    uvicorn.run(
        app,
        host=os.environ.get("TWIN_HOST", live.host),
        port=int(os.environ.get("TWIN_PORT", live.port)),
        log_level="warning",
    )
