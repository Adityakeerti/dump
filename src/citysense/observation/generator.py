"""Sensor observation generator applying sampling coverage, noise, and dropouts."""

import csv
from pathlib import Path
from typing import Any
import numpy as np

from citysense.config.schemas import ObservationConfig
from citysense.engine.types import SimulationState
from citysense.observation.types import ObservationRecord


class ObservationGenerator:
    """Generates realistic sensor observations from simulation state."""

    def __init__(self, config: ObservationConfig, n_cells: int, seed: int) -> None:
        self.config = config
        self.n_cells = n_cells
        self.rng = np.random.default_rng(seed)

        # Select monitored cells based on coverage fraction
        num_monitored = int(round(config.coverage_fraction * n_cells))
        if num_monitored <= 0:
            num_monitored = 1
        self.monitored_cells = sorted(
            self.rng.choice(n_cells, size=num_monitored, replace=False).tolist()
        )
        self.records: list[ObservationRecord] = []

    def observe_step(self, state: SimulationState, dt: float) -> list[ObservationRecord]:
        """Optionally generates observations if current time matches reporting interval."""
        reporting_interval = self.config.reporting_interval_sec
        steps_per_interval = max(1, int(round(reporting_interval / dt)))

        if state.step_count % steps_per_interval != 0:
            return []

        step_records: list[ObservationRecord] = []
        for cell_id in self.monitored_cells:
            # Check sensor dropout
            if self.rng.uniform(0.0, 1.0) < self.config.dropout_fraction:
                continue

            # Add sensor noise
            true_speed = float(state.speed[cell_id])
            noise = float(self.rng.normal(0.0, self.config.noise_std))
            noisy_speed = true_speed + noise
            clamped_speed = max(
                self.config.min_reported_speed_mps,
                min(self.config.max_reported_speed_mps, noisy_speed),
            )

            true_flow = float(state.flow[cell_id])
            # Previously `true_flow + noise * noise_std * 0.0` and then the exact flow was reported: flows were
            # noiseless. Relative noise; drawn only when configured so the speed-noise stream is unchanged otherwise.
            noisy_flow = true_flow
            if self.config.flow_noise_rel > 0.0:
                noisy_flow = max(0.0, true_flow * (1.0 + float(self.rng.normal(0.0, self.config.flow_noise_rel))))

            rec = ObservationRecord(
                cell_id=cell_id,
                timestamp_sec=state.time_sec,
                speed_mps=clamped_speed,
                flow_vps=noisy_flow,
                density_vpm=float(state.density[cell_id]),
                source=self.config.default_simulated_source,
                simulated=self.config.simulated_flag,
            )
            step_records.append(rec)
            self.records.append(rec)

        return step_records

    def save_to_csv(self, file_path: str | Path) -> Path:
        out_p = Path(file_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.config.cell_column_name,
                    self.config.time_column_name,
                    self.config.speed_column_name,
                    self.config.flow_column_name,
                    self.config.density_column_name,
                    self.config.source_column_name,
                    self.config.simulated_column_name,
                ]
            )
            for r in self.records:
                writer.writerow(
                    [
                        r.cell_id,
                        r.timestamp_sec,
                        r.speed_mps,
                        r.flow_vps,
                        r.density_vpm,
                        r.source,
                        str(r.simulated),
                    ]
                )
        return out_p
