"""Loader for sensor observation series and external traffic data."""

import csv
from pathlib import Path
from citysense.config.schemas import ObservationConfig
from citysense.observation.types import ObservationRecord


def load_speed_series(file_path: str | Path, config: ObservationConfig) -> list[ObservationRecord]:
    """Loads tabular CSV observations using column mappings from config."""
    p = Path(file_path)
    records: list[ObservationRecord] = []

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = int(row[config.cell_column_name])
            ts = float(row[config.time_column_name])
            speed = float(row[config.speed_column_name])

            flow_val = 0.0
            if config.flow_column_name in row and row[config.flow_column_name]:
                flow_val = float(row[config.flow_column_name])

            dens_val = 0.0
            if config.density_column_name in row and row[config.density_column_name]:
                dens_val = float(row[config.density_column_name])

            src_val = row.get(config.source_column_name, config.default_simulated_source)
            sim_str = row.get(config.simulated_column_name, str(config.simulated_flag)).lower()
            is_sim = sim_str == "true" or sim_str == "1"

            records.append(
                ObservationRecord(
                    cell_id=cell_id,
                    timestamp_sec=ts,
                    speed_mps=speed,
                    flow_vps=flow_val,
                    density_vpm=dens_val,
                    source=src_val,
                    simulated=is_sim,
                )
            )

    return records
