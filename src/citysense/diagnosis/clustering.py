"""Congestion detection and spatio-temporal DBSCAN clustering for traffic jams."""

from typing import Any
import numpy as np
from sklearn.cluster import DBSCAN

from citysense.config.schemas import DiagnosisConfig
from citysense.diagnosis.types import CongestionPoint, JamCluster
from citysense.observation.types import ObservationRecord


class JamDetector:
    """Detects congested space-time points, clusters them into jams, and detects recurrence."""

    def __init__(self, config: DiagnosisConfig) -> None:
        self.config = config

    def identify_congested_points(
        self,
        observations: list[ObservationRecord],
        free_flow_speeds: dict[int, float],
        day: int = 0,
    ) -> list[CongestionPoint]:
        congested: list[CongestionPoint] = []
        factor = self.config.congestion_speed_factor

        for obs in observations:
            vf = free_flow_speeds.get(obs.cell_id, self.config.wave_speed_theoretical_mps)
            threshold = vf * factor
            if obs.speed_mps < threshold:
                congested.append(
                    CongestionPoint(
                        cell_id=obs.cell_id,
                        timestamp_sec=obs.timestamp_sec,
                        speed_mps=obs.speed_mps,
                        day=day,
                    )
                )
        return congested

    def cluster_jams(
        self,
        points: list[CongestionPoint],
        cell_length_m: float,
    ) -> list[JamCluster]:
        if not points or len(points) < self.config.dbscan_min_samples:
            return []

        # Feature matrix scaled by spatial and temporal scales from config
        features = np.zeros((len(points), 2), dtype=np.float64)
        for i, pt in enumerate(points):
            x_m = pt.cell_id * cell_length_m
            features[i, 0] = x_m / self.config.spatial_scale_m
            features[i, 1] = pt.timestamp_sec / self.config.temporal_scale_s

        db = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        )
        labels = db.fit_predict(features)

        clusters: list[JamCluster] = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in sorted(unique_labels):
            indices = np.where(labels == label)[0]
            cluster_points = [points[idx] for idx in indices]

            cells = [p.cell_id for p in cluster_points]
            times = [p.timestamp_sec for p in cluster_points]
            speeds = [p.speed_mps for p in cluster_points]
            days = list({p.day for p in cluster_points})

            min_c = min(cells)
            max_c = max(cells)
            min_t = min(times)
            max_t = max(times)

            extent = (max_c - min_c + 1) * cell_length_m
            duration = max_t - min_t
            mean_speed = float(np.mean(speeds))

            clusters.append(
                JamCluster(
                    cluster_id=int(label),
                    cell_ids=cells,
                    timestamps_sec=times,
                    tail_cell=min_c,
                    head_cell=max_c,
                    start_time_sec=min_t,
                    end_time_sec=max_t,
                    extent_m=extent,
                    duration_sec=duration,
                    mean_speed_mps=mean_speed,
                    recurring=False,
                    matched_days=days,
                )
            )

        return clusters

    def match_recurring_jams(
        self,
        daily_clusters: dict[int, list[JamCluster]],
    ) -> list[JamCluster]:
        """Matches jam clusters across multiple observation days to detect recurring patterns."""
        all_jams: list[JamCluster] = []
        for d, jams in daily_clusters.items():
            all_jams.extend(jams)

        threshold_days = self.config.recurrent_jam_day_threshold

        for i in range(len(all_jams)):
            matched_days_set = set(all_jams[i].matched_days)
            for j in range(len(all_jams)):
                if i == j:
                    continue
                # Overlap criteria: spatial overlap and time-of-day overlap
                spatial_overlap = not (
                    all_jams[i].head_cell < all_jams[j].tail_cell
                    or all_jams[j].head_cell < all_jams[i].tail_cell
                )
                time_overlap = not (
                    all_jams[i].end_time_sec < all_jams[j].start_time_sec
                    or all_jams[j].end_time_sec < all_jams[i].start_time_sec
                )
                if spatial_overlap and time_overlap:
                    matched_days_set.update(all_jams[j].matched_days)

            all_jams[i].matched_days = sorted(matched_days_set)
            if len(all_jams[i].matched_days) >= threshold_days:
                all_jams[i].recurring = True

        return all_jams
