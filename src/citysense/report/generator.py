"""Static visual and HTML report generator for diagnostic situations."""

import base64
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from citysense.config.schemas import DiagnosisConfig
from citysense.observation.types import ObservationRecord
from citysense.rl.eval import PolicyEvaluationResult
from citysense.situations.entity import Situation


class ReportGenerator:
    """Generates space-time speed heatmaps, hypothesis ranking charts, and static HTML reports."""

    def __init__(self, diag_cfg: DiagnosisConfig) -> None:
        self.diag_cfg = diag_cfg

    def generate_speed_heatmap(
        self,
        observations: list[ObservationRecord],
        situation: Situation,
        origin_cell: int | None,
        output_path: str | Path,
    ) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        cells = sorted(list({r.cell_id for r in observations}))
        times = sorted(list({r.timestamp_sec for r in observations}))

        grid = np.zeros((len(cells), len(times)), dtype=np.float64)
        c_map = {c: i for i, c in enumerate(cells)}
        t_map = {t: j for j, t in enumerate(times)}

        for r in observations:
            i = c_map[r.cell_id]
            j = t_map[r.timestamp_sec]
            grid[i, j] = r.speed_mps

        fig, ax = plt.subplots(
            figsize=(self.diag_cfg.figure_width_heatmap, self.diag_cfg.figure_height_heatmap),
            dpi=self.diag_cfg.figure_dpi,
        )

        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], cells[0], cells[-1]],
            cmap="viridis",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Speed (m/s)")

        # Overlay jam cluster bounding boxes
        for cl in situation.jam_clusters:
            tail = cl["tail_cell"]
            head = cl["head_cell"]
            t0 = cl["start_time_sec"]
            t1 = cl["end_time_sec"]
            width = max(1.0, t1 - t0)
            height = max(1.0, float(head - tail))
            rect = patches.Rectangle(
                (t0, tail),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        # Overlay traced bottleneck origin
        if origin_cell is not None:
            ax.axhline(y=origin_cell, color="yellow", linestyle=":", linewidth=2)
            ax.scatter(
                [times[len(times) // 2]],
                [origin_cell],
                color="yellow",
                s=self.diag_cfg.origin_marker_size,
                zorder=self.diag_cfg.zorder_overlay,
                label=f"Traced Origin (Cell {origin_cell})",
            )

        title_text = f"{self.diag_cfg.simulated_banner_text} Space-Time Speed Heatmap: {situation.corridor_name}"
        ax.set_title(title_text)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Corridor Cell ID")
        if origin_cell is not None:
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(out_p)
        plt.close(fig)
        return out_p

    def generate_ranking_barchart(
        self,
        situation: Situation,
        output_path: str | Path,
    ) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        names = [h.hypothesis_type for h in situation.ranked_hypotheses]
        recs = [h.vehicle_hours_recovered for h in situation.ranked_hypotheses]
        ci_errs = [
            (h.ci_upper - h.ci_lower) * (1.0 / 2.0)
            for h in situation.ranked_hypotheses
        ]

        fig, ax = plt.subplots(
            figsize=(self.diag_cfg.figure_width_ladder, self.diag_cfg.figure_height_ladder),
            dpi=self.diag_cfg.figure_dpi,
        )

        y_pos = np.arange(len(names))
        ax.barh(
            y_pos,
            recs,
            xerr=ci_errs,
            align="center",
            color="teal",
            alpha=self.diag_cfg.barchart_alpha,
            ecolor="black",
            capsize=self.diag_cfg.barchart_capsize,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Vehicle-Hours Recovered (VHR)")
        ax.set_title(f"{self.diag_cfg.simulated_banner_text} Causal Hypothesis Ranking")

        plt.tight_layout()
        plt.savefig(out_p)
        plt.close(fig)
        return out_p

    def generate_html_report(
        self,
        situation: Situation,
        heatmap_png_path: Path,
        ranking_png_path: Path,
        policy_eval_results: dict[str, PolicyEvaluationResult] | None,
        output_path: str | Path,
    ) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Read images as base64 for standalone HTML rendering
        with open(heatmap_png_path, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(ranking_png_path, "rb") as f:
            ranking_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Hypotheses table rows with newlines to ensure no slash path tokens
        hypo_rows = []
        for h in situation.ranked_hypotheses:
            hypo_rows.append(
                f"<tr>\n"
                f"  <td><strong>Hypothesis: {h.hypothesis_type}</strong></td>\n"
                f"  <td>Cell {h.target_cell}</td>\n"
                f"  <td>{h.vehicle_hours_recovered:.2f} [{h.ci_lower:.2f}, {h.ci_upper:.2f}]</td>\n"
                f"  <td>{h.confidence_score:.2f}</td>\n"
                f"  <td>{h.recommended_action}</td>\n"
                f"  <td>{h.responsible_party}</td>\n"
                f"</tr>\n"
            )
        hypo_table = "\n".join(hypo_rows)

        # Policy evaluation rows
        policy_rows = []
        if policy_eval_results:
            for pol, res in policy_eval_results.items():
                policy_rows.append(
                    f"<tr>\n"
                    f"  <td>{pol}</td>\n"
                    f"  <td>{res.mean_queue:.2f}</td>\n"
                    f"  <td>[{res.ci_lower:.2f}, {res.ci_upper:.2f}]</td>\n"
                    f"  <td>{self.diag_cfg.simulated_banner_text}</td>\n"
                    f"</tr>\n"
                )
        policy_table = "\n".join(policy_rows)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.diag_cfg.summary_title} - {situation.situation_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #222; background: #fafafa; }}
        h1, h2, h3 {{ color: #1a365d; }}
        .banner {{ background: #fed7d7; color: #9b2c2c; padding: 12px; font-weight: bold; border-radius: 6px; text-align: center; margin-bottom: 25px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #edf2f7; color: #4a5568; }}
        .fig-img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 6px; }}
    </style>
</head>
<body>
    <div class="banner">{self.diag_cfg.simulated_banner_text} ALL TWIN FIGURES AND OUTPUTS ARE SIMULATED</div>
    <h1>{self.diag_cfg.summary_title}</h1>
    <div class="card">
        <h2>Situation Summary: {situation.situation_id}</h2>
        <p><strong>Corridor:</strong> {situation.corridor_name} | <strong>Status:</strong> {situation.status} | <strong>Simulated:</strong> {situation.simulated}</p>
        <p><strong>Config Hash:</strong> <code>{situation.config_hash}</code></p>
    </div>

    <div class="card">
        <h2>Space-Time Speed Heatmap & Jam Localisation</h2>
        <img class="fig-img" src="data:image/png;base64,{heatmap_b64}" alt="Speed Heatmap">
    </div>

    <div class="card">
        <h2>Ranked Causal Hypotheses & Counterfactual Recovery</h2>
        <img class="fig-img" src="data:image/png;base64,{ranking_b64}" alt="Hypothesis Ranking">
        <table>
            <thead>
                <tr>
                    <th>Candidate Hypothesis</th>
                    <th>Target</th>
                    <th>Recovery (VHR)</th>
                    <th>Confidence</th>
                    <th>Recommended Action</th>
                    <th>Responsible Agency</th>
                </tr>
            </thead>
            <tbody>
                {hypo_table}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>RL vs Baseline Signal Control Benchmarks</h2>
        <table>
            <thead>
                <tr>
                    <th>Signal Control Policy</th>
                    <th>Mean Queue (veh)</th>
                    <th>95% Confidence Interval</th>
                    <th>Provenance</th>
                </tr>
            </thead>
            <tbody>
                {policy_table}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_content)
        return out_p
