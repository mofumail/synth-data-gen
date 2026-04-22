"""
ReportGenerator

Serializes aggregated evaluation results to HTML (human-readable) and
JSON (machine-readable, required for fidelity-utility correlation analysis).

serialize() must be called alongside generate() - the correlation analysis
needs per-seed fidelity scores and TSTR scores programmatically, not just
rendered in HTML.

"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from evaluation.data_classes import AggregatedResult


class ReportGenerator:

    def generate(self, aggregated_result: AggregatedResult) -> str:
        """Build and return an HTML report string."""
        ar = aggregated_result

        def fmt(v, decimals=4):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "-"
            return f"{v:.{decimals}f}"

        def mean_std(mean_v, std_v, decimals=4):
            return f"{fmt(mean_v, decimals)} ± {fmt(std_v, decimals)}"

        # --- Fidelity tables (train and val reference, 3-column: Transformer | Markov | TRTR) ---
        FIDELITY_METRICS = [
            ("JSD (action dist.)",  "jsd_action",               "[0, 1]", "↓ lower is better"),
            ("KS (session length)", "ks_session_length",        "[0, 1]", "↓ lower is better"),
            ("KS (temporal delta)", "ks_temporal_delta",        "[0, 1]", "↓ lower is better"),
            ("L1 (action bigrams)", "l1_action_bigrams",        "[0, 2]", "↓ lower is better"),
            ("L1 (item bigrams)",   "l1_item_bigrams",          "[0, 2]", "↓ lower is better"),
            ("Sample diversity",    "sample_diversity",         "ratio",  "1.0 = ideal (synth/real Jaccard)"),
            ("Conv. rate delta",    "conversion_rate_delta",    "[0, 1]", "↓ lower is better"),
            ("Cart abandon. delta", "cart_abandonment_delta",   "[0, 1]", "↓ lower is better"),
        ]
        BIAS_METRICS = [
            ("Item coverage",     "item_coverage",          "ratio",  "1.0 = ideal (synth/real unique at same N)"),
            ("Unique SKUs (synth)",       "unique_synth_skus",        "count", "catalog-% = this / VOCAB_K"),
            ("Unique SKUs (real matched)","unique_real_matched_skus", "count", "denom of Item coverage"),
            ("JSD (popularity)",  "popularity_jsd",         "[0, 1]", "↓ lower is better"),
            ("Gini coeff. delta", "gini_coefficient_delta", "[0, 1]", "↓ lower is better"),
        ]

        def _col(mean, std, attr, bias=False):
            obj_m = mean.bias if bias else mean
            obj_s = std.bias  if bias else std
            return mean_std(getattr(obj_m, attr), getattr(obj_s, attr))

        def _fidelity_compare_table(t_mean, t_std, m_mean, m_std, rr_mean, rr_std):
            rows = []
            for label, attr, rng, direction in FIDELITY_METRICS:
                rows.append([label,
                             _col(t_mean,  t_std,  attr),
                             _col(m_mean,  m_std,  attr),
                             _col(rr_mean, rr_std, attr),
                             rng, direction])
            for label, attr, rng, direction in BIAS_METRICS:
                rows.append([label,
                             _col(t_mean,  t_std,  attr, bias=True),
                             _col(m_mean,  m_std,  attr, bias=True),
                             _col(rr_mean, rr_std, attr, bias=True),
                             rng, direction])
            return _table(
                ["Metric", "Transformer", "Markov", "TRTR (real sample)", "Range", "Direction"],
                rows,
            )

        fidelity_train_html = _fidelity_compare_table(
            ar.fidelity_train_mean,        ar.fidelity_train_std,
            ar.fidelity_markov_train_mean, ar.fidelity_markov_train_std,
            ar.fidelity_trtr_train_mean,   ar.fidelity_trtr_train_std,
        )
        fidelity_val_html = _fidelity_compare_table(
            ar.fidelity_val_mean,        ar.fidelity_val_std,
            ar.fidelity_markov_val_mean, ar.fidelity_markov_val_std,
            ar.fidelity_trtr_val_mean,   ar.fidelity_trtr_val_std,
        )

        # --- Validity table ---
        vpm, vps = ar.validity_pre_mean,  ar.validity_pre_std
        vom, vos = ar.validity_post_mean, ar.validity_post_std
        validity_rows = [
            ("Illegal transition rate",     vpm.illegal_transition_rate,     vps.illegal_transition_rate,
                                            vom.illegal_transition_rate,     vos.illegal_transition_rate),
            ("Monotonicity violation rate", vpm.monotonicity_violation_rate, vps.monotonicity_violation_rate,
                                            vom.monotonicity_violation_rate, vos.monotonicity_violation_rate),
            ("Purchase exposure rate",      vpm.purchase_exposure_rate,      vps.purchase_exposure_rate,
                                            vom.purchase_exposure_rate,      vos.purchase_exposure_rate),
        ]
        validity_html = _table(
            ["Metric", "Pre-constraint (Mean ± Std)", "Post-constraint (Mean ± Std)", "Range", "Direction"],
            [[name, mean_std(pm, ps), mean_std(om, os), "[0, 1]", "↓ lower is better"]
             for name, pm, ps, om, os in validity_rows],
        )

        # --- Utility table ---
        utility_rows = []
        for um, us in zip(ar.utility_mean, ar.utility_std):
            utility_rows.append((um.condition, um.hr_at_k, us.hr_at_k, um.ndcg_at_k, us.ndcg_at_k, um.oov_rate, us.oov_rate))
        utility_html = _table(
            ["Condition", "HR@K (Mean ± Std)", "NDCG@K (Mean ± Std)", "OOV Rate (Mean ± Std)", "Range", "Direction"],
            [[cond, mean_std(hr_m, hr_s), mean_std(nd_m, nd_s), mean_std(oov_m, oov_s), "[0, 1]", "↑ HR/NDCG higher is better"]
             for cond, hr_m, hr_s, nd_m, nd_s, oov_m, oov_s in utility_rows],
        )

        # --- Correlation table ---
        if ar.correlations:
            corr_html = _table(
                ["Fidelity Metric", "Pearson r", "p-value"],
                [[c.fidelity_metric, fmt(c.correlation, 4), fmt(c.p_value, 4)]
                 for c in ar.correlations],
            )
        else:
            corr_html = "<p><em>Not computed (requires ≥ 3 seeds).</em></p>"

        num_seeds = len(ar.seed_results)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; color: #222; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin-bottom: 1.5em; min-width: 40em; }}
  th, td {{ border: 1px solid #bbb; padding: 6px 12px; text-align: left; }}
  th {{ background: #e8eaf6; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Synthetic Session Evaluation Report</h1>
<p><strong>Seeds:</strong> {num_seeds}</p>

<h2>Fidelity vs Training Distribution</h2>
<p>How well does each dataset match the distribution the generator was trained on? TRTR (real sample) is the upper bound reference.</p>
{fidelity_train_html}

<h2>Fidelity vs Validation Distribution (Generalization Check)</h2>
<p>How well does each dataset match the unseen validation period (Nov 15 – Dec 1)?</p>
{fidelity_val_html}

<h2>Validity</h2>
{validity_html}

<h2>Downstream Utility (TSTR)</h2>
{utility_html}

<h2>Fidelity–Utility Correlation</h2>
{corr_html}
</body>
</html>"""
        return html

    def save(self, report: str, path) -> None:
        """Write HTML report to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"  HTML report saved -> {path}")

    def serialize(self, aggregated_result: AggregatedResult, path) -> None:
        """Write machine-readable JSON of all results (including per-seed) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dataclasses.asdict(aggregated_result)
        path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        print(f"  JSON results saved -> {path}")


# Helpers

def _table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f"<td>{cell}</td>" for cell in row)
        body += f"<tr>{tds}</tr>\n"
    return f"<table>\n<thead><tr>{th}</tr></thead>\n<tbody>\n{body}</tbody>\n</table>"


def _json_default(obj):
    import math
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
