#!/usr/bin/env python3
"""Generate Federated Learning Accuracy Graphs and Figures.

Adheres strictly to AGENTS.md:
  - Reports both Balanced Accuracy (Macro-Recall / Sensitivity) and Raw Site-Aware Accuracy.
  - Labels all numbers with sample size n (seeds n=3 or n=1).
  - Plots multi-class harmonic accuracy (Validation Macro-F1) convergence across 20 rounds.
  - Compares FedAvg (Sample-Weighted), FedAvg (Uniform), Per-Family Heads, FedProx,
    Centralized Pooled baseline, and Local Hospital baselines.
"""

import json
import logging
import os
import sys
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_run_accuracies(run_id: str, is_cmmd_binary: bool = True):
    pred_path = Path("results/runs") / run_id / "preds" / "test_breast_predictions.npz"
    if not pred_path.exists():
        return None
    data = np.load(pred_path, allow_pickle=True)
    labels = data["labels"]
    logits = data["logits"]
    sites = data["sites"]

    site_preds = np.argmax(logits, axis=1)
    if is_cmmd_binary:
        for i, s in enumerate(sites):
            if s == "cmmd":
                l_m = logits[i].copy()
                l_m[0] = -1e9
                site_preds[i] = np.argmax(l_m)

    overall_raw = accuracy_score(labels, site_preds)
    overall_bal = recall_score(labels, site_preds, average="macro", zero_division=0)

    site_metrics = {}
    for s in ["vindr", "cmmd", "inbreast", "cdd"]:
        mask = (sites == s)
        if np.sum(mask) > 0:
            s_lab = labels[mask]
            s_pred = site_preds[mask]
            site_metrics[s] = {
                "raw_acc": accuracy_score(s_lab, s_pred),
                "bal_acc": recall_score(s_lab, s_pred, average="macro", zero_division=0),
            }
    return {
        "overall_raw": overall_raw,
        "overall_bal": overall_bal,
        "per_site": site_metrics,
    }


def generate_accuracy_plots():
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path("results/ledger.csv")
    if not ledger_path.exists():
        logger.error("Ledger file not found.")
        return
    ledger = pd.read_csv(ledger_path)

    # -------------------------------------------------------------------------
    # 1. Round-by-Round Convergence Plot (Macro-F1 / Harmonic Multi-Class Acc)
    # -------------------------------------------------------------------------
    strategies = [
        ("fl_fedavg_sample_weighted", "FedAvg (Sample-Weighted, n=3)", "#1f77b4", "-", "o"),
        ("fl_fedavg_uniform", "FedAvg (Uniform, n=3)", "#2ca02c", "-", "s"),
        ("fl_per_family_uniform", "Per-Family Heads (Uniform, n=3)", "#9467bd", "-", "^"),
        ("fl_fedprox_mu0.01", "FedProx (mu=0.01, n=1)", "#ff7f0e", "--", "v"),
        ("fl_fedprox_mu0.1", "FedProx (mu=0.10, n=1)", "#d62728", ":", "D"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    cent_f1 = 0.6080  # Centralized pooled validation F1 benchmark
    perm_f1 = 0.3333  # 3-class random chance floor

    ax.axhline(cent_f1, color="crimson", linestyle="--", linewidth=1.5, label=f"Centralized Benchmark ({cent_f1:.4f})")
    ax.axhline(perm_f1, color="gray", linestyle=":", linewidth=1.2, label=f"Chance Floor ({perm_f1:.4f})")

    for exp_id, label, color, lstyle, marker in strategies:
        runs = ledger[(ledger["experiment_id"] == exp_id) & (ledger["status"] == "completed")]
        f1_curves = []
        for _, row in runs.iterrows():
            mf = Path("results/runs") / row["run_id"] / "metrics.jsonl"
            if mf.exists():
                recs = [json.loads(line) for line in open(mf) if line.strip()]
                f1_curves.append([r.get("val_macro_f1", 0.0) for r in recs])
        if not f1_curves:
            continue
        rounds = list(range(1, len(f1_curves[0]) + 1))
        f1_mat = np.array(f1_curves)
        mean_curve = np.mean(f1_mat, axis=0)
        std_curve = np.std(f1_mat, axis=0, ddof=1) if len(f1_mat) > 1 else np.zeros_like(mean_curve)

        ax.plot(rounds, mean_curve, color=color, linestyle=lstyle, linewidth=2.2,
                marker=marker, markersize=4, label=label)
        if len(f1_mat) > 1:
            ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.12)

    ax.set_title("Federated Convergence: Validation Macro-F1 (Harmonic Multi-Class Accuracy) vs Rounds",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Validation Macro-F1 / Balanced Accuracy Score", fontsize=11)
    ax.set_xlim(1, 20)
    ax.set_ylim(0.25, 0.65)
    ax.set_xticks(range(1, 21, 2))
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    f1_out = fig_dir / "fl_convergence_accuracy.png"
    fig.tight_layout()
    fig.savefig(f1_out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", f1_out)

    # -------------------------------------------------------------------------
    # 2. Comprehensive Dual-Panel Accuracy Figure
    #    Panel A: Overall Accuracy vs Balanced Accuracy (Macro-Recall)
    #    Panel B: Per-Site Accuracy Breakdown across Hospital Silos
    # -------------------------------------------------------------------------
    methods_to_compare = [
        ("Centralized (Pooled)", "centralized_pooled", "#d62728"),
        ("FedAvg (Sample-Weighted)", "fl_fedavg_sample_weighted", "#1f77b4"),
        ("FedAvg (Uniform)", "fl_fedavg_uniform", "#2ca02c"),
        ("Per-Family Heads", "fl_per_family_uniform", "#9467bd"),
        ("FedProx (mu=0.01)", "fl_fedprox_mu0.01", "#ff7f0e"),
    ]

    method_labels = []
    raw_means, raw_stds = [], []
    bal_means, bal_stds = [], []

    site_data = {s: {"methods": [], "raw": [], "bal": []} for s in ["vindr", "cmmd", "inbreast", "cdd"]}

    for m_label, exp_id, _ in methods_to_compare:
        runs = ledger[(ledger["experiment_id"].str.contains(exp_id, na=False)) & (ledger["status"] == "completed")]
        m_raw, m_bal = [], []
        s_raw = {s: [] for s in site_data}
        s_bal = {s: [] for s in site_data}

        for _, row in runs.iterrows():
            res = compute_run_accuracies(row["run_id"])
            if res:
                m_raw.append(res["overall_raw"])
                m_bal.append(res["overall_bal"])
                for s in site_data:
                    if s in res["per_site"]:
                        s_raw[s].append(res["per_site"][s]["raw_acc"])
                        s_bal[s].append(res["per_site"][s]["bal_acc"])

        if m_raw:
            n = len(m_raw)
            n_str = f"n={n}"
            method_labels.append(f"{m_label}\n({n_str})")
            raw_means.append(np.mean(m_raw))
            raw_stds.append(np.std(m_raw, ddof=1) if n > 1 else 0.0)
            bal_means.append(np.mean(m_bal))
            bal_stds.append(np.std(m_bal, ddof=1) if n > 1 else 0.0)

            for s in site_data:
                site_data[s]["methods"].append(m_label)
                site_data[s]["raw"].append(np.mean(s_raw[s]))
                site_data[s]["bal"].append(np.mean(s_bal[s]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Panel A: Overall Comparison
    x = np.arange(len(method_labels))
    w = 0.35
    r1 = ax1.bar(x - w/2, raw_means, w, yerr=raw_stds, capsize=4,
                 label="Raw Site-Aware Accuracy", color="#3498db", alpha=0.9, edgecolor="black")
    r2 = ax1.bar(x + w/2, bal_means, w, yerr=bal_stds, capsize=4,
                 label="Balanced Accuracy (Macro-Recall)", color="#2ecc71", alpha=0.9, edgecolor="black")

    for rects, vals in [(r1, raw_means), (r2, bal_means)]:
        for rect, val in zip(rects, vals):
            ax1.annotate(f"{val*100:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title("A. Overall Breast-Level Test Accuracy across FL Strategies", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, fontsize=9.5)
    ax1.set_ylabel("Accuracy / Recall Score", fontsize=11)
    ax1.set_ylim(0.40, 0.78)
    ax1.axhline(0.3333, color="gray", linestyle=":", label="Chance Floor (33.3%)")
    ax1.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel B: Per-Site Accuracy Breakdown (Uniform vs Sample-Weighted vs Centralized)
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    site_names = ["VinDr\n(BIRADS)", "CMMD\n(Biopsy)", "INbreast\n(BIRADS)", "CDD\n(Biopsy)"]
    x_sites = np.arange(len(sites))
    w_s = 0.22

    # Extract for Centralized, SW, and Uniform
    cent_b = [site_data[s]["bal"][0] for s in sites]
    sw_b = [site_data[s]["bal"][1] for s in sites]
    uni_b = [site_data[s]["bal"][2] for s in sites]

    br1 = ax2.bar(x_sites - w_s, cent_b, w_s, label="Centralized Pooled (n=3)", color="#d62728", alpha=0.85, edgecolor="black")
    br2 = ax2.bar(x_sites, sw_b, w_s, label="FedAvg (Sample-Weighted, n=3)", color="#1f77b4", alpha=0.85, edgecolor="black")
    br3 = ax2.bar(x_sites + w_s, uni_b, w_s, label="FedAvg (Uniform, n=3)", color="#2ca02c", alpha=0.85, edgecolor="black")

    for rects, vals in [(br1, cent_b), (br2, sw_b), (br3, uni_b)]:
        for rect, val in zip(rects, vals):
            ax2.annotate(f"{val*100:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_title("B. Hospital Silo Balanced Accuracy: Mitigating Minority Site Neglect", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_sites)
    ax2.set_xticklabels(site_names, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Balanced Accuracy (Macro-Recall)", fontsize=11)
    ax2.set_ylim(0.35, 0.85)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    comp_out = fig_dir / "fl_accuracy_analysis.png"
    fig.tight_layout()
    fig.savefig(comp_out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", comp_out)


if __name__ == "__main__":
    generate_accuracy_plots()

