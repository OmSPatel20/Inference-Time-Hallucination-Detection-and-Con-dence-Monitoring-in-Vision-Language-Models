"""
11_analysis_plots.py — Generate all figures and tables for the final report.

Reads results from outputs/ and generates publication-quality plots.

Usage:
    python scripts/11_analysis_plots.py
"""

import sys
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, load_jsonl


def set_plot_style():
    """Set consistent plot style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox_inches": "tight",
    })


def plot_roc_curves(config):
    """
    Plot 1: ROC curves for entropy detection across POPE splits.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    splits = ["random", "popular", "adversarial"]

    # Auto-detect which quantization files exist
    pope_dir = os.path.join(config["paths"]["output_dir"], "pope_results")
    quant = config.get("hardware", {}).get("default_quantization", "int4")

    for ax, split in zip(axes, splits):
        # Try the default quant, then fall back to any available file
        filepath = os.path.join(pope_dir, f"pope_{split}_{quant}.jsonl")
        if not os.path.exists(filepath):
            # Try other quants
            for q in ["int4", "int8", "fp16"]:
                alt = os.path.join(pope_dir, f"pope_{split}_{q}.jsonl")
                if os.path.exists(alt):
                    filepath = alt
                    break
        if not os.path.exists(filepath):
            ax.set_title(f"{split} (no data)")
            continue

        data = load_jsonl(filepath)
        df = pd.DataFrame(data)

        # Binary labels: 1 = hallucination (model said yes when answer is no)
        labels = df["is_hallucination"].astype(int).values

        # Plot ROC for entropy score
        if "entropy_halluc_score" in df.columns:
            scores = df["entropy_halluc_score"].values
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Entropy (AUC={roc_auc:.3f})", linewidth=2)

        # Plot ROC for contrastive score
        if "contrastive_halluc_score" in df.columns:
            valid = df["contrastive_halluc_score"].notna()
            if valid.sum() > 50:
                scores = df.loc[valid, "contrastive_halluc_score"].values
                labs = labels[valid.values]
                fpr, tpr, _ = roc_curve(labs, scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Contrastive (AUC={roc_auc:.3f})",
                        linewidth=2, linestyle="--")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"POPE {split.capitalize()}")
        ax.legend(loc="lower right")

    plt.suptitle("ROC Curves: Hallucination Detection on POPE", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "roc_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_quantization_comparison(config):
    """
    Plot 2: Bar chart comparing accuracy and hallucination rate across quantizations.
    """
    summary_file = os.path.join(config["paths"]["output_dir"], "experiment_matrix_results.json")
    if not os.path.exists(summary_file):
        print("  [SKIP] No experiment matrix results found")
        return

    with open(summary_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy by quantization and split
    pivot_acc = df.pivot(index="split", columns="quantization", values="accuracy")
    pivot_acc.plot(kind="bar", ax=ax1, rot=0)
    ax1.set_title("Accuracy by Quantization")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(title="Quantization")

    # Hallucination rate by quantization and split
    pivot_halluc = df.pivot(index="split", columns="quantization", values="hallucination_rate")
    pivot_halluc.plot(kind="bar", ax=ax2, rot=0)
    ax2.set_title("Hallucination Rate by Quantization")
    ax2.set_ylabel("Hallucination Rate")
    ax2.legend(title="Quantization")

    plt.suptitle("Effect of Quantization on POPE Performance", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "quantization_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_entropy_vs_hallucination(config):
    """
    Plot 3: Scatter plot of entropy score vs. hallucination status.
    """
    quant = config.get("hardware", {}).get("default_quantization", "int4")

    filepath = os.path.join(
        config["paths"]["output_dir"], "pope_results", f"pope_adversarial_{quant}.jsonl"
    )
    if not os.path.exists(filepath):
        # Try any available file
        for q in ["int4", "int8", "fp16"]:
            for s in ["adversarial", "random", "popular"]:
                alt = os.path.join(config["paths"]["output_dir"], "pope_results", f"pope_{s}_{q}.jsonl")
                if os.path.exists(alt):
                    filepath = alt
                    break
            if os.path.exists(filepath):
                break
    if not os.path.exists(filepath):
        print("  [SKIP] No POPE results found for scatter plot")
        return

    data = load_jsonl(filepath)
    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy distribution by hallucination status
    colors = {True: "#e74c3c", False: "#2ecc71"}
    for is_halluc in [False, True]:
        subset = df[df["is_hallucination"] == is_halluc]
        label = "Hallucination" if is_halluc else "Correct"
        ax1.hist(subset["entropy_mean"], bins=50, alpha=0.6,
                 label=label, color=colors[is_halluc])

    ax1.set_xlabel("Mean Token Entropy")
    ax1.set_ylabel("Count")
    ax1.set_title("Entropy Distribution")
    ax1.legend()

    # Entropy vs top-1 probability, colored by hallucination
    scatter = ax2.scatter(
        df["entropy_mean"], df["top_prob_mean"],
        c=df["is_hallucination"].map({True: "#e74c3c", False: "#2ecc71"}),
        alpha=0.3, s=10,
    )
    ax2.set_xlabel("Mean Token Entropy")
    ax2.set_ylabel("Mean Top-1 Probability")
    ax2.set_title("Entropy vs. Confidence")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Hallucination"),
    ]
    ax2.legend(handles=legend_elements)

    plt.suptitle("Entropy Analysis of Hallucinations", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "entropy_analysis.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confidence_drift(config):
    """
    Plot 4: Time series of confidence metrics over sustained inference.
    """
    drift_dir = os.path.join(config["paths"]["output_dir"], "drift_monitor")
    drift_files = glob.glob(os.path.join(drift_dir, "drift_*.jsonl"))

    if not drift_files:
        print("  [SKIP] No drift monitor data found")
        return

    # Use the first drift file found
    filepath = drift_files[0]
    data = load_jsonl(filepath)
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = 100  # Rolling window for smoothing

    # (a) Entropy over time
    ax = axes[0, 0]
    ax.plot(df["idx"], df["entropy_mean"], alpha=0.2, color="blue", linewidth=0.5)
    ax.plot(df["idx"], df["entropy_mean"].rolling(window).mean(),
            color="blue", linewidth=2, label=f"Rolling avg ({window})")
    ax.set_xlabel("Inference Index")
    ax.set_ylabel("Mean Entropy")
    ax.set_title("(a) Entropy Drift Over Time")
    ax.legend()

    # (b) Latency over time
    ax = axes[0, 1]
    ax.plot(df["idx"], df["latency_ms"], alpha=0.2, color="orange", linewidth=0.5)
    ax.plot(df["idx"], df["latency_ms"].rolling(window).mean(),
            color="orange", linewidth=2, label=f"Rolling avg ({window})")
    ax.set_xlabel("Inference Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(b) Latency Drift Over Time")
    ax.legend()

    # (c) GPU memory over time
    ax = axes[1, 0]
    ax.plot(df["idx"], df["gpu_mem_allocated_mb"], color="green", linewidth=1)
    if "gpu_mem_reserved_mb" in df.columns:
        ax.plot(df["idx"], df["gpu_mem_reserved_mb"], color="green",
                linewidth=1, linestyle="--", alpha=0.5, label="Reserved")
    ax.set_xlabel("Inference Index")
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("(c) GPU Memory Over Time")

    # (d) Rolling accuracy
    ax = axes[1, 1]
    rolling_acc = df["is_correct"].rolling(window).mean()
    ax.plot(df["idx"], rolling_acc, color="purple", linewidth=2)
    ax.set_xlabel("Inference Index")
    ax.set_ylabel(f"Rolling Accuracy (window={window})")
    ax.set_title("(d) Accuracy Drift Over Time")
    ax.set_ylim(0.5, 1.0)

    quant = os.path.basename(filepath).split("_")[1]
    plt.suptitle(f"Confidence Drift Analysis ({quant})", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "confidence_drift.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_detection_heatmap(config):
    """
    Plot 5: Heatmap of detection AUC across quantization × POPE split.
    """
    pope_dir = os.path.join(config["paths"]["output_dir"], "pope_results")
    if not os.path.exists(pope_dir):
        print("  [SKIP] No POPE results directory")
        return

    results = {}
    for filepath in glob.glob(os.path.join(pope_dir, "pope_*.jsonl")):
        fname = os.path.basename(filepath)
        # Parse: pope_{split}_{quant}.jsonl
        parts = fname.replace(".jsonl", "").split("_")
        if len(parts) < 3:
            continue
        split = parts[1]
        quant = parts[2]

        data = load_jsonl(filepath)
        df = pd.DataFrame(data)

        labels = df["is_hallucination"].astype(int).values
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        scores = df["entropy_halluc_score"].values
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            results[(quant, split)] = roc_auc
        except Exception:
            pass

    if not results:
        print("  [SKIP] Not enough data for heatmap")
        return

    # Build matrix
    quants = sorted(set(k[0] for k in results))
    splits = sorted(set(k[1] for k in results))

    matrix = np.zeros((len(quants), len(splits)))
    for i, q in enumerate(quants):
        for j, s in enumerate(splits):
            matrix[i, j] = results.get((q, s), 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=splits, yticklabels=quants,
        vmin=0.5, vmax=1.0, ax=ax,
    )
    ax.set_xlabel("POPE Split")
    ax.set_ylabel("Quantization")
    ax.set_title("Entropy Detection AUC: Quantization × POPE Split")

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "detection_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def generate_summary_table(config):
    """
    Generate a LaTeX-ready summary table of all results.
    """
    summary_file = os.path.join(config["paths"]["output_dir"], "experiment_matrix_results.json")
    if not os.path.exists(summary_file):
        print("  [SKIP] No experiment matrix results")
        return

    with open(summary_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{POPE Evaluation Results Across Conditions}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Quant. & Split & Accuracy & Halluc. Rate & Samples \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['quantization']} & {row['split']} & "
            f"{row['accuracy']:.4f} & {row['hallucination_rate']:.4f} & "
            f"{row['total_samples']} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\label{tab:pope_results}",
        r"\end{table}",
    ])

    out_path = os.path.join(config["paths"]["output_dir"], "plots", "results_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    config = load_config()
    set_plot_style()

    print("=" * 60)
    print("GENERATING ANALYSIS PLOTS")
    print("=" * 60)

    print("\n1. ROC curves...")
    plot_roc_curves(config)

    print("\n2. Quantization comparison...")
    plot_quantization_comparison(config)

    print("\n3. Entropy analysis...")
    plot_entropy_vs_hallucination(config)

    print("\n4. Confidence drift...")
    plot_confidence_drift(config)

    print("\n5. Detection heatmap...")
    plot_detection_heatmap(config)

    print("\n6. Summary table (LaTeX)...")
    generate_summary_table(config)

    print(f"\nAll plots saved to: {os.path.join(config['paths']['output_dir'], 'plots')}/")


if __name__ == "__main__":
    main()
