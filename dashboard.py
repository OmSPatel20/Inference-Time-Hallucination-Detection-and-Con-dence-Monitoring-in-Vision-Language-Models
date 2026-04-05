"""
dashboard.py — Streamlit dashboard for exploring experiment results.

Launch with:
    python run.py dashboard
    OR
    streamlit run dashboard.py

Requirements:
    pip install streamlit plotly
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs")
POPE_DIR = OUTPUT_DIR / "pope_results"
CHAIR_DIR = OUTPUT_DIR / "chair_results"
DRIFT_DIR = OUTPUT_DIR / "drift_monitor"


def load_jsonl(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VLM Hallucination Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 VLM Hallucination Detection Dashboard")
st.caption("Inference-time detection on LLaVA-1.5-7B")


# ---------------------------------------------------------------------------
# Sidebar — Data loading
# ---------------------------------------------------------------------------

st.sidebar.header("📁 Data")

# Detect available result files
pope_files = sorted(glob.glob(str(POPE_DIR / "pope_*.jsonl")))
chair_files = sorted(glob.glob(str(CHAIR_DIR / "chair_*.jsonl")))
drift_files = sorted(glob.glob(str(DRIFT_DIR / "drift_*.jsonl")))

if not pope_files and not chair_files and not drift_files:
    st.error(
        "No results found in `outputs/`. "
        "Run experiments first:\n\n"
        "```\npython run.py test\n```"
    )
    st.stop()

st.sidebar.write(f"POPE files: {len(pope_files)}")
st.sidebar.write(f"CHAIR files: {len(chair_files)}")
st.sidebar.write(f"Drift files: {len(drift_files)}")

# Tabs
tab_overview, tab_pope, tab_chair, tab_drift, tab_compare = st.tabs([
    "📊 Overview", "🎯 POPE Results", "📝 CHAIR Results",
    "📈 Confidence Drift", "⚖️ Detection Comparison"
])


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.header("Experiment Overview")

    # Load summaries
    summary_file = OUTPUT_DIR / "experiment_matrix_results.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summaries = json.load(f)
        df_summary = pd.DataFrame(summaries)

        col1, col2, col3 = st.columns(3)
        with col1:
            avg_acc = df_summary["accuracy"].mean()
            st.metric("Avg Accuracy", f"{avg_acc:.1%}")
        with col2:
            avg_halluc = df_summary["hallucination_rate"].mean()
            st.metric("Avg Hallucination Rate", f"{avg_halluc:.1%}")
        with col3:
            total = df_summary["total_samples"].sum()
            st.metric("Total Samples", f"{total:,}")

        st.dataframe(
            df_summary[["split", "quantization", "accuracy",
                         "hallucination_rate", "total_samples"]].round(4),
            use_container_width=True,
        )
    else:
        st.info("Run `python run.py pope` to generate overview data.")

    # CHAIR summary
    chair_summaries = sorted(glob.glob(str(CHAIR_DIR / "chair_summary_*.json")))
    if chair_summaries:
        st.subheader("CHAIR Results")
        for sf in chair_summaries:
            with open(sf) as f:
                cs = json.load(f)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"CHAIR_i ({cs.get('quantization', '?')})",
                          f"{cs.get('CHAIR_i', 0):.4f}")
            with col2:
                st.metric(f"CHAIR_s ({cs.get('quantization', '?')})",
                          f"{cs.get('CHAIR_s', 0):.4f}")
            with col3:
                st.metric("Captions evaluated", cs.get("num_samples", "?"))


# ---------------------------------------------------------------------------
# Tab 2: POPE Results
# ---------------------------------------------------------------------------

with tab_pope:
    st.header("POPE Benchmark Analysis")

    if not pope_files:
        st.info("No POPE results yet. Run: `python run.py pope`")
    else:
        # File selector
        selected_pope = st.selectbox(
            "Select result file",
            pope_files,
            format_func=lambda x: Path(x).stem,
        )

        df = pd.DataFrame(load_jsonl(selected_pope))
        n = len(df)

        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{df['is_correct'].mean():.1%}")
        with col2:
            st.metric("Hallucination Rate", f"{df['is_hallucination'].mean():.1%}")
        with col3:
            st.metric("Samples", n)
        with col4:
            st.metric("Avg Latency", f"{df['latency_ms'].mean():.0f} ms")

        # --- ROC Curve ---
        st.subheader("ROC Curve — Entropy Detection")

        labels = df["is_hallucination"].astype(int).values

        if labels.sum() > 0 and labels.sum() < len(labels):
            from sklearn.metrics import roc_curve, auc

            fig = go.Figure()

            # Entropy ROC
            if "entropy_halluc_score" in df.columns:
                fpr, tpr, _ = roc_curve(labels, df["entropy_halluc_score"].values)
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"Entropy (AUC={roc_auc:.3f})",
                    line=dict(width=2),
                ))

            # Contrastive ROC
            if "contrastive_halluc_score" in df.columns:
                valid = df["contrastive_halluc_score"].notna()
                if valid.sum() > 50:
                    fpr, tpr, _ = roc_curve(
                        labels[valid.values],
                        df.loc[valid, "contrastive_halluc_score"].values,
                    )
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"Contrastive (AUC={roc_auc:.3f})",
                        line=dict(width=2, dash="dash"),
                    ))

            # Diagonal
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="gray", dash="dot"),
                showlegend=False,
            ))

            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough hallucination examples to plot ROC curve.")

        # --- Entropy distribution ---
        st.subheader("Entropy Distribution")

        fig = px.histogram(
            df, x="entropy_mean", color="is_hallucination",
            nbins=60, barmode="overlay", opacity=0.6,
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            labels={"is_hallucination": "Hallucination", "entropy_mean": "Mean Token Entropy"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # --- Sample browser ---
        st.subheader("Browse Individual Samples")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            show_only = st.radio(
                "Filter", ["All", "Hallucinations only", "Correct only"],
                horizontal=True,
            )
        with filter_col2:
            sort_by = st.selectbox(
                "Sort by",
                ["entropy_mean", "entropy_max", "latency_ms", "idx"],
            )

        filtered = df.copy()
        if show_only == "Hallucinations only":
            filtered = filtered[filtered["is_hallucination"]]
        elif show_only == "Correct only":
            filtered = filtered[filtered["is_correct"]]

        filtered = filtered.sort_values(sort_by, ascending=False)

        display_cols = [
            "idx", "question", "gt_label", "pred_label", "answer_raw",
            "is_hallucination", "entropy_mean", "top_prob_mean", "latency_ms",
        ]
        existing_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[existing_cols].head(100).round(4),
            use_container_width=True,
            height=400,
        )


# ---------------------------------------------------------------------------
# Tab 3: CHAIR Results
# ---------------------------------------------------------------------------

with tab_chair:
    st.header("CHAIR — Caption Hallucination Analysis")

    if not chair_files:
        st.info("No CHAIR results yet. Run: `python run.py chair`")
    else:
        selected_chair = st.selectbox(
            "Select CHAIR file",
            chair_files,
            format_func=lambda x: Path(x).stem,
            key="chair_select",
        )

        df_chair = pd.DataFrame(load_jsonl(selected_chair))

        col1, col2, col3 = st.columns(3)
        with col1:
            chair_i = df_chair["num_hallucinated"].sum() / max(df_chair["num_mentioned"].sum(), 1)
            st.metric("CHAIR_i", f"{chair_i:.4f}")
        with col2:
            chair_s = df_chair["has_hallucination"].mean()
            st.metric("CHAIR_s", f"{chair_s:.4f}")
        with col3:
            st.metric("Captions", len(df_chair))

        # Hallucinated objects breakdown
        st.subheader("Most Hallucinated Objects")

        all_halluc_objects = []
        for _, row in df_chair.iterrows():
            all_halluc_objects.extend(row["hallucinated_objects"])

        if all_halluc_objects:
            from collections import Counter
            counts = Counter(all_halluc_objects).most_common(15)
            df_counts = pd.DataFrame(counts, columns=["Object", "Count"])

            fig = px.bar(df_counts, x="Count", y="Object", orientation="h",
                         color="Count", color_continuous_scale="Reds")
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No hallucinated objects found!")

        # Entropy vs hallucination
        if "entropy_mean" in df_chair.columns:
            st.subheader("Entropy vs Caption Hallucination")
            fig = px.scatter(
                df_chair, x="entropy_mean", y="num_hallucinated",
                color="has_hallucination",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                labels={
                    "entropy_mean": "Mean Token Entropy",
                    "num_hallucinated": "# Hallucinated Objects",
                },
                hover_data=["caption"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Browse captions
        st.subheader("Browse Captions")
        halluc_only = st.checkbox("Show only captions with hallucinations", value=False)
        browse_df = df_chair[df_chair["has_hallucination"]] if halluc_only else df_chair

        for _, row in browse_df.head(20).iterrows():
            with st.expander(f"Image {row['image_id']} — {row['num_hallucinated']} hallucinated"):
                st.write(f"**Caption:** {row['caption']}")
                st.write(f"**Mentioned:** {', '.join(row['mentioned_objects'])}")
                st.write(f"**Ground truth:** {', '.join(row['gt_objects'][:15])}...")
                if row["hallucinated_objects"]:
                    st.error(f"**Hallucinated:** {', '.join(row['hallucinated_objects'])}")
                else:
                    st.success("No hallucinations")


# ---------------------------------------------------------------------------
# Tab 4: Confidence Drift
# ---------------------------------------------------------------------------

with tab_drift:
    st.header("Confidence Drift Over Time")

    if not drift_files:
        st.info("No drift data yet. Run: `python run.py drift`")
    else:
        selected_drift = st.selectbox(
            "Select drift file",
            drift_files,
            format_func=lambda x: Path(x).stem,
            key="drift_select",
        )

        df_drift = pd.DataFrame(load_jsonl(selected_drift))
        n = len(df_drift)

        st.write(f"**{n} consecutive inferences** logged")

        # Rolling window selector
        window = st.slider("Smoothing window", 10, 200, 50, step=10)

        # 2x2 chart grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Mean Entropy Over Time",
                "Latency Over Time",
                "GPU Memory Over Time",
                "Rolling Accuracy",
            ],
        )

        # (1) Entropy
        fig.add_trace(go.Scatter(
            x=df_drift["idx"], y=df_drift["entropy_mean"],
            mode="lines", opacity=0.15, line=dict(color="blue"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_drift["idx"],
            y=df_drift["entropy_mean"].rolling(window).mean(),
            mode="lines", line=dict(color="blue", width=2),
            name="Entropy (smoothed)",
        ), row=1, col=1)

        # (2) Latency
        fig.add_trace(go.Scatter(
            x=df_drift["idx"], y=df_drift["latency_ms"],
            mode="lines", opacity=0.15, line=dict(color="orange"),
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=df_drift["idx"],
            y=df_drift["latency_ms"].rolling(window).mean(),
            mode="lines", line=dict(color="orange", width=2),
            name="Latency (smoothed)",
        ), row=1, col=2)

        # (3) GPU Memory
        fig.add_trace(go.Scatter(
            x=df_drift["idx"], y=df_drift["gpu_mem_allocated_mb"],
            mode="lines", line=dict(color="green", width=1),
            name="GPU Allocated",
        ), row=2, col=1)
        if "gpu_mem_reserved_mb" in df_drift.columns:
            fig.add_trace(go.Scatter(
                x=df_drift["idx"], y=df_drift["gpu_mem_reserved_mb"],
                mode="lines", line=dict(color="green", width=1, dash="dash"),
                name="GPU Reserved",
            ), row=2, col=1)

        # (4) Rolling accuracy
        rolling_acc = df_drift["is_correct"].rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=df_drift["idx"], y=rolling_acc,
            mode="lines", line=dict(color="purple", width=2),
            name="Accuracy (rolling)",
        ), row=2, col=2)

        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        st.subheader("Drift Statistics")

        # Split into first half vs second half
        half = n // 2
        first = df_drift.iloc[:half]
        second = df_drift.iloc[half:]

        drift_stats = pd.DataFrame({
            "Metric": ["Mean Entropy", "Mean Latency (ms)", "Accuracy",
                       "Hallucination Rate", "GPU Memory (MB)"],
            "First Half": [
                first["entropy_mean"].mean(),
                first["latency_ms"].mean(),
                first["is_correct"].mean(),
                first["is_hallucination"].mean(),
                first["gpu_mem_allocated_mb"].mean(),
            ],
            "Second Half": [
                second["entropy_mean"].mean(),
                second["latency_ms"].mean(),
                second["is_correct"].mean(),
                second["is_hallucination"].mean(),
                second["gpu_mem_allocated_mb"].mean(),
            ],
        })
        drift_stats["Change"] = drift_stats["Second Half"] - drift_stats["First Half"]
        drift_stats["Change %"] = (
            (drift_stats["Change"] / drift_stats["First Half"]) * 100
        ).round(2)

        st.dataframe(drift_stats.round(4), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Detection Method Comparison
# ---------------------------------------------------------------------------

with tab_compare:
    st.header("Detection Method Comparison")

    if not pope_files:
        st.info("No POPE results yet. Run: `python run.py pope`")
    else:
        # Load all POPE results
        all_data = []
        for f in pope_files:
            data = load_jsonl(f)
            all_data.extend(data)

        if not all_data:
            st.warning("POPE result files are empty.")
        else:
            df_all = pd.DataFrame(all_data)

            st.subheader("Entropy Score Distribution by Outcome")

            fig = px.box(
                df_all, x="split", y="entropy_mean", color="is_hallucination",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                labels={
                    "entropy_mean": "Mean Token Entropy",
                    "split": "POPE Split",
                    "is_hallucination": "Hallucination",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Contrastive comparison (if available)
            if "contrastive_halluc_score" in df_all.columns:
                valid = df_all["contrastive_halluc_score"].notna()
                if valid.sum() > 100:
                    st.subheader("Entropy vs Contrastive Detection Scores")

                    fig = px.scatter(
                        df_all[valid],
                        x="entropy_halluc_score",
                        y="contrastive_halluc_score",
                        color="is_hallucination",
                        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                        opacity=0.4,
                        labels={
                            "entropy_halluc_score": "Entropy Halluc. Score",
                            "contrastive_halluc_score": "Contrastive Halluc. Score",
                        },
                    )
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)

            # AUC comparison table
            st.subheader("Detection AUC by Split")

            from sklearn.metrics import roc_auc_score

            auc_rows = []
            for split in df_all["split"].unique():
                subset = df_all[df_all["split"] == split]
                labels = subset["is_hallucination"].astype(int).values

                if labels.sum() == 0 or labels.sum() == len(labels):
                    continue

                row = {"Split": split}

                # Entropy AUC
                if "entropy_halluc_score" in subset.columns:
                    try:
                        row["Entropy AUC"] = roc_auc_score(
                            labels, subset["entropy_halluc_score"].values
                        )
                    except Exception:
                        row["Entropy AUC"] = None

                # Contrastive AUC
                if "contrastive_halluc_score" in subset.columns:
                    valid = subset["contrastive_halluc_score"].notna()
                    if valid.sum() > 50:
                        try:
                            row["Contrastive AUC"] = roc_auc_score(
                                labels[valid.values],
                                subset.loc[valid, "contrastive_halluc_score"].values,
                            )
                        except Exception:
                            row["Contrastive AUC"] = None

                auc_rows.append(row)

            if auc_rows:
                df_auc = pd.DataFrame(auc_rows)
                st.dataframe(df_auc.round(4), use_container_width=True)

                # Bar chart
                df_melt = df_auc.melt(
                    id_vars="Split",
                    value_vars=[c for c in df_auc.columns if "AUC" in c],
                    var_name="Method", value_name="AUC",
                )
                fig = px.bar(
                    df_melt, x="Split", y="AUC", color="Method",
                    barmode="group",
                    color_discrete_sequence=["#3498db", "#e67e22"],
                )
                fig.update_layout(height=350, yaxis_range=[0.4, 1.0])
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "VLM Hallucination Detection · MS AI Systems · University of Florida · "
    f"Results from: `{OUTPUT_DIR}`"
)
