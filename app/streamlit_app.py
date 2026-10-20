import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.load_model import load_toxicity_model
from src.prediction import (
    predict_toxicity,
    predict_batch_binary,
)
from src.utils import (
    LABELS,
    LABEL_NAMES,
    POLICY_PRESETS,
    OPTIMAL_THRESHOLDS,
    explain_text_attribution,
)

# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# MODEL CACHING (Streamlit Native Resource Cache)
# =========================================================

@st.cache_resource(show_spinner="Loading DistilBERT weights from Hugging Face...")
def get_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl = load_toxicity_model(device=dev)
    return tok, mdl, dev


tokenizer, model, device = get_model()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("🛡️ Moderation Controls")

policy_choice = st.sidebar.selectbox(
    "Policy Preset",
    options=list(POLICY_PRESETS.keys()) + ["Custom Thresholds"],
    index=0,
    help="Balanced optimizes F1 across rare and frequent classes. Strict maximizes detection recall.",
)

active_thresholds = {}

if policy_choice == "Custom Thresholds":
    with st.sidebar.expander("Adjust Category Thresholds", expanded=True):
        for label in LABELS:
            default_val = float(OPTIMAL_THRESHOLDS.get(label, 0.50))
            active_thresholds[label] = st.slider(
                LABEL_NAMES[label],
                min_value=0.05,
                max_value=0.95,
                value=default_val,
                step=0.05,
            )
else:
    active_thresholds = POLICY_PRESETS[policy_choice].copy()

st.sidebar.divider()
st.sidebar.caption(f"Compute Backend: **{'CUDA GPU' if torch.cuda.is_available() else 'CPU'}**")
st.sidebar.caption("Backbone: **DistilBERT (66M)**")

# =========================================================
# HEADER
# =========================================================

st.title("Comment Toxicity Detector")
st.caption("Automated multi-label content moderation and explainable toxicity analysis powered by DistilBERT")

# =========================================================
# TABS
# =========================================================

tab_single, tab_batch, tab_benchmarks, tab_info = st.tabs(
    [
        "Single Comment",
        "Batch CSV Moderation",
        "Model Benchmarks",
        "Dataset & Details",
    ]
)

# =========================================================
# TAB 1 - SINGLE COMMENT ANALYSIS
# =========================================================

SAMPLE_COMMENTS = {
    "Constructive Feedback": "Thank you for the detailed feedback. I really appreciate your patience.",
    "Mild Insult": "You are completely useless and have no idea what you are doing.",
    "Obscene & Abusive": "Shut your mouth, you disgusting piece of trash.",
    "Violent Threat": "I will find you and beat you up if you post this again.",
    "Neutral Identity Mention": "I am a proud Muslim student organizing a community food bank.",
}

with tab_single:
    col_input, col_sample = st.columns([3, 2])
    with col_sample:
        selected_sample = st.selectbox(
            "Quick Example",
            options=["-- Select a sample comment --"] + list(SAMPLE_COMMENTS.keys()),
        )

    initial_text = SAMPLE_COMMENTS[selected_sample] if selected_sample in SAMPLE_COMMENTS else ""

    user_comment = st.text_area(
        "Comment Text",
        value=initial_text,
        height=120,
        placeholder="Enter or paste a comment here...",
    )

    analyze_clicked = st.button("Analyze Comment", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_comment.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing comment..."):
                results = predict_toxicity(
                    user_comment,
                    threshold=active_thresholds,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )

                # Determine top category automatically for explainability
                top_label = max(LABELS, key=lambda l: results[l]["probability"])
                top_prob = results[top_label]["probability"]
                top_idx = LABELS.index(top_label)

                # Compute word attribution automatically
                highlighted_text, word_attributions = explain_text_attribution(
                    user_comment,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    target_label_idx=top_idx,
                )

            summary = results["_summary"]
            sev_score = summary["severity_score"]
            sev_lvl = summary["severity_level"]
            is_toxic = summary["is_toxic"]

            # 1. Clean Summary Banner
            if sev_score >= 70 or results["threat"]["prediction"] or results["severe_toxic"]["prediction"]:
                st.error(f"High Risk Detected: {sev_lvl} (Severity Score: {sev_score}/100)")
            elif is_toxic:
                st.warning(f"Toxic Content Detected: {sev_lvl} (Severity Score: {sev_score}/100)")
            else:
                st.success(f"Clean Comment: {sev_lvl} (Severity Score: {sev_score}/100)")

            # 2. Key Metrics Card
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Toxicity Severity Index", f"{sev_score} / 100")
                with m2:
                    flag_count = sum(1 for l in LABELS if results[l]["prediction"])
                    st.metric("Violations Flagged", f"{flag_count} of 6")
                with m3:
                    st.metric("Primary Driver", f"{LABEL_NAMES[top_label]} ({top_prob:.1%})")

            # 3. Automatic Word Attribution Card (No manual category selector needed!)
            if is_toxic or top_prob >= 0.20:
                with st.container(border=True):
                    st.subheader("Word Attribution Analysis")
                    st.caption(
                        f"Automatically explaining the primary trigger: **{LABEL_NAMES[top_label]}**. "
                        "Key drivers are highlighted in :red[**red**] and :orange[**orange**]."
                    )
                    st.markdown(highlighted_text)

            # 4. Clean Category Breakdown Table
            with st.container(border=True):
                st.subheader("Category Breakdown")
                breakdown_rows = [
                    {
                        "Category": LABEL_NAMES[l],
                        "Probability": results[l]["probability"],
                        "Threshold": results[l]["threshold"],
                        "Status": "FLAGGED ⚠️" if results[l]["prediction"] else "CLEAN ✅",
                    }
                    for l in LABELS
                ]
                breakdown_df = pd.DataFrame(breakdown_rows)

                st.dataframe(
                    breakdown_df,
                    column_config={
                        "Category": st.column_config.TextColumn("Category"),
                        "Probability": st.column_config.ProgressColumn(
                            "Probability",
                            format="%.1f%%",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Threshold": st.column_config.NumberColumn(
                            "Threshold",
                            format="%.2f",
                        ),
                        "Status": st.column_config.TextColumn("Status"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            # 5. Export JSON
            st.download_button(
                "Download JSON Report",
                data=json.dumps(results, indent=2),
                file_name="toxicity_report.json",
                mime="application/json",
            )

# =========================================================
# TAB 2 - BATCH CSV MODERATION
# =========================================================

with tab_batch:
    st.header("Batch CSV Moderation")
    st.write("Upload a CSV file containing a `comment_text` column for batch evaluation.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "comment_text" not in df.columns:
                st.error("The CSV must contain a `comment_text` column.")
            elif df.empty:
                st.warning("The uploaded file is empty.")
            else:
                with st.container(border=True):
                    b1, b2 = st.columns(2)
                    b1.metric("Total Comments", f"{len(df):,}")
                    b2.metric("Columns", len(df.columns))

                if st.button("Process Batch", type="primary", use_container_width=True):
                    raw_texts = df["comment_text"].fillna("").astype(str).tolist()
                    progress_bar = st.progress(0.0)
                    chunk_size = 100
                    all_preds = []

                    with st.spinner("Classifying comments..."):
                        for i in range(0, len(raw_texts), chunk_size):
                            chunk = raw_texts[i:i + chunk_size]
                            preds = predict_batch_binary(
                                chunk,
                                threshold=active_thresholds,
                                batch_size=32,
                                tokenizer=tokenizer,
                                model=model,
                                device=device,
                            )
                            all_preds.extend(preds)
                            progress_bar.progress(min(1.0, (i + len(chunk)) / len(raw_texts)))

                    result_df = df.copy()
                    for idx, label in enumerate(LABELS):
                        result_df[label] = [r[idx] for r in all_preds]
                    result_df["is_toxic"] = result_df[LABELS].max(axis=1)

                    toxic_count = int(result_df["is_toxic"].sum())
                    clean_count = len(result_df) - toxic_count
                    toxic_pct = (toxic_count / len(result_df)) * 100

                    st.success("Batch processing complete.")

                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Processed", f"{len(result_df):,}")
                        c2.metric("Toxic Flagged", f"{toxic_count:,}")
                        c3.metric("Clean", f"{clean_count:,}")
                        c4.metric("Toxicity Rate", f"{toxic_pct:.1f}%")

                    # Frequency chart
                    cat_totals = {
                        LABEL_NAMES[l]: int(result_df[l].sum())
                        for l in LABELS
                    }
                    freq_df = pd.DataFrame(
                        {"Category": list(cat_totals.keys()), "Flagged Count": list(cat_totals.values())}
                    )
                    st.subheader("Violations by Category")
                    st.bar_chart(freq_df, x="Category", y="Flagged Count", use_container_width=True)

                    # Filter
                    st.subheader("Filter Results")
                    selected_filter = st.multiselect(
                        "Filter by category:",
                        options=LABELS,
                        format_func=lambda x: LABEL_NAMES[x],
                    )

                    filtered_view = result_df
                    if selected_filter:
                        for sf in selected_filter:
                            filtered_view = filtered_view[filtered_view[sf] == 1]

                    st.dataframe(filtered_view, hide_index=True, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Moderated CSV",
                        data=csv_bytes,
                        file_name="moderated_results.csv",
                        mime="text/csv",
                        type="primary",
                    )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# =========================================================
# TAB 3 - MODEL BENCHMARKS
# =========================================================

with tab_benchmarks:
    st.header("Model Benchmarks & Comparison")
    st.write("Validation metrics on the Jigsaw Toxic Comment dataset across all trained architectures.")

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("DistilBERT ROC-AUC", "98.83%")
        m2.metric("Micro F1", "77.72%")
        m3.metric("Micro Precision", "83.75%")
        m4.metric("Hamming Loss", "1.50%")

    st.subheader("Architecture Comparison Matrix")
    comparison_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "BiLSTM",
                "GloVe + BiLSTM",
                "DistilBERT",
            ],
            "Hamming Loss": [0.0285, 0.0173, 0.0169, 0.0150],
            "Micro Precision": [0.5700, 0.8227, 0.7944, 0.8375],
            "Micro Recall": [0.8647, 0.6662, 0.7202, 0.7250],
            "Micro F1": [0.6871, 0.7363, 0.7555, 0.7772],
            "Macro F1": [0.5629, 0.4437, 0.5879, 0.5885],
            "ROC AUC": [0.9806, 0.9752, 0.9839, 0.9883],
        }
    )
    st.dataframe(comparison_data, hide_index=True, use_container_width=True)

    st.subheader("Per-Label F1 Score Comparison")
    f1_df = pd.DataFrame(
        {
            "Category": ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"],
            "BiLSTM": [0.79, 0.37, 0.81, 0.00, 0.69, 0.00],
            "GloVe + BiLSTM": [0.79, 0.24, 0.81, 0.42, 0.74, 0.54],
            "DistilBERT": [0.82, 0.49, 0.84, 0.53, 0.75, 0.61],
        }
    )
    st.dataframe(f1_df, hide_index=True, use_container_width=True)
    st.bar_chart(f1_df, x="Category", y=["BiLSTM", "GloVe + BiLSTM", "DistilBERT"], use_container_width=True)

    st.info(
        "Minority class challenge: Rare categories like Threat (0.3% base rate) suffer from low recall "
        "when using a flat 0.5 threshold. Calibrated thresholds (e.g. 0.20 for Threat) recover missed violations."
    )

# =========================================================
# TAB 4 - DATASET & DETAILS
# =========================================================

with tab_info:
    st.header("Dataset & Pipeline Details")

    with st.container(border=True):
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Samples", "159,571")
        d2.metric("Toxic Samples", "16,225 (10.2%)")
        d3.metric("Non-Toxic Samples", "143,346 (89.8%)")

    st.subheader("Label Distribution")
    label_dist_df = pd.DataFrame(
        {
            "Category": [
                "Toxic",
                "Severe Toxic",
                "Obscene",
                "Threat",
                "Insult",
                "Identity Hate",
            ],
            "Samples": [15294, 1595, 8449, 478, 7877, 1405],
            "Corpus Frequency": ["9.58%", "1.00%", "5.29%", "0.30%", "4.94%", "0.88%"],
        }
    )
    st.dataframe(label_dist_df, hide_index=True, use_container_width=True)
    st.bar_chart(label_dist_df, x="Category", y="Samples", use_container_width=True)

    st.divider()
    st.subheader("Model Artifact")
    st.write(
        "Fine-tuned DistilBERT model weights and tokenizer configurations are hosted on Hugging Face Hub:"
    )
    st.caption("https://huggingface.co/gopinath-sara1n/comment-toxicity-distilbert")

st.divider()
st.caption("Comment Toxicity Detection System | DistilBERT | Streamlit Native")