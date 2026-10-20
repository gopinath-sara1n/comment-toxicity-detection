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
# PAGE CONFIGURATION (Streamlit Native)
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

@st.cache_resource(show_spinner="Loading fine-tuned DistilBERT model...")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_toxicity_model(device=device)
    return tokenizer, model, device


tokenizer, model, device = get_model()


# =========================================================
# SIDEBAR: MODERATION CONTROLS & THRESHOLDS
# =========================================================

st.sidebar.title("⚙️ Moderation Controls")
st.sidebar.caption("Calibrate sensitivity and classification thresholds")

policy_choice = st.sidebar.selectbox(
    "Moderation Policy Preset",
    options=list(POLICY_PRESETS.keys()) + ["Custom Thresholds"],
    index=0,
    help=(
        "Balanced optimizes per-class F1 for imbalanced categories. "
        "Strict minimizes missed violations. Relaxed minimizes false alarms."
    ),
)

active_thresholds = {}

if policy_choice == "Custom Thresholds":
    st.sidebar.subheader("Per-Category Sensitivity")
    for label in LABELS:
        default_val = float(OPTIMAL_THRESHOLDS.get(label, 0.50))
        active_thresholds[label] = st.sidebar.slider(
            f"{LABEL_NAMES[label]} Threshold",
            min_value=0.05,
            max_value=0.95,
            value=default_val,
            step=0.05,
            help=f"Comments with {LABEL_NAMES[label]} probability above this value are flagged.",
        )
else:
    active_thresholds = POLICY_PRESETS[policy_choice].copy()

with st.sidebar.expander("Active Thresholds", expanded=False):
    thresh_df = pd.DataFrame(
        [
            {"Category": LABEL_NAMES[l], "Threshold": f"{active_thresholds[l]:.2f}"}
            for l in LABELS
        ]
    )
    st.dataframe(thresh_df, hide_index=True, use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("System Telemetry")
st.sidebar.text(f"Hardware: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
st.sidebar.text("Model: DistilBERT Multi-Label")
st.sidebar.text("Hugging Face: comment-toxicity-distilbert")


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Comment Toxicity Detector")
st.caption(
    "Multi-label toxicity classification & explainable content moderation powered by DistilBERT"
)


# =========================================================
# MAIN NAVIGATION TABS
# =========================================================

tab_analyze, tab_batch, tab_explain, tab_performance, tab_dataset, tab_about = st.tabs(
    [
        "Analyze Comment",
        "Batch CSV Analysis",
        "Explainability (XAI)",
        "Model Benchmarks",
        "Dataset Insights",
        "Architecture",
    ]
)


# =========================================================
# TAB 1 - ANALYZE SINGLE COMMENT
# =========================================================

with tab_analyze:
    st.header("Analyze a Comment")
    st.write(
        "Enter comment text below to evaluate toxicity probabilities across "
        "all six categories using the active moderation policy."
    )

    user_comment = st.text_area(
        "Comment Text",
        height=140,
        placeholder="Type or paste a comment here to test...",
    )

    col_btn, col_target = st.columns([2, 2])
    with col_btn:
        analyze_clicked = st.button("Analyze Comment", type="primary", use_container_width=True)
    with col_target:
        target_explain_label = st.selectbox(
            "Primary category for word attribution:",
            options=LABELS,
            format_func=lambda x: LABEL_NAMES[x],
            index=0,
            help="Select which category you want to highlight key contributing words for.",
        )

    if analyze_clicked:
        if not user_comment.strip():
            st.warning("Please enter a non-empty comment to analyze.")
        else:
            with st.spinner("Classifying and computing token attribution..."):
                results = predict_toxicity(
                    user_comment,
                    threshold=active_thresholds,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
                highlighted_text, word_attributions = explain_text_attribution(
                    user_comment,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    target_label_idx=LABELS.index(target_explain_label),
                )

            summary = results["_summary"]
            sev_score = summary["severity_score"]
            sev_lvl = summary["severity_level"]
            is_toxic = summary["is_toxic"]

            st.divider()

            # Status Banner
            if sev_score >= 75 or results["threat"]["prediction"] or results["severe_toxic"]["prediction"]:
                st.error(
                    f"⚠️ High Hazard Detected: {sev_lvl} (Severity Score: {sev_score}/100). "
                    "This comment exceeds critical safety thresholds."
                )
            elif is_toxic:
                st.warning(
                    f"⚠️ Toxicity Flagged: {sev_lvl} (Severity Score: {sev_score}/100). "
                    "One or more categories exceeded the active moderation threshold."
                )
            else:
                st.success(
                    f"✅ Safe Comment: {sev_lvl} (Severity Score: {sev_score}/100). "
                    "All categories are within acceptable bounds."
                )

            # High-level Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Severity Index", f"{sev_score} / 100")
            with col_m2:
                flagged_count = sum(1 for l in LABELS if results[l]["prediction"])
                st.metric("Triggered Categories", f"{flagged_count} of 6")
            with col_m3:
                highest_label = max(LABELS, key=lambda l: results[l]["probability"])
                st.metric(
                    "Primary Driver",
                    f"{LABEL_NAMES[highest_label]} ({results[highest_label]['probability']:.1%})",
                )

            # Category Breakdown
            st.subheader("Category Breakdown")
            for label in LABELS:
                prob = results[label]["probability"]
                thresh = results[label]["threshold"]
                detected = results[label]["prediction"]

                col_label, col_bar, col_prob, col_badge = st.columns([2, 5, 1, 2])
                with col_label:
                    if detected:
                        st.write(f"**:red[{LABEL_NAMES[label]}] ⚠️**")
                    else:
                        st.write(f"**{LABEL_NAMES[label]}**")
                with col_bar:
                    st.progress(min(max(prob, 0.0), 1.0))
                with col_prob:
                    st.write(f"**{prob:.1%}**")
                with col_badge:
                    if detected:
                        st.write(f":red[FLAGGED] (>{thresh:.2f})")
                    else:
                        st.write(f":green[CLEAN] (<{thresh:.2f})")

            # Native Streamlit Explainability
            st.divider()
            st.subheader("🔍 Explainable AI: Contributing Words")
            st.write(
                f"Word-level attribution for **{LABEL_NAMES[target_explain_label]}**. "
                "Words colored in :red[**red**] indicate primary toxicity drivers; "
                ":orange[**orange**] indicates secondary contributing words."
            )
            st.markdown(highlighted_text)

            if word_attributions:
                with st.expander("View Token Saliency Scores", expanded=False):
                    attr_df = pd.DataFrame(word_attributions)
                    st.dataframe(
                        attr_df,
                        column_config={
                            "Attribution": st.column_config.ProgressColumn(
                                "Attribution Score",
                                min_value=0.0,
                                max_value=1.0,
                                format="%.2f",
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

            # Export Analysis
            json_report = json.dumps(results, indent=2)
            st.download_button(
                "Download Analysis Report (JSON)",
                data=json_report,
                file_name="comment_toxicity_analysis.json",
                mime="application/json",
            )

    # Example Comments Sandbox
    st.divider()
    st.subheader("Try Preset Examples")
    sample_comments = {
        "Clean Discussion": "Thank you for the constructive feedback, I really appreciate your help.",
        "Mild Insult": "You are completely incompetent and have no idea what you are doing.",
        "Obscene & Toxic": "Shut up and get out of here, you absolute idiot.",
        "Explicit Threat": "I will find you and hurt you if you show up here again.",
        "Identity Bias Test": "I am a proud Muslim woman sharing our community food drive.",
    }

    selected_sample_title = st.selectbox(
        "Choose an example comment:",
        options=list(sample_comments.keys()),
    )
    sample_body = sample_comments[selected_sample_title]
    st.info(sample_body)

    if st.button("Evaluate Preset Example"):
        sample_results = predict_toxicity(
            sample_body,
            threshold=active_thresholds,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        sample_rows = [
            {
                "Category": LABEL_NAMES[l],
                "Probability": sample_results[l]["probability"],
                "Threshold": sample_results[l]["threshold"],
                "Status": "FLAGGED" if sample_results[l]["prediction"] else "CLEAN",
            }
            for l in LABELS
        ]
        sample_df = pd.DataFrame(sample_rows)
        st.dataframe(
            sample_df,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# TAB 2 - BATCH CSV ANALYSIS
# =========================================================

with tab_batch:
    st.header("Batch CSV Comment Moderation")
    st.write(
        "Upload a CSV dataset containing a `comment_text` column. "
        "The model will process comments in vectorized batches using the active moderation policy."
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "comment_text" not in df.columns:
                st.error("The uploaded CSV must contain a `comment_text` column.")
            elif df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col_b2:
                    st.metric("Columns", len(df.columns))

                with st.expander("Preview Uploaded Data", expanded=True):
                    st.dataframe(df.head(5), hide_index=True, use_container_width=True)

                if st.button("Run Batch Moderation", type="primary", use_container_width=True):
                    raw_texts = df["comment_text"].fillna("").astype(str).tolist()

                    progress_bar = st.progress(0.0)
                    chunk_size = 100
                    total_chunks = (len(raw_texts) + chunk_size - 1) // chunk_size
                    all_binary_preds = []

                    with st.spinner(f"Analyzing {len(raw_texts):,} comments in batches..."):
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
                            all_binary_preds.extend(preds)
                            progress_bar.progress(min(1.0, (i + len(chunk)) / len(raw_texts)))

                    result_df = df.copy()
                    for idx, label in enumerate(LABELS):
                        result_df[label] = [row[idx] for row in all_binary_preds]

                    result_df["is_toxic"] = result_df[LABELS].max(axis=1)

                    toxic_total = int(result_df["is_toxic"].sum())
                    clean_total = len(result_df) - toxic_total
                    toxic_pct = (toxic_total / len(result_df)) * 100

                    st.success("Batch moderation completed successfully!")

                    # Summary Metrics
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.metric("Total Analyzed", f"{len(result_df):,}")
                    with col_s2:
                        st.metric("Flagged Toxic", f"{toxic_total:,}")
                    with col_s3:
                        st.metric("Clean Comments", f"{clean_total:,}")
                    with col_s4:
                        st.metric("Toxicity Rate", f"{toxic_pct:.1f}%")

                    # Breakdown chart
                    cat_counts = {
                        LABEL_NAMES[l]: int(result_df[l].sum())
                        for l in LABELS
                    }
                    chart_data = pd.DataFrame(
                        {"Category": list(cat_counts.keys()), "Flags": list(cat_counts.values())}
                    )
                    st.subheader("Category Violation Frequency")
                    st.bar_chart(chart_data, x="Category", y="Flags", use_container_width=True)

                    # Interactive Filter
                    st.subheader("Filter and Inspect Results")
                    selected_filters = st.multiselect(
                        "Filter comments flagged for specific categories:",
                        options=LABELS,
                        format_func=lambda x: LABEL_NAMES[x],
                    )

                    filtered_df = result_df
                    if selected_filters:
                        for sf in selected_filters:
                            filtered_df = filtered_df[filtered_df[sf] == 1]

                    st.dataframe(filtered_df, hide_index=True, use_container_width=True)

                    # Download
                    csv_export = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Full Results CSV",
                        data=csv_export,
                        file_name="moderated_comments.csv",
                        mime="text/csv",
                        type="primary",
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# =========================================================
# TAB 3 - EXPLAINABILITY EXPLORER (XAI)
# =========================================================

with tab_explain:
    st.header("Explainable AI (XAI) Attribution Explorer")
    st.write(
        "Inspect the exact contribution of each word toward individual toxicity categories. "
        "This uses model embedding gradients to explain transformer decisions transparently."
    )

    explain_input = st.text_area(
        "Enter comment to explain:",
        value="I hate you and you are a violent threat",
        height=100,
    )

    col_e1, col_e2 = st.columns([2, 1])
    with col_e1:
        chosen_cat = st.selectbox(
            "Target category to explain:",
            options=LABELS,
            format_func=lambda x: LABEL_NAMES[x],
            index=0,
            key="xai_cat_select",
        )
    with col_e2:
        explain_btn = st.button("Generate Attribution", type="primary", use_container_width=True)

    if explain_btn and explain_input.strip():
        with st.spinner("Computing token gradients..."):
            cat_idx = LABELS.index(chosen_cat)
            highlighted, tokens_df_data = explain_text_attribution(
                explain_input,
                tokenizer=tokenizer,
                model=model,
                device=device,
                target_label_idx=cat_idx,
            )

        st.subheader("Highlighted Text")
        st.markdown(highlighted)

        if tokens_df_data:
            st.subheader("Word Attribution Bar Chart")
            bar_df = pd.DataFrame(tokens_df_data)
            st.bar_chart(bar_df, x="Word", y="Attribution", use_container_width=True)


# =========================================================
# TAB 4 - MODEL BENCHMARKS & COMPARISON
# =========================================================

with tab_performance:
    st.header("Model Performance & Comparison")
    st.write(
        "Comprehensive validation benchmark across traditional ML baselines, deep learning, "
        "and transformer architectures on the Jigsaw dataset."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("ROC-AUC", "98.83%")
    with col_p2:
        st.metric("Micro F1", "77.72%")
    with col_p3:
        st.metric("Hamming Loss", "1.50%")

    col_p4, col_p5, col_p6 = st.columns(3)
    with col_p4:
        st.metric("Micro Precision", "83.75%")
    with col_p5:
        st.metric("Micro Recall", "72.50%")
    with col_p6:
        st.metric("Macro F1", "58.85%")

    st.divider()
    st.subheader("Model Comparison Matrix")
    comparison_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "BiLSTM",
                "GloVe + BiLSTM",
                "DistilBERT",
            ],
            "Hamming Loss": [0.028481, 0.017262, 0.016860, 0.015032],
            "Micro Precision": [0.570028, 0.822722, 0.794361, 0.837504],
            "Micro Recall": [0.864674, 0.666233, 0.720248, 0.725014],
            "Micro F1": [0.687095, 0.736254, 0.755492, 0.777210],
            "Macro F1": [0.562904, 0.443658, 0.587926, 0.588496],
            "ROC AUC": [0.980627, 0.975242, 0.983861, 0.988290],
        }
    )
    st.dataframe(comparison_data, hide_index=True, use_container_width=True)

    st.subheader("Category-wise F1 Scores")
    f1_comparison = pd.DataFrame(
        {
            "Category": ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"],
            "BiLSTM": [0.79, 0.37, 0.81, 0.00, 0.69, 0.00],
            "GloVe + BiLSTM": [0.79, 0.24, 0.81, 0.42, 0.74, 0.54],
            "DistilBERT": [0.82, 0.49, 0.84, 0.53, 0.75, 0.61],
        }
    )
    st.dataframe(f1_comparison, hide_index=True, use_container_width=True)
    st.bar_chart(f1_comparison, x="Category", y=["BiLSTM", "GloVe + BiLSTM", "DistilBERT"], use_container_width=True)

    st.info(
        "💡 Insight: In extreme class imbalance (e.g. Threat has only 478 examples), "
        "a flat 0.5 threshold suppresses rare-class recall. Using calibrated per-label "
        "thresholds restores balanced detection for minority hazards."
    )


# =========================================================
# TAB 5 - DATASET INSIGHTS
# =========================================================

with tab_dataset:
    st.header("Dataset Insights & Distribution")
    st.write("The models were trained on the Jigsaw Toxic Comment Classification challenge.")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Total Comments", "159,571")
    with col_d2:
        st.metric("Toxic Comments", "16,225 (10.2%)")
    with col_d3:
        st.metric("Clean Comments", "143,346 (89.8%)")

    st.divider()
    st.subheader("Training Label Counts")
    label_counts_df = pd.DataFrame(
        {
            "Category": [
                "Toxic",
                "Severe Toxic",
                "Obscene",
                "Threat",
                "Insult",
                "Identity Hate",
            ],
            "Count": [15294, 1595, 8449, 478, 7877, 1405],
            "Frequency": ["9.58%", "1.00%", "5.29%", "0.30%", "4.94%", "0.88%"],
        }
    )
    st.dataframe(label_counts_df, hide_index=True, use_container_width=True)
    st.bar_chart(label_counts_df, x="Category", y="Count", use_container_width=True)


# =========================================================
# TAB 6 - ARCHITECTURE & DETAILS
# =========================================================

with tab_about:
    st.header("Architecture & Pipeline")
    st.write(
        "This project implements a multi-label toxicity classification pipeline "
        "using Hugging Face Transformers and PyTorch."
    )

    st.subheader("Pipeline Steps")
    st.write("1. **Raw Text Sanitization:** Normalizes whitespace, expands contractions, cleans punctuation.")
    st.write("2. **WordPiece Tokenization:** DistilBERT subword tokenization preserving semantic context.")
    st.write("3. **Transformer Encoding:** 6-layer bidirectional transformer with multi-head self-attention.")
    st.write("4. **Multi-Label Sigmoid Head:** Independent binary outputs for each of the 6 toxicity classes.")
    st.write("5. **Calibrated Thresholding:** Dynamic per-category decision boundaries.")

    st.divider()
    st.subheader("Model Artifact")
    st.write("The fine-tuned model weights are hosted publicly on Hugging Face Hub:")
    st.caption("https://huggingface.co/gopinath-sara1n/comment-toxicity-distilbert")

st.divider()
st.caption("Comment Toxicity Detection | DistilBERT | Built with Streamlit")