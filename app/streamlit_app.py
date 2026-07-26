import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.prediction import (
    predict_toxicity,
    predict_batch_binary,
)


# =========================================================
# CONSTANTS
# =========================================================

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Comment Toxicity Detector")

st.caption(
    "Multi-label toxic comment classification powered by DistilBERT"
)


# =========================================================
# TABS
# =========================================================

tab_analyze, tab_batch, tab_performance, tab_dataset, tab_about = st.tabs(
    [
        "Analyze",
        "Batch Analysis",
        "Model Performance",
        "Dataset",
        "About",
    ]
)


# =========================================================
# TAB 1 - ANALYZE COMMENT
# =========================================================

with tab_analyze:

    st.header("Analyze a Comment")

    st.write(
        "Enter a comment below to detect toxicity across "
        "six different categories."
    )

    comment = st.text_area(
        "Comment",
        height=150,
        placeholder="Type or paste a comment here...",
    )

    if st.button(
        "Analyze Comment",
        type="primary",
        use_container_width=True,
    ):

        if not comment.strip():

            st.warning("Please enter a comment before analyzing.")

        else:

            with st.spinner("Analyzing comment..."):
                results = predict_toxicity(comment)

            toxic_detected = any(
                result["prediction"]
                for result in results.values()
            )

            st.divider()

            if toxic_detected:

                st.error(
                    "Potential toxic content detected. "
                    "One or more toxicity categories exceeded "
                    "the classification threshold."
                )

            else:

                st.success(
                    "No toxic content detected. "
                    "All categories are below the "
                    "classification threshold."
                )

            st.subheader("Category Probabilities")

            for label in LABELS:

                result = results[label]

                probability = float(
                    result["probability"]
                )

                prediction = result["prediction"]

                col1, col2, col3 = st.columns(
                    [2, 5, 1]
                )

                with col1:

                    if prediction:
                        st.write(
                            f"**{LABEL_NAMES[label]} ⚠️**"
                        )
                    else:
                        st.write(
                            f"**{LABEL_NAMES[label]}**"
                        )

                with col2:

                    st.progress(
                        min(
                            max(probability, 0.0),
                            1.0,
                        )
                    )

                with col3:

                    st.write(
                        f"**{probability:.1%}**"
                    )


    # =====================================================
    # SAMPLE TEST
    # =====================================================

    st.divider()

    st.subheader("Try an Example")

    sample_comments = {
        "Normal Comment":
            "Thank you for your help. I really appreciate it.",

        "Insult":
            "You are a disgusting idiot.",

        "Toxic Comment":
            "I hate you and you are completely useless.",

        "Threat":
            "I will hurt you if you do that again.",
    }

    sample_type = st.selectbox(
        "Select example",
        list(sample_comments.keys()),
    )

    selected_sample = sample_comments[sample_type]

    st.info(selected_sample)

    if st.button("Analyze Example"):

        with st.spinner("Analyzing example..."):

            sample_result = predict_toxicity(
                selected_sample
            )

        sample_data = []

        for label in LABELS:

            probability = float(
                sample_result[label]["probability"]
            )

            prediction = sample_result[label][
                "prediction"
            ]

            sample_data.append(
                {
                    "Category": LABEL_NAMES[label],
                    "Probability": probability,
                    "Detected": (
                        "Yes"
                        if prediction
                        else "No"
                    ),
                }
            )

        sample_df = pd.DataFrame(
            sample_data
        )

        st.subheader("Prediction Results")

        st.dataframe(
            sample_df,
            column_config={
                "Probability":
                    st.column_config.ProgressColumn(
                        "Probability",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
            },
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# TAB 2 - BATCH ANALYSIS
# =========================================================

with tab_batch:

    st.header("Batch Comment Analysis")

    st.write(
        "Upload a CSV containing a `comment_text` column. "
        "Maximum 1,000 comments."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            if "comment_text" not in df.columns:

                st.error(
                    "CSV must contain a "
                    "`comment_text` column."
                )

            elif df.empty:

                st.warning(
                    "The uploaded CSV is empty."
                )

            elif len(df) > 1000:

                st.error(
                    f"The file contains "
                    f"{len(df):,} comments. "
                    f"Maximum allowed is 1,000."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Comments",
                        f"{len(df):,}",
                    )

                with col2:

                    st.metric(
                        "Columns",
                        len(df.columns),
                    )

                with st.expander(
                    "Preview Uploaded Data",
                    expanded=True,
                ):

                    st.dataframe(
                        df.head(10),
                        hide_index=True,
                        use_container_width=True,
                    )

                if st.button(
                    "Analyze CSV",
                    type="primary",
                    use_container_width=True,
                ):

                    texts = (
                        df["comment_text"]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                    with st.spinner(
                        f"Analyzing "
                        f"{len(texts):,} comments..."
                    ):

                        predictions = (
                            predict_batch_binary(
                                texts,
                                threshold=0.5,
                                batch_size=32,
                            )
                        )

                    result_df = df.copy()

                    for index, label in enumerate(
                        LABELS
                    ):

                        result_df[label] = [
                            row[index]
                            for row in predictions
                        ]

                    result_df["is_toxic"] = (
                        result_df[LABELS]
                        .max(axis=1)
                        .astype(int)
                    )

                    toxic_count = int(
                        result_df[
                            "is_toxic"
                        ].sum()
                    )

                    clean_count = (
                        len(result_df)
                        - toxic_count
                    )

                    st.success(
                        "Analysis completed successfully."
                    )

                    st.subheader("Analysis Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Total Comments",
                            f"{len(result_df):,}",
                        )

                    with col2:

                        st.metric(
                            "Toxic",
                            f"{toxic_count:,}",
                        )

                    with col3:

                        st.metric(
                            "Non-Toxic",
                            f"{clean_count:,}",
                        )

                    st.subheader(
                        "Classification Results"
                    )

                    st.dataframe(
                        result_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                    csv_data = (
                        result_df
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        "Download Predictions",
                        data=csv_data,
                        file_name=(
                            "toxicity_predictions.csv"
                        ),
                        mime="text/csv",
                        type="primary",
                    )

        except Exception as e:

            st.error(
                f"Unable to read CSV: {e}"
            )


# =========================================================
# TAB 3 - MODEL PERFORMANCE
# =========================================================

with tab_performance:

    st.header("Model Performance")

    st.write(
        "Final DistilBERT validation performance."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ROC-AUC",
            "98.83%",
        )

    with col2:
        st.metric(
            "Micro F1",
            "77.72%",
        )

    with col3:
        st.metric(
            "Hamming Loss",
            "1.50%",
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Micro Precision",
            "83.75%",
        )

    with col5:
        st.metric(
            "Micro Recall",
            "72.50%",
        )

    with col6:
        st.metric(
            "Macro F1",
            "58.85%",
        )

    st.divider()

    st.subheader("Model Comparison")

    comparison_data = {
        "Model": [
            "Logistic Regression",
            "BiLSTM",
            "GloVe + BiLSTM",
            "DistilBERT",
        ],
        "Hamming Loss": [
            0.028481,
            0.017262,
            0.016860,
            0.015032,
        ],
        "Micro Precision": [
            0.570028,
            0.822722,
            0.794361,
            0.837504,
        ],
        "Micro Recall": [
            0.864674,
            0.666233,
            0.720248,
            0.725014,
        ],
        "Micro F1": [
            0.687095,
            0.736254,
            0.755492,
            0.777210,
        ],
        "Macro F1": [
            0.562904,
            0.443658,
            0.587926,
            0.588496,
        ],
        "ROC AUC": [
            0.980627,
            0.975242,
            0.983861,
            0.988290,
        ],
    }

    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "DistilBERT was selected as the final deployment "
        "model because it achieved the highest overall "
        "ROC-AUC and Micro F1."
    )


# =========================================================
# TAB 4 - DATASET
# =========================================================

with tab_dataset:

    st.header("Dataset Insights")

    st.write(
        "The model was trained using the Jigsaw "
        "Toxic Comment Classification dataset."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Comments",
            "159,571",
        )

    with col2:

        st.metric(
            "Toxic Comments",
            "16,225",
        )

    with col3:

        st.metric(
            "Non-Toxic Comments",
            "143,346",
        )

    st.divider()

    st.subheader(
        "Toxicity Label Distribution"
    )

    label_data = pd.DataFrame(
        {
            "Category": [
                "Toxic",
                "Severe Toxic",
                "Obscene",
                "Threat",
                "Insult",
                "Identity Hate",
            ],
            "Count": [
                15294,
                1595,
                8449,
                478,
                7877,
                1405,
            ],
        }
    )

    st.bar_chart(
        label_data,
        x="Category",
        y="Count",
        use_container_width=True,
    )

    with st.expander(
        "More Dataset Statistics"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Training samples:** 159,571"
            )

            st.write(
                "**Toxic samples:** 16,225"
            )

            st.write(
                "**Non-toxic samples:** 143,346"
            )

            st.write(
                "**Toxicity labels:** 6"
            )

        with col2:

            st.write(
                "**Average comment length:** "
                "~394 characters"
            )

            st.write(
                "**Unique words:** ~194,792"
            )

            st.write(
                "**Non-toxic comments:** ~90%"
            )

            st.write(
                "**Task type:** Multi-label classification"
            )


# =========================================================
# TAB 5 - ABOUT
# =========================================================

with tab_about:

    st.header("About the Model")

    st.subheader(
        "DistilBERT Multi-Label Classifier"
    )

    st.write(
        "This application uses a fine-tuned DistilBERT "
        "transformer model to identify toxic content "
        "in text comments."
    )

    st.write(
        "Because this is a multi-label classification "
        "problem, a single comment can belong to more "
        "than one toxicity category."
    )

    st.subheader("Toxicity Categories")

    st.write(
        """
        - Toxic
        - Severe Toxic
        - Obscene
        - Threat
        - Insult
        - Identity Hate
        """
    )

    st.divider()

    st.subheader("How It Works")

    st.write(
        "1. The user enters a comment."
    )

    st.write(
        "2. The text is preprocessed and tokenized."
    )

    st.write(
        "3. DistilBERT processes the comment."
    )

    st.write(
        "4. The model generates a probability "
        "for each toxicity category."
    )

    st.write(
        "5. Categories exceeding the threshold "
        "are classified as detected."
    )

    st.divider()

    st.subheader("Final Model Details")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Architecture",
            value="DistilBERT",
        )

        st.metric(
            label="Task",
            value="Multi-label Classification",
        )

        st.metric(
            label="Number of Labels",
            value="6",
        )


    with col2:

        st.metric(
            label="ROC-AUC",
            value="0.9883",
        )

        st.metric(
            label="Micro F1",
            value="0.7772",
        )

        st.metric(
            label="Hamming Loss",
            value="0.0150",
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Comment Toxicity Detection | DistilBERT | NLP Project"
)