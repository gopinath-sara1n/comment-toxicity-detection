# Comment Toxicity Detection using DistilBERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://comment-toxicity-detection-ktp7dqnn9m9ab3rkczjpgz.streamlit.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DistilBERT%20Model-yellow)](https://huggingface.co/gopinath-sara1n/comment-toxicity-distilbert)

A production-ready, multi-label Natural Language Processing (NLP) system for detecting toxic content in online comments using machine learning, deep learning, and transformer-based models.

The application features a fine-tuned **DistilBERT** model hosted on Hugging Face Hub, an interactive **Streamlit web application** with native explainable AI (XAI), and a high-performance **FastAPI microservice**.

---

## Live Demo

Try the deployed Streamlit web application:

👉 **[Launch Live Toxicity Detector](https://comment-toxicity-detection-ktp7dqnn9m9ab3rkczjpgz.streamlit.app/)**

---

## Project Overview

Online platforms receive large volumes of user-generated comments, some of which contain abusive, insulting, threatening, obscene, or hateful content.

This project develops a multi-label toxicity classification system capable of identifying six toxicity categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

A single comment may belong to more than one category.

---

## Dataset

The project uses the **Jigsaw Toxic Comment Classification** dataset.

### Training Dataset

- Total comments: 159,571
- Toxic comments: 16,225 (10.2%)
- Non-toxic comments: 143,346 (89.8%)
- Number of target labels: 6

### Label Distribution

| Category | Count | Frequency |
|---|---:|---:|
| Toxic | 15,294 | 9.58% |
| Severe Toxic | 1,595 | 1.00% |
| Obscene | 8,449 | 5.29% |
| Threat | 478 | 0.30% |
| Insult | 7,877 | 4.94% |
| Identity Hate | 1,405 | 0.88% |

The dataset is highly imbalanced, with approximately 90% of comments containing no toxicity label, and minority categories like Threat representing as little as 0.30% of the corpus.

---

## Models & Evaluation

Multiple approaches were explored before selecting the final model:

- Logistic Regression (TF-IDF baseline)
- BiLSTM (Keras baseline)
- GloVe + BiLSTM (Pre-trained embeddings)
- DistilBERT (Fine-tuned transformer)

### Benchmark Comparison

| Model | Hamming Loss | Micro Precision | Micro Recall | Micro F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.028481 | 0.570028 | 0.864674 | 0.687095 | 0.562904 | 0.980627 |
| BiLSTM | 0.017262 | 0.822722 | 0.666233 | 0.736254 | 0.443658 | 0.975242 |
| GloVe + BiLSTM | 0.016860 | 0.794361 | 0.720248 | 0.755492 | 0.587926 | 0.983861 |
| **DistilBERT** | **0.015032** | **0.837504** | **0.725014** | **0.777210** | **0.588496** | **0.988290** |

DistilBERT was selected as the primary deployment model because it achieved the highest ROC-AUC (0.9883), highest Micro F1 (0.7772), and lowest Hamming Loss (0.0150).

---

## Moderation Policy & Calibrated Thresholds

In severe class imbalance, a rigid flat threshold of `0.5` causes high false negative rates on minority classes like **Threat** and **Severe Toxic**. 

This system implements dynamic, calibrated decision thresholds:

| Category | Standard Baseline | Balanced (F1-Optimized) | Strict Moderation | Relaxed Policy |
|---|---:|---:|---:|---:|
| Toxic | 0.50 | 0.45 | 0.30 | 0.60 |
| Severe Toxic | 0.50 | 0.25 | 0.15 | 0.40 |
| Obscene | 0.50 | 0.45 | 0.30 | 0.60 |
| Threat | 0.50 | 0.20 | 0.10 | 0.35 |
| Insult | 0.50 | 0.40 | 0.25 | 0.55 |
| Identity Hate | 0.50 | 0.25 | 0.15 | 0.40 |

---

## Streamlit Application

An interactive Streamlit application built **strictly using native Streamlit components and documentation** (no raw HTML or custom CSS).

Try the live version: [**comment-toxicity-detection.streamlit.app**](https://comment-toxicity-detection-ktp7dqnn9m9ab3rkczjpgz.streamlit.app/)

### Features

- **Real-Time Comment Analysis:** Instant toxicity detection across all 6 categories.
- **Toxicity Severity Index:** Composite 0–100 risk scoring with hazard weighting.
- **Explainable AI (XAI):** Gradient-based word attribution automatically highlighting contributing words (`:red[...]` / `:orange[...]`).
- **Policy Sensitivity Controls:** Sidebar selector for presets (Balanced, Strict, Relaxed, Custom Sliders).
- **Batch CSV Moderation:** Vectorized chunked CSV processing with category filters and instant download.
- **Interactive Visualizations:** Native Streamlit bar charts and progress dataframes.
- **Performance & Benchmark Tables:** Side-by-side model and label comparison.

---

## FastAPI Production Microservice

A production-grade REST API is provided in `app/api.py`:

- `GET /`: Health and service information.
- `GET /policies`: Retrieve available moderation presets and active thresholds.
- `POST /predict`: Single comment inference with probability breakdown and severity score.
- `POST /predict/batch`: High-throughput batch inference returning detailed and binary prediction matrices.

Run the API locally:

```bash
python -m uvicorn app.api:app --reload --port 8000
```

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Project Structure

```text
comment-toxicity-detection/
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|
|-- app/
|   |-- streamlit_app.py
|   |-- api.py
|
|-- src/
|   |-- __init__.py
|   |-- load_model.py
|   |-- prediction.py
|   |-- preprocessing.py
|   |-- utils.py
|
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_TFIDF_LogisticRegression.ipynb
|   |-- 04_BiLSTM.ipynb
|   |-- 05_GloVe_BiLSTM.ipynb
|   |-- 06_DistilBERT_.ipynb
|   |-- 07_Model_Comparison.ipynb
|   |-- 08_Test_Data_Prediction.ipynb
|
|-- requirements.txt
|-- test_model.py
|-- .gitignore
|-- README.md
```

---

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/gopinath-sara1n/comment-toxicity-detection.git
cd comment-toxicity-detection
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv ctm
ctm\Scripts\activate

# Linux / macOS
python -m venv ctm
source ctm/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Applications

### 1. Launch Streamlit Web UI

```bash
python -m streamlit run app/streamlit_app.py
```

### 2. Launch FastAPI Microservice

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 3. Run Automated Tests

```bash
python test_model.py
```

---

## Trained Model Artifact

The fine-tuned DistilBERT weights are published on the Hugging Face Hub:

- [gopinath-sara1n/comment-toxicity-distilbert](https://huggingface.co/gopinath-sara1n/comment-toxicity-distilbert)

---

## Technologies Used

* **Language:** Python
* **NLP & Modeling:** PyTorch, Hugging Face Transformers, DistilBERT, NLTK, Scikit-learn
* **Serving & Web:** Streamlit (100% Native API), FastAPI, Uvicorn, Pydantic
* **Testing & CI:** Pytest, GitHub Actions CI
* **Deep Learning Baselines:** TensorFlow, Keras, GloVe Embeddings

---

## Author

**Gopinath S**  
Data Science & Machine Learning
