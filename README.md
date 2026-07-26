
# Comment Toxicity Detection using DistilBERT

A multi-label Natural Language Processing (NLP) project for detecting toxic content in online comments using machine learning, deep learning, and transformer-based models.

The final application uses a fine-tuned **DistilBERT** model and provides real-time comment analysis and bulk CSV prediction through a **Streamlit web application**.

---

## Project Overview

Online platforms receive large volumes of user-generated comments, some of which may contain abusive, insulting, threatening, obscene, or hateful content.

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
- Toxic comments: 16,225
- Non-toxic comments: 143,346
- Number of target labels: 6

### Label Distribution

| Category | Count |
|---|---:|
| Toxic | 15,294 |
| Severe Toxic | 1,595 |
| Obscene | 8,449 |
| Threat | 478 |
| Insult | 7,877 |
| Identity Hate | 1,405 |

The dataset is highly imbalanced, with approximately 90% of comments containing no toxicity label.

---

## Text Preprocessing

The text preprocessing pipeline includes:

1. Converting text to lowercase
2. Removing URLs
3. Removing email addresses
4. Removing HTML tags
5. Removing emojis
6. Expanding contractions
7. Removing unnecessary characters
8. Normalizing repeated punctuation
9. Removing extra whitespace
10. Tokenization
11. Lemmatization

The same preprocessing pipeline is used during model inference to maintain consistency with model training.

---

## Models

Multiple approaches were explored before selecting the final model.

- Logistic Regression
- BiLSTM
- GloVe + BiLSTM
- DistilBERT

### Model Comparison

| Model | Hamming Loss | Micro Precision | Micro Recall | Micro F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.028481 | 0.570028 | 0.864674 | 0.687095 | 0.562904 | 0.980627 |
| BiLSTM | 0.017262 | 0.822722 | 0.666233 | 0.736254 | 0.443658 | 0.975242 |
| GloVe + BiLSTM | 0.016860 | 0.794361 | 0.720248 | 0.755492 | 0.587926 | 0.983861 |
| DistilBERT | **0.015032** | **0.837504** | 0.725014 | **0.777210** | **0.588496** | **0.988290** |

DistilBERT was selected as the final model because it achieved the strongest overall performance, including the highest ROC-AUC and Micro F1 and the lowest Hamming Loss.

---

## Final Model Performance

The final DistilBERT model achieved:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9883 |
| Micro F1 | 0.7772 |
| Micro Precision | 0.8375 |
| Micro Recall | 0.7250 |
| Macro F1 | 0.5885 |
| Hamming Loss | 0.0150 |

---

## Streamlit Application

An interactive Streamlit application was developed for model inference and visualization.

### Features

The application provides:

- Real-time single-comment toxicity detection
- Probability score for each toxicity category
- Sample test comments
- CSV batch prediction
- Maximum 1,000 comments per Streamlit batch
- Binary `0/1` output for bulk classification
- Downloadable prediction results
- Model performance metrics
- Model comparison
- Dataset insights
- Toxicity label distribution

---

## Prediction Modes

### Single Comment Prediction

For individual comments, the application displays the probability associated with each toxicity category.

Example:

```text
Toxic          99.56%
Severe Toxic    5.60%
Obscene        51.66%
Threat          1.73%
Insult         94.33%
Identity Hate   2.37%
````

### Bulk CSV Prediction

CSV files containing a `comment_text` column can be uploaded for batch analysis.

The bulk output uses binary classification:

```text
toxic,severe_toxic,obscene,threat,insult,identity_hate
1,0,1,0,1,0
0,0,0,0,0,0
```

A classification threshold of `0.5` is used to convert probabilities into binary predictions.

---

## Full Test Dataset Prediction

The complete preprocessed test dataset contains **153,164 comments**.

Because transformer inference over the complete dataset is computationally intensive, full test-set prediction is performed separately using a GPU-enabled Google Colab notebook rather than through Streamlit.

The inference pipeline is:

```text
Preprocessed Test Data
        |
        v
DistilBERT Tokenizer
        |
        v
Fine-tuned DistilBERT
        |
        v
Sigmoid Probabilities
        |
        v
Threshold = 0.5
        |
        v
Binary 0/1 Predictions
```

The final output contains:

```text
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

---

## Project Structure

```text
Comment Toxicity/
|
|-- app/
|   |-- streamlit_app.py
|
|-- src/
|   |-- __init__.py
|   |-- load_model.py
|   |-- prediction.py
|   |-- preprocessing.py
|
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_Preprocessing.ipynb
|   |-- 03_Logistic_Regression.ipynb
|   |-- 04_BiLSTM.ipynb
|   |-- 05_GloVe_BiLSTM.ipynb
|   |-- 06_DistilBERT.ipynb
|   |-- 07_Model_Comparison.ipynb
|   |-- 08_Test_Data_Prediction.ipynb
|
|-- data/
|-- models/
|-- outputs/
|
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- test_model.py
```

Actual notebook filenames may vary depending on the development workflow.

---

## Installation

Clone the repository and move into the project directory.

Create a virtual environment:

```bash
python -m venv ctm
```

### Windows

Activate the environment:

```bash
ctm\Scripts\activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Run the Streamlit Application

From the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

The application will start locally in the browser.

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* PyTorch
* Hugging Face Transformers
* DistilBERT
* TensorFlow / Keras
* GloVe Embeddings
* Scikit-learn
* Streamlit
* Google Colab
* Jupyter Notebook
* VS Code

---

## Key Learnings

This project demonstrates:

* NLP text preprocessing
* Multi-label classification
* Handling highly imbalanced text data
* Traditional machine-learning classification
* Bidirectional LSTM networks
* Pre-trained word embeddings
* Transformer fine-tuning
* Multi-label model evaluation
* GPU-based model training and inference
* Batch inference
* Model deployment using Streamlit
* Modular ML project development

---

## Future Improvements

Possible improvements include:

* Per-label threshold optimization
* Improved handling of minority toxicity classes
* Model quantization for faster CPU inference
* Additional transformer architectures
* API-based model serving
* Cloud deployment
* Model monitoring and feedback collection

---

## Author

**Gopinath S**

Data Science & Machine Learning

