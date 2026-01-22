
---

# 📘 Deep Learning Drug Recommendation System – Jupyter Notebook

This notebook presents a **deep learning–based drug recommendation system** that analyzes **patient conditions and free-text medical reviews** to generate **Top-K ranked drug recommendations with probability scores**. The notebook demonstrates the complete machine learning workflow from data preprocessing to model evaluation and persistence.

> ⚠️ **Disclaimer:** This notebook is for educational and research purposes only and does not provide medical advice.

---

## 🎯 Objective

The goal of this notebook is to:

* Process noisy, unstructured healthcare text
* Learn contextual representations using deep learning
* Recommend relevant drugs based on patient-reported information
* Serve as a research and experimentation baseline for healthcare NLP

---

## 🧠 Model Overview

The notebook implements the following deep learning architecture:

```
Text Input (Condition + Review)
        ↓
TextVectorization
        ↓
Embedding Layer
        ↓
Bidirectional LSTM
        ↓
Dense Layers
        ↓
Softmax Output (Drug Probabilities)
```

This design enables the model to capture contextual meaning in medical language rather than relying on simple keyword matching.

---

## 📂 Dataset Description

The dataset used in this notebook contains:

* **Condition:** Medical condition or diagnosis
* **Review:** Free-text patient experience
* **Drug Name:** Prescribed medication (target variable)

To improve training stability, rare drug classes are filtered and the dataset is split using a stratified train–test strategy.

---

## ⚙️ Notebook Workflow

The notebook follows these steps:

1. Load and explore the dataset
2. Preprocess and normalize medical text
3. Encode drug labels for multi-class classification
4. Build a BiLSTM-based deep learning model
5. Train and evaluate the model
6. Generate Top-K drug recommendations
7. Save the trained model and metadata

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Macro-averaged Precision, Recall, and F1-score
* Top-K recommendation accuracy
* Confusion matrix analysis

Top-K metrics are emphasized to align with real-world clinical decision-making.

---

## 🧪 Example Output

Given a patient condition and symptom description, the notebook outputs:

* A ranked list of recommended drugs
* Associated probability scores for each recommendation

This probabilistic ranking improves transparency and interpretability.

---


---
