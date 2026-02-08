# Comparative Analysis of Medical Text Classification

**Team:** Theodora, Elyse, Fadh, Egide  

**Objective:** Evaluate how different model architectures (SVM vs. RNN vs. LSTM vs. GRU) perform on medical symptom data when paired with four distinct embedding techniques.

---

## Problem Definition & Dataset

**Classification problem:** Multi-class classification of medical symptom descriptions (free text) into disease/condition labels. The task is to predict the correct medical condition from patient-reported or clinical symptom text.

**Research questions:**
- Do different embedding techniques (TF-IDF, Word2Vec, GloVe, FastText) lead to different performance when used with the same model?
- Does architecture choice (SVM vs. RNN vs. LSTM vs. GRU) interact with embedding type (e.g. does TF-IDF suit SVM better than RNN)?
- Can transfer learning (GloVe) or sub-word models (FastText) improve medical text classification over in-domain embeddings (Word2Vec)?

**Dataset justification:** We use a medical symptom/disease text dataset (e.g. Kaggle `sarimahsan/daugmented-model` or `dataset/medical_llm_dataset.csv`) so that (1) the domain is clinically relevant and (2) the vocabulary and class distribution allow a fair comparison of embeddings and architectures. The same dataset and shared preprocessing ensure apples-to-apples comparison across team members.

---

## 1. Role & Model Distribution

Each team member owns one model and runs it against all four embeddings to maximize individual technical contribution.

| Team Member | Assigned Model | Role & Strategy | Notebook |
|-------------|----------------|-----------------|----------|
| **Elyse** | SVM / Logistic | The Baseline. Proves if deep learning is necessary. Uses TF-IDF heavily. | `02_Elyse_svm.ipynb` |
| **Theodora** | RNN | The Foundation. Tests basic sequence learning. Good for discussing vanishing gradients and long text. | `03_theodora_rnn.ipynb` |
| **Egide** | GRU | The Efficiency Champ. Faster, simplified LSTM. *(Notebook pending.)* | `04_egide_gru.ipynb` |
| **Fadh** | LSTM | The Heavy Hitter. Standard for sequence data. Expected high accuracy, slower training. | `05_fadh_lstm.ipynb` |

---

## 2. The Four Embedding Techniques (Experiment List)

Every member runs their assigned model with these **exact four embeddings** for apples-to-apples comparison in the final report.

| Embedding | Type | Hypothesis |
|-----------|------|------------|
| **TF-IDF** | Statistical baseline | Works well for SVM; likely poorly for RNN/LSTM. |
| **Skip-gram (Word2Vec)** | Contextual | Good for capturing rare medical terms. |
| **GloVe** | Pre-trained | Tests whether transfer learning (e.g. Wikipedia) helps. |
| **FastText** | Sub-word | Expected to perform best; handles medical prefixes/suffixes (e.g. "gastro-", "-itis"). |

---

## 3. Shared Preprocessing Strategy (Mandatory)

Same cleaning logic across all notebooks (required for full preprocessing marks):

- **Rule 1:** Lowercase everything.
- **Rule 2:** Keep hyphens (e.g. "X-ray", "Type-2").
- **Rule 3:** Remove medical stopwords (e.g. "patient", "doctor") that add noise.

**Lemmatization:** We use **no lemmatization** in the shared pipeline (per assignment spec) so that word forms are preserved for sequence models. If a notebook (e.g. Elyse’s SVM) uses lemmatization for a specific embedding, it is documented there as an adaptation.

Use the shared preprocessing code block at the top of each notebook.

---

## 4. Repository Structure

*Current layout (Egide’s notebook and optional shared EDA notebook may be added later):*

```
Formative2_Medical_Text_Classification/
├── README.md                    # This file — problem definition, tables, setup
├── requirements.txt             # pandas, sklearn, torch, etc.
├── data/
│   └── (dataset CSV and outputs; see notebooks for paths)
├── notebooks/
│   ├── 02_Elyse_svm.ipynb
│   ├── 03_theodora_rnn.ipynb
│   ├── 05_fadh_lstm.ipynb
│   └── (04_egide_gru.ipynb — pending)
└── results/
    └── final_results_table.csv  # Consolidated Accuracy & F1 (model × embedding)
```

---

## 5. Experiment Tables (for Report)

*Fill from each notebook and from `results/final_results_table.csv` when available.*

**Table 1 — Performance by model (each row = one model, columns = embeddings)**

| Model | TF-IDF (Acc / F1) | Word2Vec (Acc / F1) | GloVe (Acc / F1) | FastText (Acc / F1) |
|-------|-------------------|---------------------|------------------|---------------------|
| SVM (Elyse) | — | — | — | — |
| RNN (Theodora) | — | — | — | — |
| LSTM (Fadh) | — | — | — | — |
| GRU (Egide) | — | — | — | — |

**Table 2 — Performance by embedding (each row = one embedding, columns = models)**

| Embedding | SVM | RNN | LSTM | GRU |
|-----------|-----|-----|------|-----|
| TF-IDF | — | — | — | — |
| Word2Vec | — | — | — | — |
| GloVe | — | — | — | — |
| FastText | — | — | — | — |

---

## 6. Execution Plan

- **Phase 1 — Independent:** Explore data, generate 4 plots (class balance, text length, top 20 terms, vocabulary size), build model, run 4 embedding experiments, log results to the shared results spreadsheet and/or `results/final_results_table.csv` after each run.
- **Phase 2 — Report:** Combine results into the tables above, add comparative discussion, limitations, and citations.

---

## 7. Setup & Running

1. **Clone and enter the repo:**
   ```bash
   cd Formative2_Medical_Text_Classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data:** Use the dataset in `data/` or `dataset/` (e.g. `medical_llm_dataset.csv`). Some notebooks support Kaggle download — set `USE_KAGGLE` and paths as described in the notebook.

4. **GloVe (if used):** Place `glove.6B.100d.txt` in `data/` or let the notebook download it (see notebook instructions). Do not commit the large GloVe zip/txt files.

5. **Run notebooks** in order; log Accuracy and Weighted F1 to `results/final_results_table.csv` after each experiment.

---

## 8. Results & Report

- **Comparison tables** and **final graphs** for the report are in this README (Tables 1–2 above) and in `results/` once filled in.
- Each notebook saves its own plots (e.g. EDA, embedding comparison, confusion matrix) in `data/` or as specified in the notebook.
- **Evaluation metrics:** Accuracy, Weighted F1, and confusion matrices (as in each notebook).

---

## 9. Reproducibility

- Shared preprocessing and seed (e.g. `42`) across notebooks where applicable.
- Same train/val/test split strategy for fair comparison.
