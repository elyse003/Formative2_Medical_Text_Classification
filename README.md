# Comparative Analysis of Medical Text Classification

**Team:** Theodora, Elyse, Fadh, Egide  

**Objective:** Evaluate how different model architectures (SVM vs. RNN vs. LSTM vs. GRU) perform on medical symptom data when paired with four distinct embedding techniques.

---

## 1. Role & Model Distribution

Each team member owns one model and runs it against all four embeddings to maximize individual technical contribution.

| Team Member | Assigned Model | Role & Strategy | Notebook |
|-------------|----------------|-----------------|----------|
| **Elyse** | SVM / Logistic | The Baseline. Proves if deep learning is necessary. Uses TF-IDF heavily. | `02_elyse_svm.ipynb` |
| **Theodora** | RNN | The Foundation. Tests basic sequence learning. Good for discussing vanishing gradients and long text. | `03_theodora_rnn.ipynb` |
| **Egide** | GRU | The Efficiency Champ. Faster, simplified LSTM. Expected to perform well with quicker training. | `04_egide_gru.ipynb` |
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

Use the shared preprocessing code block at the top of each notebook.

---

## 4. Repository Structure

```
Formative2_Medical_Text_Classification/
├── README.md                    # This file — comparison tables & final graphs for report
├── requirements.txt             # pandas, sklearn, torch, etc.
├── data/
│   └── medical_text.csv         # (or dataset/medical_llm_dataset.csv — see notebooks)
├── notebooks/
│   ├── 01_data_exploration_shared.ipynb
│   ├── 02_elyse_svm.ipynb
│   ├── 03_theodora_rnn.ipynb
│   ├── 04_egide_gru.ipynb
│   └── 05_fadh_lstm.ipynb
└── results/
    └── final_results_table.csv  # Consolidated Accuracy & F1 (model × embedding)
```

---

## 5. Execution Plan

- **Phase 1 — Independent:** Explore data, generate 4 plots (class balance, text length, top 20 terms, vocabulary size), build model, run 4 embedding experiments, log results to the shared results spreadsheet after each run.
- **Phase 2 — Report:** Combine results, comparison tables, and discussion (see README and `results/`).

---

## 6. Setup & Running

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

## 7. Results & Report

- **Comparison tables** and **final graphs** for the report live in this README (or in `results/`) once filled in.
- Each notebook saves its own plots (e.g. EDA, embedding comparison, confusion matrix) in `data/` or as specified in the notebook.

---

## 8. Reproducibility

- Shared preprocessing and seed (e.g. `42`) across notebooks where applicable.
- Same train/val/test split strategy for fair comparison.
