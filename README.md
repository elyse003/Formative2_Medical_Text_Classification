# A Comparative Analysis of Medical Text Classification: Model Architectures Across Multiple Word Embeddings

**Multi-class medical symptom text classification** comparing four model architectures (SVM, RNN, LSTM, GRU) and four word-embedding techniques (TF-IDF, Skip-gram Word2Vec, GloVe, FastText) on a single dataset with shared preprocessing.

---

## Team & Facilitator

| Role | Name |
|------|------|
| **Team** | Theodora Egbunike, Elyse Marie Uyiringiye, Fadhlullah Abdulazeez, Egide Harerimana |
| **Facilitator** | Ms Samiratu Ntohsi |

---

## What This Repository Contains

- **Report:** `medical text classification document.txt` — full written report (methodology, results, discussion, references).
- **Notebooks:** One notebook per model plus a unified comparison notebook (see below).
- **Results:** Consolidated performance tables and comparison CSVs in `results/`.
- **Setup:** `requirements.txt` for Python dependencies.

---

## Problem & Objectives

**Task:** Classify free-text medical symptom descriptions into one of 105 condition labels (multi-class classification).

**Goals:**
- Implement four model architectures, each evaluated with the same four embeddings.
- Use a **shared preprocessing pipeline** across all experiments for fair comparison.
- Perform hyperparameter tuning per embedding within each model.
- Produce **multiple comparison tables and figures** and analyse why certain embeddings suit certain architectures (with literature support).

**Research questions:**
- Do TF-IDF, Skip-gram, GloVe, and FastText lead to different performance when used with the same model?
- Does architecture (SVM vs RNN vs LSTM vs GRU) interact with embedding type?
- Does transfer learning (GloVe) or sub-word modelling (FastText) improve over in-domain Word2Vec on this task?

---

## Dataset

- **Source:** [Kaggle — Medical NLP / symptom text](https://www.kaggle.com/datasets/sarimahsan/daugmented-model) (or equivalent medical symptom dataset as used in the notebooks).
- **Domain:** Medical symptom descriptions; labels are condition/symptom classes.
- **Scale (representative):** ~77k samples, 105 classes; train/val/test splits (e.g. 70/15/15 or per-notebook). Vocabulary and max sequence length (e.g. 50) are set per notebook.
- **Justification:** Single dataset and aligned preprocessing allow apples-to-apples comparison of embeddings and architectures.

---

## Models & Embeddings

Each team member **owns one model** and runs it with **all four embeddings**.

| Team Member | Model | Notebook | Framework |
|-------------|--------|----------|-----------|
| **Elyse** | SVM (LinearSVC) | `02_Elyse_svm.ipynb` | scikit-learn |
| **Theodora** | RNN (MLP for TF-IDF; RNN for sequences) | `03_theodora_rnn.ipynb` | PyTorch |
| **Fadhlullah** | LSTM | `05_fadh_lstm.ipynb` | PyTorch |
| **Egide** | GRU | `04_egide_gru.ipynb` | TensorFlow/Keras |

**Unified comparison:** `01_unified_comparison.ipynb` — builds consolidated tables (Tables 2–8) and unified bar charts/heatmaps from all model–embedding results.

**The four embeddings (every model × every embedding):**

| Embedding | Type | Notes |
|-----------|------|--------|
| **TF-IDF** | Statistical | Document-term vectors; baseline for SVM/MLP. |
| **Skip-gram (Word2Vec)** | Contextual | In-domain training; good for rare terms. |
| **GloVe** | Pre-trained | e.g. glove.6B.100d; transfer learning. |
| **FastText** | Sub-word | Character n-grams; handles OOV and morphology. |

---

## Shared Preprocessing

All notebooks follow the same core rules for fair comparison:

1. **Lowercase** all text.
2. **Preserve hyphens** (e.g. "X-ray", "Type-2").
3. **Remove medical stopwords** (e.g. "patient", "doctor", "symptoms") in addition to standard stopwords.
4. **No lemmatisation** in the shared pipeline (word forms preserved for sequence models). Any notebook-specific adaptation (e.g. lemmatisation in one notebook) is documented there.

Preprocessing is then **adapted per embedding**: e.g. tokenisation and sequence length for Word2Vec/GloVe/FastText; vectorisation for TF-IDF.

---

## Repository Structure

```
Formative2_Medical_Text_Classification/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── medical text classification document.txt   # Final report
├── .gitignore
├── data/                               # Dataset, GloVe vectors, saved figures (create if needed)
├── notebooks/
│   ├── 01_unified_comparison.ipynb     # Consolidated tables & figures (Tables 2–8, bar charts, heatmaps)
│   ├── 02_Elyse_svm.ipynb              # SVM + TF-IDF, Word2Vec, GloVe, FastText
│   ├── 03_theodora_rnn.ipynb           # RNN (MLP + sequence) + four embeddings
│   ├── 04_egide_gru.ipynb              # GRU + four embeddings
│   └── 05_fadh_lstm.ipynb              # LSTM + four embeddings
└── results/
    ├── consolidated_final_results.csv # Table 2: all 16 model–embedding Accuracy & F1
    ├── table3_best_per_model.csv      # Best configuration per model
    ├── table4_by_embedding.csv         # Comparison by embedding (best model, mean F1)
    ├── table5_ranking.csv              # Top 10 configurations by F1
    ├── table6_setup_methodology.csv    # Setup/methodology per model
    ├── table7_extended_metrics.csv     # Extended metrics (optional columns)
    └── table8_*_embedding_ranking.csv  # Per-model embedding ranking (SVM, RNN, LSTM, GRU)
```

---

## Setup & Running

1. **Clone the repository and enter the project directory:**
   ```bash
   git clone <repo-url>
   cd Formative2_Medical_Text_Classification
   ```

2. **Create a virtual environment (recommended) and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data:**  
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sarimahsan/daugmented-model) or use the path expected by each notebook (e.g. `data/` or `dataset/`).  
   - Some notebooks use `kagglehub` for download; install with `pip install kagglehub` and configure Kaggle credentials if needed.

4. **GloVe (for LSTM/GRU/RNN):**  
   Place `glove.6B.100d.txt` in `data/` or follow the download instructions inside the notebook. Do not commit large GloVe files.

5. **Run notebooks:**  
   - Run **02, 03, 04, 05** to reproduce each model’s results (each notebook runs four embedding experiments).  
   - Run **01_unified_comparison.ipynb** to regenerate consolidated tables and figures (uses the same numbers as in the report; can also load from `results/consolidated_final_results.csv`).

6. **Reproducibility:**  
   Use the same random seed (e.g. 42) and split strategy where documented; shared preprocessing keeps comparisons fair.

---

## Results Summary

- **Best overall (working runs):** GRU + TF-IDF (Accuracy 0.9878, F1 0.9877). SVM + TF-IDF and RNN (MLP) + TF-IDF are close (0.9865–0.9867).
- **TF-IDF** is strong with SVM and with MLP/GRU as a document vector; **centroid-aggregated** Word2Vec/GloVe/FastText with SVM underperform.
- **LSTM and GRU** achieve high performance with all four embeddings when the sequence pipeline is correct.
- **RNN sequence runs** (Skip-gram, GloVe, FastText) in the current pipeline yield near-zero test performance (implementation/vocabulary issue); only RNN + TF-IDF (MLP) results are used in the comparison.

Full tables (dataset stats, performance matrix, best per model, by embedding, top 10, setup, per-model rankings) are in the **report** and in `results/` as CSVs. Figures (class balance, text length, keywords, vocab, accuracy/F1 bar charts, heatmaps, best-per-model plot) are produced in the notebooks and referenced in the report.

---

## Report & Contribution Tracker

- **Report:** See: https://docs.google.com/document/d/1COUNlRYJLK0SfyecSquL2aF_SfgI6IKphquTDsMaOdE/edit?tab=t.0
- **Team contribution tracker:** [Google Sheet](https://docs.google.com/spreadsheets/d/1RxjsGhus9zsNAexQp4CXROhDn1yDyL-lg4517rlwgFo/edit?usp=sharing) (link also in report Appendix 8.1).

---

## Evaluation Metrics

- **Accuracy** — fraction of correct predictions.
- **F1** — weighted F1 (most notebooks) or macro F1 (LSTM notebook); reported per table.
- **Confusion matrices** — used in notebooks for selected model–embedding pairs; referenced in the report.

---

## Citation & License

If you use this code or results, please cite the dataset (Kaggle) and the report. This project was produced for academic coursework; see the report and contribution tracker for author details and references.
