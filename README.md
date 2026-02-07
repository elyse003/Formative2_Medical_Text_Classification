# Medical Symptom Text Classification: Simple RNN with Multiple Embeddings

This repository contains the **Simple RNN** implementation for a comparative text classification study on a Medical Symptom dataset. It is part of a group assignment comparing different model architectures (SVM, RNN, LSTM, GRU) across multiple word embedding strategies.

## Author Role

- **Model:** Simple RNN (PyTorch)
- **Embeddings evaluated:** Random Initialization (baseline), Word2Vec (Skip-gram), GloVe, FastText
- **Dataset:** Medical Symptom (configurable; see below)

## Repository Structure

```
medical_rnn_text_classification/
├── Medical_RNN_Embeddings.ipynb   # Main notebook: EDA, preprocessing, RNN, 4 experiments, results
├── data/                           # Dataset and outputs (dataset.csv, GloVe, figures)
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo** (or download) and enter the directory:
   ```bash
   cd medical_rnn_text_classification
   ```

2. **Create environment and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (first run of the notebook will do this):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Dataset

- **Option A — Kaggle:** Set `USE_KAGGLE = True` in the notebook and set `KAGGLE_DATASET` to your dataset slug (e.g. `niyarrbarman/symptom2disease`). Ensure the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed and configured.
- **Option B — Local/URL:** Set `USE_KAGGLE = False` and set `DATA_PATH` to the path of your CSV (or a URL). The CSV must have a text column and a label column (names configurable via `TEXT_COL` and `LABEL_COL`).
- **Fallback:** If no file is found at `DATA_PATH`, the notebook creates a small **synthetic** medical symptom dataset in `data/dataset.csv` so the pipeline runs end-to-end. Replace with your team’s dataset for real experiments.

## Preprocessing (Shared Pipeline)

- Lowercase; remove punctuation except hyphens (e.g. preserve "X-ray").
- Remove custom medical stopwords (`patient`, `doctor`, `history`) and NLTK English stopwords.
- **No lemmatization** (per assignment spec).
- Vocabulary: index 0 = `<PAD>`, 1 = `<UNK>`; sequences padded/truncated to **MAX_LEN=50**.

## Running the Notebook

1. Open `Medical_RNN_Embeddings.ipynb` in Jupyter or VS Code.
2. Run all cells in order.
3. **GloVe:** The notebook can download `glove.6B.100d.txt` into `data/` if missing. Alternatively, place the file at `data/glove.6B.100d.txt` yourself.

## Outputs

- **Tables:** Accuracy and Weighted F1 for each embedding (Random, Word2Vec, GloVe, FastText) in a Pandas DataFrame and formatted for the report.
- **Plots:** Saved in `data/`:
  - `eda_visualizations.png` — EDA (class distribution, text length, vocabulary, word length).
  - `rnn_embedding_comparison.png` — Bar chart of Accuracy and Weighted F1 by embedding.
  - `confusion_matrix.png` — Confusion matrix for the best-performing embedding.

## Reproducibility

- **Seed:** All experiments use `seed=42` (random, numpy, torch).
- Same train/val/test split and preprocessing for every embedding.

## Citation / Report

When writing the report, use the methodology and results from this notebook. Include:
- Dataset description and EDA figures.
- Preprocessing and embedding strategy (with citations).
- At least two comparison tables (e.g. RNN-by-embedding; full team model-by-embedding).
- Discussion of why certain embeddings work better with the RNN, with references.
