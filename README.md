# Multi-Class Text Classification with Deep Learning using BERT

End-to-end project that fine-tunes **Google's BERT** (Bidirectional Encoder Representations from Transformers) to classify news headlines into **4 categories** on the **AG News** dataset:

| Label | Class |
|:-----:|-------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

The same solution is provided in **two forms**: exploratory Jupyter notebooks (for learning) and a clean, production-style **modular code** pipeline (for reuse).

---

## 📖 Project Takeaways

This project teaches the complete journey from classical NLP to Transformers. All **18 takeaways** are explained in detail in **[PROJECT_TAKEAWAYS.md](PROJECT_TAKEAWAYS.md)**:

<details>
<summary><b>Click to view all 18 takeaways (quick list)</b></summary>

**Concepts (1–10)**
1. Understanding the business problem
2. Basics of NLP
3. Bag of Words (BOW) and TF-IDF models
4. Word embeddings — Word2Vec, GloVe, FastText
5. RNN and LSTM models
6. Transformers & encoder-decoder architecture
7. The Attention mechanism
8. The BERT model
9. Special tokens used in BERT
10. BERT pre-training — MLM and NSP

**Hands-on coding (11–18)**

11. Importing required libraries
12. Importing data from the Hugging Face `datasets` library
13. Creating DataFrame objects for train/test data
14. Data preprocessing (BERT tokenization via ktrain)
15. Creating the BERT model
16. Compiling & training BERT in a learner object
17. Evaluating performance metrics
18. Saving / reloading the fine-tuned model

</details>

---

## 🗂 Repository Structure

```
.
├── README.md                                  ← You are here
├── PROJECT_TAKEAWAYS.md                        ← All 18 takeaways explained
├── BERT solution methodology-converted.pdf     ← Original project methodology document
├── .gitignore
├── Notebook_Code/                              ← Exploratory, learning-oriented notebooks
│   ├── Transformers_in_NLP_1_Explained.ipynb   ← Theory: BOW → TF-IDF → Word2Vec → RNN/LSTM → Attention → BERT
│   └── Bert_AG_News_Classification.ipynb       ← Practical: AG News classification with BERT step by step
└── Project_Modular_Code/                       ← Production-style, reusable pipeline
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   ├── engine.py                           ← Entry point — orchestrates the whole pipeline
    │   └── ML_Pipeline/
    │       ├── utils.py                        ← Dataset loading & DataFrame conversion
    │       ├── feature_engineering.py           ← BERT preprocessing (tokenization)
    │       └── model.py                         ← Build, train, evaluate, save, reload BERT
    ├── output/
    │   └── bert-ag-news-predictor/             ← Saved fine-tuned model (load & predict, no retraining)
    └── lib/                                    ← Reference copies of the notebooks
```

---

## ⚙️ How It Works (Pipeline)

1. **Load dataset** — `ag_news` is fetched from the Hugging Face Hub (`datasets` library): 120,000 training + 7,600 test samples (30,000 train / 1,900 test per class).
2. **Convert to DataFrames** — train/test splits become `pandas` DataFrames (`text`, `label` columns).
3. **Preprocess (feature engineering)** — `ktrain.text.texts_from_df()` tokenizes text into BERT input format (max sequence length **512**).
4. **Build & train** — `bert-base-uncased` classifier is fine-tuned with the **1cycle policy**:
   - Learning rate: `2e-5`
   - Epochs: `3`
   - Batch size: `6`
5. **Evaluate** — validation report with per-class precision / recall / F1.
6. **Save & reload** — the predictor is saved to `output/bert-ag-news-predictor` for future predictions without retraining.

---

## 🚀 Quickstart (Modular Code)

**Python 3.8.10** is the supported version.

```bash
cd Project_Modular_Code

# 1. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the pipeline
python src/engine.py
```

`engine.py` automatically runs every step: load data → preprocess → train → evaluate → save → reload.

> ⚠️ **Note:** Training on the full 120k samples on CPU is slow. Use a GPU/TPU (the notebooks were built on Google Colab). You can also train on a subset first to verify everything works.

### Making a prediction after training

```python
import ktrain
predictor = ktrain.load_predictor('output/bert-ag-news-predictor')
print(predictor.predict("The stock market rallied after the Fed announcement."))
# → 'Business'
```

---

## 🔀 Notebook Code vs Modular Code — What's the Difference?

Both solve the **same problem with the same model, data, and hyperparameters** — they differ in *purpose and form*:

| Aspect | `Notebook_Code/` (Notebooks) | `Project_Modular_Code/` (Modular) |
|---|---|---|
| **Purpose** | Learning, exploration & demonstration | Reusable, production-style pipeline |
| **Structure** | One long linear flow of cells | Split into small functions across modules |
| **Organization** | All logic interleaved with markdown explanations & outputs | `engine.py` orchestrates; `ML_Pipeline/` holds the logic (`utils.py`, `feature_engineering.py`, `model.py`) |
| **Readability** | Great for following *why* each step is done (rich theory notes) | Great for seeing *how* the pipeline is assembled at a glance |
| **Reusability** | Copy-paste cells to reuse code | Import functions anywhere; each step is callable independently |
| **Maintainability** | Changes require editing the right cell in a big notebook | Change one function in one file; the rest is untouched |
| **Testing / debugging** | Manual, cell by cell | Functions are isolated — easy to unit-test and debug |
| **Execution** | Interactive, step-by-step (Colab) | One command: `python src/engine.py` |
| **Extras** | Contains theory notebook (`Transformers_in_NLP_1_Explained.ipynb`), GPU checks (`nvidia-smi`), pip installs in cells, zipping of model | Only pipeline logic; installs handled by `requirements.txt`, error handling with try/except |
| **Model saving path** | `/content/bert-ag-news-predictor` (Colab) | `output/bert-ag-news-predictor` (relative, portable) |

**Rule of thumb:** read the notebooks to *learn the concepts*; use the modular code to *build on or deploy* the solution.

---

## 🧰 Tech Stack

- **Language:** Python 3.8
- **Core libraries:** `ktrain` (wrapper around TensorFlow + Transformers that simplifies BERT fine-tuning), `transformers` (Hugging Face), `datasets` (Hugging Face Hub data), `tensorflow`, `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` / `seaborn`
- **Dataset:** [AG News (Hugging Face)](https://huggingface.co/datasets/ag_news)
- **Model:** `bert-base-uncased`

---

## 📚 Further Reading

- [PROJECT_TAKEAWAYS.md](PROJECT_TAKEAWAYS.md) — detailed explanation of all 18 project takeaways
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — original paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the Transformer paper
- [ktrain documentation](https://github.com/camrobjones/ktrain)
