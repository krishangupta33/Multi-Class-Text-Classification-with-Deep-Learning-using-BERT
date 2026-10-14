# 📚 Project Takeaways — AG News Classification with BERT

This document explains everything you learn from this project, step by step. It combines the **theory** (from `Notebook_Code/Transformers_in_NLP_1_Explained.ipynb`) and the **practice** (from `Bert_AG_News_Classification.ipynb` and `Project_Modular_Code/`).

---

## Part A — NLP & Deep Learning Concepts (1–10)

### 1. Understanding the Business Problem

Text is the most abundant data in the world, but it is **unstructured** — extracting insights manually is slow and expensive. A powerful automatic **text-processing / classification** system is therefore a necessity, not a luxury. The business objective here: classify 100k+ news articles into 4 categories (World, Sports, Business, Sci/Tech) automatically using a state-of-the-art model — **BERT**.

### 2. Basics of NLP

Natural Language Processing lets machines understand and generate human language. Common NLP applications include sentiment analysis, translation, summarization, question answering, named-entity recognition, and — our focus — **text classification**.

### 3. Bag of Words (BOW) & TF-IDF

The classical way to turn text into numbers:

- **BOW** — build a vocabulary from all documents and represent each document as a count vector of word frequencies. Simple, but ignores word order and context.
- **TF-IDF** — weighs words by how important they are: frequent in a document (TF) but rare across documents (IDF) ⇒ high weight. Better than raw counts, but still just a frequency-based method with no notion of *meaning*.

### 4. Word Embeddings — Word2Vec, GloVe, FastText

Frequency methods can't capture semantic similarity. **Word embeddings** represent words as **dense, fixed-length vectors** where similar meanings are close in vector space:

- **Word2Vec** — a shallow 2-layer network trained to predict context (CBOW) or the target word from context (Skip-Gram).
- **GloVe** — builds a global word co-occurrence matrix and factors it into vectors.
- **FastText** — improves Word2Vec by also learning vectors for **character n-grams**, so it can handle out-of-vocabulary and misspelled words.

### 5. RNN & LSTM

- **RNN** — processes text sequentially, carrying a hidden state across time steps, so word order matters. But it suffers from vanishing gradients and forgets long-range dependencies.
- **LSTM** — adds gates (input/forget/output) and a cell state to selectively remember or forget information, fixing much of the long-range memory problem. Still sequential ⇒ slow and hard to parallelize.

### 6. Transformers & Encoder–Decoder Architecture

The Transformer replaced recurrence with **attention**, processing all tokens **in parallel**:

- The **encoder** stack builds a rich contextual representation of the input.
- The **decoder** stack generates output, attending both to the encoder output and to previously generated tokens.
- Each block contains identical, stacked layers — no recurrence, no convolution.

### 7. The Attention Mechanism (Self-Attention & Multi-Head)

Attention lets the model decide **which words to focus on** when encoding each word. Each word embedding is projected into **Query / Key / Value** vectors; compatibility between queries and keys determines how much each value contributes. **Multi-Head** attention runs several attention "heads" in parallel so the model can capture different relationship types (syntax, coreference, etc.) at once. E.g., in *"The animal didn't cross the street because **it** was tired"*, attention links *"it"* → *"animal"*.

### 8. The BERT Model

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers — Google's model built **only from the Transformer encoder stack**. Unlike GPT (left-to-right) or ELMo (shallow bidirectional), BERT looks at context from **both directions simultaneously** at every layer, producing deeply contextual embeddings that achieve state-of-the-art results on a wide variety of NLP tasks.

### 9. Special Tokens in BERT

- `[CLS]` — prepended to every input; its final hidden state is used as the **aggregate sequence representation** (used for classification).
- `[SEP]` — marks sentence / segment boundaries.
- `[MASK]` — substituted for tokens during Masked Language Model training.
- Each token embedding = **token embedding + segment embedding + position embedding**.

### 10. BERT Pre-training — MLM & NSP

BERT is pre-trained on two unsupervised tasks over huge corpora (BooksCorpus + Wikipedia):

- **Masked Language Modeling (MLM)** — ~15% of tokens are masked, and BERT must predict them using bidirectional context (fill-in-the-blank).
- **Next Sentence Prediction (NSP)** — given two sentences, predict whether the second truly follows the first (learning sentence relationships).

After pre-training, BERT is **fine-tuned** on a downstream task (here: AG News classification) with just one extra classification layer.

---

## Part B — Hands-On Implementation (11–18)

These map directly to the modular pipeline in `Project_Modular_Code/src/`.

### 11. Importing Required Libraries

Core stack: `ktrain`, `transformers`, `datasets`, `tensorflow`, `torch`, `pandas`, `numpy`, `timeit`. In the modular code this step lives in the import sections of `engine.py` and each `ML_Pipeline` module, with a `requirements.txt` pinning versions.

### 12. Importing Data from the Hugging Face Library

Using the `datasets` library — check available datasets with `list_datasets()`, then load AG News with `load_dataset('ag_news')` and inspect its shape, columns (`text`, `label`), and sample rows. (In modular code: `utils.load_and_display_dataset_details()`.)

### 13. Creating DataFrame Objects for Train & Test Data

Train and test splits are loaded separately and converted into **pandas DataFrames**, which is the input format ktrain's preprocessing expects. Class label names are defined: `['World', 'Sports', 'Business', 'Sci/Tech']`. (In modular code: `utils.load_and_convert_data_to_df()`.)

### 14. Data Preprocessing (Feature Engineering)

`ktrain.text.texts_from_df()` converts the DataFrames into BERT-ready inputs — tokenization, special-token insertion, attention masks, and integer label encoding — with **max sequence length 512** (`preprocess_mode='bert'`). This also returns the `preprocessing_var` needed later for saving/reloading the model consistently. (In modular code: `feature_engineering.perform_feature_engineering()`.)

### 15. Creating the BERT Model

`ktrain.text.text_classifier(name='bert', ...)` builds a `bert-base-uncased` model with a classification head on top of the `[CLS]` representation. (In modular code: `model.create_and_train_bert_model()`.)

### 16. Compiling & Training BERT in a Learner Object

`ktrain.get_learner()` wraps the model with train/validation data and batch size; `fit_onecycle(lr=2e-5, epochs=3)` applies the **1cycle policy** for smooth, stable fine-tuning:

| Hyperparameter | Value |
|---|---|
| Max sequence length | 512 |
| Batch size | 6 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Schedule | 1cycle |

Training time is measured with `timeit`.

### 17. Evaluating Performance Metrics

`bert_learner.validate()` prints the classification report — overall accuracy plus **per-class precision, recall, and F1-score** (with and without human-readable class names). Fine-tuned BERT reaches ~94–95% accuracy on AG News, surpassing classical approaches.

### 18. Saving (and Reloading) the Model

`ktrain.get_predictor(model, preproc).save(path)` persists both the **fine-tuned model and its preprocessor** to `output/bert-ag-news-predictor`, so it can later be reloaded with `ktrain.load_predictor(path)` and used for instant predictions — no retraining required.

---

## Summary

You move from **how text becomes numbers** (BOW → TF-IDF → embeddings) → **how sequences were modeled** (RNN → LSTM) → **how Transformers changed everything** (attention, encoder-decoder) → **BERT** (architecture, tokens, pre-training) → and finally **fine-tune it end-to-end** on a real 100k+ example dataset in both notebook and production-style modular form.
