# NLP: Punctuation and Capitalization Restoration with BERT

## Overview

This project solves the task of **automatic punctuation and capitalization restoration** in text using a transformer-based model.

Automatic Speech Recognition (ASR) systems usually produce text without punctuation and capitalization. Such raw text:

- is difficult for humans to read
- reduces the performance of downstream NLP tasks
- lacks sentence boundaries and named entity cues

The goal of this project is to **restore punctuation marks and capitalization in unformatted text** using a deep learning model based on **BERT**.

---

## Problem Statement

Input:

`hello how are you doing today i hope everything is fine`  


Expected output:

`Hello, how are you doing today? I hope everything is fine.`


The model predicts:

1. **Punctuation after each token**
2. **Capitalization of each token**

---

## Model Architecture

The model uses a **multi-task learning architecture** built on top of a pretrained transformer.

### Backbone

- **BERT** (Bidirectional Encoder Representations from Transformers)
- Generates contextual embeddings for each token.

### Classification Heads

Two independent classification heads are applied on top of BERT outputs.

#### 1. Punctuation Head

Predicts punctuation mark that should appear **after the token**.

Possible classes:

1) O -> no punctuation
2) , -> comma
3) . -> period
4) ? -> question mark
5) ! -> exclamation mark


#### 2. Capitalization Head

Predicts capitalization of the **current token**.

Classes:

1) L -> lowercase
2) U -> uppercase


---

## Training Objective

The model is trained with **joint learning**.

Total loss:

`Loss_total = Loss_punct + Loss_cap`


Where:

- `Loss_punct` — CrossEntropy loss for punctuation classification
- `Loss_cap` — CrossEntropy loss for capitalization classification

---

## Pipeline

The notebook implements the following pipeline:

1. **Data preprocessing**
   - tokenization
   - alignment of labels with tokens
   - creation of punctuation and capitalization labels

2. **Dataset preparation**
   - PyTorch Dataset
   - batching and padding

3. **Model construction**
   - pretrained BERT encoder
   - two classification heads

4. **Training**
   - joint loss optimization
   - training loop
   - loss tracking

5. **Evaluation**
   - validation metrics
   - qualitative predictions

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- NumPy
- Pandas
- Matplotlib

---
