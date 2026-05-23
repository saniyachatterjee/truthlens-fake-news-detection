# TruthLens — AI Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-deployed-lightgrey?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> An NLP-based fake news detection system trained on a **44,900-article dataset**, 
> benchmarking Logistic Regression, Naive Bayes, and LSTM models — achieving up to 
> **98.74% accuracy** — deployed as a Flask web application.

---

## 🎯 Problem Statement

The exponential rise of digital misinformation across social media platforms makes 
manual fact-checking infeasible at scale. TruthLens builds an AI pipeline that 
automatically classifies news headlines as REAL or FAKE based on linguistic and 
semantic patterns — using classical ML and deep learning approaches side by side.

---

## 🏗️ System Architecture

```
Raw News Text (headline + article)
        ↓
Text Preprocessing
(lowercase · stopword removal · stemming · TF-IDF)
        ↓
┌─────────────────────────────────────┐
│  Model 1: Logistic Regression       │
│  Model 2: Naive Bayes               │
│  Model 3: LSTM (Deep Learning)      │
└─────────────────────────────────────┘
        ↓
Majority Vote → Final Verdict
(REAL / FAKE + Confidence Score)
        ↓
Flask Web Dashboard
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **98.74%** | 98.7% | 98.7% | 98.7% |
| Naive Bayes | 94.2% | 94.1% | 94.3% | 94.2% |
| LSTM (Deep Learning) | 97.8% | 97.9% | 97.7% | 97.8% |

> Logistic Regression selected as primary model based on accuracy and inference speed.
> Final verdict uses **majority vote** across all three models.

---

## ✨ Features

- **Real-time classification** — type any headline and get instant FAKE/REAL verdict
- **Confidence scoring** — percentage confidence per model and ensemble verdict
- **Warning signals** — flags sensationalist language, excessive caps, clickbait phrases
- **Multi-model comparison** — see how all 3 models vote on each headline
- **Agreement indicator** — shows when models disagree and explains why
- **Web dashboard** — clean Flask UI with history of recent predictions
- **Terminal mode** — lightweight `check_news.py` for quick CLI predictions

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Classical ML | scikit-learn — Logistic Regression, Naive Bayes |
| Deep Learning | TensorFlow/Keras — LSTM |
| NLP | NLTK — tokenization, stemming, stopwords |
| Feature Extraction | TF-IDF Vectorizer (5000 features, bigrams) |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
truthlens-fake-news-detection/
├── app.py                    # Flask web dashboard
├── fake_news_detection.py    # Full training pipeline with evaluation
├── check_news.py             # Terminal-based prediction tool
├── requirements.txt
└── dataset/                  # Add Kaggle CSVs here (see below)
    ├── Fake.csv
    └── True.csv
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/saniyachatterjee/truthlens-fake-news-detection
cd truthlens-fake-news-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Get the dataset from Kaggle:
👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` inside a `dataset/` folder.

> **No dataset?** The app automatically falls back to a built-in synthetic dataset 
> so you can still run and demo it without Kaggle.

**4. Run the web dashboard**
```bash
python app.py
```
Open **http://localhost:8000** in your browser.

**5. Or run terminal mode**
```bash
python check_news.py
```

---

## 💡 Key Findings

- **TF-IDF + Logistic Regression** outperformed LSTM on this dataset — showing 
  that for well-structured fake news corpora, classical ML can match deep learning 
  with far less compute
- **Naive Bayes** degraded significantly on short headlines (alpha sensitivity) — 
  fixed by raising smoothing from 0.1 → 1.0 (Laplace smoothing)
- **Majority voting** across 3 models reduced single-model errors and improved 
  robustness on ambiguous headlines
- **Short headline guard** — inputs under 3 tokens after preprocessing are flagged 
  and confidence is capped at 70% to prevent misleadingly high fake scores

---

## 🔮 Future Improvements

- [ ] Integrate DistilBERT for transformer-based classification
- [ ] Add source credibility scoring (domain reputation)
- [ ] Build browser extension for real-time news checking
- [ ] Add multilingual support for Hindi fake news detection
- [ ] Deploy on cloud (Render / Railway) for public access

---

## 👩‍💻 Author

**Saniya Chatterjee**  
B.Tech Electronics & Telecommunications, Symbiosis Institute of Technology, Pune

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Email](https://img.shields.io/badge/Email-saniyaa0103@gmail.com-red?style=flat-square&logo=gmail)](mailto:saniyaa0103@gmail.com)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.