"""
=============================================================
  Fake News Detection using NLP and Deep Learning
  FILE: fake_news_detection.py
  RUN:  python fake_news_detection.py

=============================================================

PROBLEM STATEMENT:
  The exponential rise of digital misinformation across social
  media platforms constitutes a Big Data challenge, with millions
  of news articles published daily. Manual fact-checking is
  infeasible at this scale. This project builds an AI system
  using NLP and Deep Learning to automatically classify news
  articles as REAL or FAKE based on linguistic and semantic
  patterns in text - a clear, well-defined binary classification
  problem scoped for large-scale text corpora.
"""

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, re, string, os
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import PorterStemmer

from sklearn.model_selection     import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.naive_bayes         import MultinomialNB
from sklearn.metrics             import (accuracy_score, precision_score,
                                          recall_score, f1_score,
                                          confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import (Embedding, LSTM, Dense,
                                          Dropout, SpatialDropout1D)
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks  import EarlyStopping

# ── 1. LOAD DATASET ──────────────────────────────────────────
print("\n" + "="*60)
print("  FAKE NEWS DETECTION — FULL TRAINING PIPELINE")
print("="*60)
print("\n[STEP 1] Loading dataset...")

try:
    pd.read_csv('dataset/Fake.csv')
    pd.read_csv('dataset/True.csv')
    fake['label'] = 0
    real['label'] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df = df[['text','label']].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Loaded from Kaggle CSVs")
except FileNotFoundError:
    print("  Fake.csv / True.csv not found — using synthetic dataset")
    np.random.seed(42)
    real_t = [
        "Scientists confirm {} has significant impact on {} after trials.",
        "Government announces new policy on {} to improve {} outcomes.",
        "WHO report confirms {} is safe based on {} clinical studies.",
        "Research published in journal shows {} link to {} is proven.",
        "University study finds {} reduces {} risk by significant margin.",
    ]
    fake_t = [
        "SHOCKING: {} secretly causes {} — government hiding the truth!",
        "EXPOSED: {} linked to massive {} cover-up by deep state!",
        "Scientists BANNED from revealing {} cures {} instantly!",
        "You won't believe what {} is doing to {} — media blackout!",
        "BREAKING: {} admits {} in leaked documents the elite hide!",
    ]
    topics  = ["vaccines","5G towers","climate change","elections","COVID-19",
               "artificial intelligence","water supply","food additives"]
    subjs   = ["public health","economy","global politics","environment",
               "technology","national security","education","human rights"]
    rows = []
    for _ in range(4000):
        lbl  = np.random.randint(0, 2)
        tmpl = np.random.choice(real_t if lbl else fake_t)
        text = tmpl.format(np.random.choice(topics), np.random.choice(subjs))
        rows.append({'text': text, 'label': lbl})
    df = pd.DataFrame(rows)

print(f"  Total articles : {len(df)}")
print(f"  Real news      : {df['label'].sum()}")
print(f"  Fake news      : {(df['label']==0).sum()}")

# ── 2. TEXT PREPROCESSING ────────────────────────────────────
print("\n[STEP 2] Preprocessing text...")

stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

df['processed'] = df['text'].apply(preprocess)
print("  Done.")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────
print("\n[STEP 3] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. TF-IDF VECTORIZATION ──────────────────────────────────
print("\n[STEP 4] TF-IDF Vectorization...")
tfidf         = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  Feature matrix: {X_train_tfidf.shape}")

# ── 5. CLASSIFICATION ALGORITHMS ────────────────────────────
print("\n[STEP 5] Training Classification Algorithms...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

lr_acc  = round(accuracy_score (y_test, y_pred_lr) * 100, 2)
lr_prec = round(precision_score(y_test, y_pred_lr) * 100, 2)
lr_rec  = round(recall_score   (y_test, y_pred_lr) * 100, 2)
lr_f1   = round(f1_score       (y_test, y_pred_lr) * 100, 2)

print(f"\n  Logistic Regression:")
print(f"    Accuracy  : {lr_acc}%")
print(f"    Precision : {lr_prec}%")
print(f"    Recall    : {lr_rec}%")
print(f"    F1-Score  : {lr_f1}%")

# Naive Bayes
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

nb_acc  = round(accuracy_score (y_test, y_pred_nb) * 100, 2)
nb_prec = round(precision_score(y_test, y_pred_nb) * 100, 2)
nb_rec  = round(recall_score   (y_test, y_pred_nb) * 100, 2)
nb_f1   = round(f1_score       (y_test, y_pred_nb) * 100, 2)

print(f"\n  Naive Bayes:")
print(f"    Accuracy  : {nb_acc}%")
print(f"    Precision : {nb_prec}%")
print(f"    Recall    : {nb_rec}%")
print(f"    F1-Score  : {nb_f1}%")

# ── 6. DEEP LEARNING — LSTM ──────────────────────────────────
print("\n[STEP 6] Training LSTM Deep Learning Model...")

MAX_VOCAB  = 10000
MAX_LEN    = 100
EMBED_DIM  = 64
LSTM_UNITS = 64

tokenizer_dl = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer_dl.fit_on_texts(X_train)

X_train_seq = tokenizer_dl.texts_to_sequences(X_train)
X_test_seq  = tokenizer_dl.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post')

lstm_model = Sequential([
    Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
print("\n  LSTM Architecture:")
lstm_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3,
                            restore_best_weights=True)
history = lstm_model.fit(
    X_train_pad, y_train.values,
    epochs=10, batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob  = lstm_model.predict(X_test_pad, verbose=0)
y_pred_lstm  = (y_pred_prob > 0.5).astype(int).flatten()

lstm_acc  = round(accuracy_score (y_test, y_pred_lstm) * 100, 2)
lstm_prec = round(precision_score(y_test, y_pred_lstm) * 100, 2)
lstm_rec  = round(recall_score   (y_test, y_pred_lstm) * 100, 2)
lstm_f1   = round(f1_score       (y_test, y_pred_lstm) * 100, 2)

print(f"\n  LSTM Results:")
print(f"    Accuracy  : {lstm_acc}%")
print(f"    Precision : {lstm_prec}%")
print(f"    Recall    : {lstm_rec}%")
print(f"    F1-Score  : {lstm_f1}%")

# ── 7. RESULTS GRAPH ─────────────────────────────────────────
print("\n[STEP 7] Saving results graph...")

os.makedirs('outputs', exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fake News Detection — Results Dashboard',
             fontsize=16, fontweight='bold')

labels = ['Fake', 'Real']

# 1 — Label distribution
axes[0,0].pie(
    [sum(y_test==0), sum(y_test==1)],
    labels=labels, autopct='%1.1f%%',
    colors=['#ef4444','#22c55e'], startangle=140
)
axes[0,0].set_title('Dataset Label Distribution')

# 2 — Confusion Matrix LR
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0,1])
axes[0,1].set_title('Confusion Matrix — Logistic Regression')
axes[0,1].set_xlabel('Predicted'); axes[0,1].set_ylabel('Actual')

# 3 — Confusion Matrix LSTM
cm_lstm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=axes[0,2])
axes[0,2].set_title('Confusion Matrix — LSTM')
axes[0,2].set_xlabel('Predicted'); axes[0,2].set_ylabel('Actual')

# 4 — Model comparison bar chart
model_names = ['Logistic\nRegression', 'Naive\nBayes', 'LSTM']
accs  = [lr_acc,  nb_acc,  lstm_acc]
precs = [lr_prec, nb_prec, lstm_prec]
recs  = [lr_rec,  nb_rec,  lstm_rec]
f1s   = [lr_f1,   nb_f1,   lstm_f1]

x = np.arange(len(model_names)); w = 0.2
axes[1,0].bar(x - 1.5*w, accs,  w, label='Accuracy',  color='#3b82f6', alpha=0.85)
axes[1,0].bar(x - 0.5*w, precs, w, label='Precision', color='#8b5cf6', alpha=0.85)
axes[1,0].bar(x + 0.5*w, recs,  w, label='Recall',    color='#f59e0b', alpha=0.85)
axes[1,0].bar(x + 1.5*w, f1s,   w, label='F1-Score',  color='#22c55e', alpha=0.85)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(model_names)
axes[1,0].set_ylim(0, 115)
axes[1,0].legend(fontsize=9)
axes[1,0].set_title('Model Performance Comparison (%)')
for i, (a,p,r,f) in enumerate(zip(accs,precs,recs,f1s)):
    axes[1,0].text(i-1.5*w, a+1, str(a), ha='center', fontsize=7)

# 5 — LSTM Training Accuracy
axes[1,1].plot(history.history['accuracy'],
               label='Train Accuracy', color='#3b82f6', linewidth=2)
axes[1,1].plot(history.history['val_accuracy'],
               label='Val Accuracy',   color='#ef4444', linewidth=2)
axes[1,1].set_title('LSTM Training Accuracy')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Accuracy')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

# 6 — LSTM Training Loss
axes[1,2].plot(history.history['loss'],
               label='Train Loss', color='#22c55e', linewidth=2)
axes[1,2].plot(history.history['val_loss'],
               label='Val Loss',   color='#f59e0b', linewidth=2)
axes[1,2].set_title('LSTM Training Loss')
axes[1,2].set_xlabel('Epoch'); axes[1,2].set_ylabel('Loss')
axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

plt.tight_layout()
graph_path = 'outputs/results_dashboard.png'
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
print(f"  Graph saved to: {graph_path}")

# ── 8. SAMPLE PREDICTIONS ────────────────────────────────────
print("\n[STEP 8] Sample Predictions:")
print("-"*60)

test_headlines = [
    "Scientists from WHO confirm COVID vaccine is safe and effective after global clinical trials.",
    "SHOCKING: Government secretly hiding alien cure — doctors BANNED from speaking out!"
]

for h in test_headlines:
    cleaned  = preprocess(h)
    feat     = tfidf.transform([cleaned])
    lr_p     = lr.predict(feat)[0]
    lr_c     = round(max(lr.predict_proba(feat)[0]) * 100, 1)
    seq      = tokenizer_dl.texts_to_sequences([cleaned])
    pad      = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob     = lstm_model.predict(pad, verbose=0)[0][0]
    lstm_p   = 'REAL' if prob > 0.5 else 'FAKE'
    lstm_c   = round(max(prob, 1-prob) * 100, 1)

    print(f"\n  Headline : {h[:70]}...")
    print(f"  LR       : {'REAL' if lr_p==1 else 'FAKE'} ({lr_c}%)")
    print(f"  LSTM     : {lstm_p} ({lstm_c}%)")

print("\n" + "="*60)
print("  DONE — Training complete.")
print("  Logistic Regression, Naive Bayes, and LSTM evaluated.")
print("="*60 + "\n")