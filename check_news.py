"""
=============================================================
  Fake News Detector — Terminal Input Version
  FILE: check_news.py
  RUN:  python check_news.py

  Type any headline → get REAL or FAKE instantly
  Type 'quit' to exit
=============================================================
"""

import re, string, nltk, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import PorterStemmer

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.naive_bayes             import MultinomialNB
from sklearn.metrics                 import accuracy_score

# ── Preprocessing ─────────────────────────────────────────────
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

# ── Load & Train ──────────────────────────────────────────────
print()
print("=" * 55)
print("      TRUTHLENS — FAKE NEWS DETECTOR")
print("=" * 55)
print("\n  Loading dataset and training model...")
print("  Please wait about 30 seconds...\n")

try:
    pd.read_csv('dataset/Fake.csv')
    pd.read_csv('dataset/True.csv')
    fake['label'] = 0
    real['label'] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df = df[['text','label']].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Dataset: {len(df)} articles loaded from Kaggle CSVs")
except FileNotFoundError:
    print("  Fake.csv / True.csv not found — using built-in data")
    np.random.seed(42)
    real_t = [
        "Scientists confirm {} has significant impact on {} after trials.",
        "Government announces new policy on {} to improve {} outcomes.",
        "WHO confirms {} is safe based on {} clinical studies.",
        "Research shows {} link to {} proven in controlled study.",
    ]
    fake_t = [
        "SHOCKING: {} secretly causes {} government hiding truth!",
        "EXPOSED: {} linked to {} cover-up by deep state!",
        "Scientists BANNED from revealing {} cures {} instantly!",
        "You won't believe what {} is doing to {} media blackout!",
    ]
    topics = ["vaccines","5G","climate change","elections","COVID",
              "AI","water","food","medicine","energy"]
    subjs  = ["health","economy","politics","environment","technology",
              "security","education","rights","society","finance"]
    rows = []
    for _ in range(4000):
        lbl  = np.random.randint(0, 2)
        tmpl = np.random.choice(real_t if lbl else fake_t)
        rows.append({'text': tmpl.format(np.random.choice(topics),
                                          np.random.choice(subjs)),
                     'label': lbl})
    df = pd.DataFrame(rows)
    print(f"  Dataset: {len(df)} synthetic articles generated")

df['processed'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tr  = tfidf.fit_transform(X_train)
X_te  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_tr, y_train)
lr_acc = round(accuracy_score(y_test, lr.predict(X_te)) * 100, 1)

nb = MultinomialNB(alpha=0.1)
nb.fit(X_tr, y_train)
nb_acc = round(accuracy_score(y_test, nb.predict(X_te)) * 100, 1)

print(f"\n  Logistic Regression accuracy : {lr_acc}%")
print(f"  Naive Bayes accuracy         : {nb_acc}%")
print("\n  Models ready!")
print()
print("=" * 55)
print("  TYPE A HEADLINE AND PRESS ENTER")
print("  Ctrl+Enter not needed — just press Enter")
print("  Type  'quit'  to exit")
print("=" * 55)

# ── Warning signal checker ────────────────────────────────────
def check_signals(text):
    signals = []
    h = text.upper()
    if any(w in h for w in ['SHOCKING','EXPOSED','BANNED','HIDDEN',
                              'SECRET','COVER','THEY DON']):
        signals.append('Sensationalist language')
    if text.count('!') > 1:
        signals.append('Excessive exclamation marks')
    if sum(1 for c in text if c.isupper()) > len(text) * 0.4:
        signals.append('Excessive capitalization')
    return signals

# ── Main Loop ─────────────────────────────────────────────────
while True:
    print()
    headline = input("  Headline : ").strip()

    if headline.lower() in ['quit', 'exit', 'q']:
        print("\n  Goodbye!\n")
        break

    if headline == '':
        print("  Please type a headline.")
        continue

    cleaned  = preprocess(headline)
    features = tfidf.transform([cleaned])

    lr_pred = int(lr.predict(features)[0])
    lr_prob = lr.predict_proba(features)[0]
    lr_conf = round(float(max(lr_prob)) * 100, 1)

    nb_pred = int(nb.predict(features)[0])
    nb_prob = nb.predict_proba(features)[0]
    nb_conf = round(float(max(nb_prob)) * 100, 1)

    both_agree  = lr_pred == nb_pred
    final_label = 'REAL NEWS' if lr_pred == 1 else 'FAKE NEWS'
    avg_conf    = round((lr_conf + nb_conf) / 2, 1) if both_agree else lr_conf
    signals     = check_signals(headline)

    print()
    print("  " + "-" * 51)
    if 'FAKE' in final_label:
        print(f"  RESULT     :  *** {final_label} ***")
    else:
        print(f"  RESULT     :  {final_label}")
    print(f"  CONFIDENCE :  {avg_conf}%")
    print(f"  LR Model   :  {'REAL' if lr_pred==1 else 'FAKE'} ({lr_conf}%)")
    print(f"  NB Model   :  {'REAL' if nb_pred==1 else 'FAKE'} ({nb_conf}%)")
    if both_agree:
        print(f"  AGREEMENT  :  Both models agree")
    else:
        print(f"  AGREEMENT  :  Models disagree — LR result used")
    if signals:
        print(f"  WARNINGS   :  {' | '.join(signals)}")
    print("  " + "-" * 51)
