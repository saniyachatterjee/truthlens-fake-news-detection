"""
=============================================================
  Fake News Detection — Web Dashboard Backend  (FIXED)
  FILE: app.py
  RUN:  python app.py
  OPEN: http://localhost:8000

  FIXES APPLIED:
  1. Train on TITLES (not full article text) — matches what
     users actually type as headlines.
  2. Also include a combined title+text model for richer signal.
  3. Raised NB alpha 0.1 → 1.0 to handle short/sparse inputs.
  4. Added minimum-token guard: if preprocessed text is too
     short after cleaning, return a low-confidence warning
     instead of a biased prediction.
  5. Final verdict uses majority vote (LR + NB), not LR alone.
  6. Confidence is now calibrated: short inputs get capped at
     70% to prevent misleadingly high fake scores.
=============================================================
"""

from flask import Flask, request, jsonify, render_template
import re, string, nltk, warnings, os
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
from datetime import datetime

app = Flask(__name__)

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

# ── Global state ──────────────────────────────────────────────
state = {
    'ready':    False,
    'lr_acc':   0,
    'nb_acc':   0,
    'total':    0,
    'history':  []
}
tfidf    = None
lr_model = None
nb_model = None

# ── Train on startup ──────────────────────────────────────────
def train():
    global tfidf, lr_model, nb_model, state
    print("\n  Loading dataset...")

    try:
        pd.read_csv('dataset/Fake.csv')
        pd.read_csv('dataset/True.csv')
        fake['label'] = 0
        real['label'] = 1

        # ── FIX 1: Use TITLE column (headlines) for training ──
        # The original code used 'text' (full articles), but users
        # input short headlines. Training on titles makes the model
        # match what it will actually receive at inference time.
        #
        # We also build a combined field: title + first 200 chars
        # of text, so the model still benefits from article context.

        def make_input(row):
            title = str(row.get('title', '') or '')
            text  = str(row.get('text',  '') or '')
            # Use title + short lead sentence for richer signal
            return (title + ' ' + text[:200]).strip()

        fake['input'] = fake.apply(make_input, axis=1)
        real['input'] = real.apply(make_input, axis=1)

        df = pd.concat([
            fake[['input','label']],
            real[['input','label']]
        ], ignore_index=True)
        df = df.dropna(subset=['input'])
        df = df[df['input'].str.strip() != '']
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Loaded {len(df)} articles from Kaggle CSVs (title+lead)")

    except FileNotFoundError:
        print("  Fake.csv / True.csv not found — using built-in data")
        np.random.seed(42)
        real_t = [
            "Scientists confirm {} has significant impact on {} after trials.",
            "Government announces new policy on {} to improve {} outcomes.",
            "WHO confirms {} is safe based on {} clinical studies.",
            "Research shows {} link to {} proven in study.",
            "{} officials announce measures to address {} concerns.",
            "New report highlights progress in {} amid growing {} debate.",
        ]
        fake_t = [
            "SHOCKING: {} secretly causes {} government hiding truth!",
            "EXPOSED: {} linked to {} cover-up by deep state!",
            "Scientists BANNED from revealing {} cures {} instantly!",
            "You won't believe what {} is doing to {} media blackout!",
            "BREAKING: Elite hiding {} truth about {} from public!",
            "THEY DON'T WANT YOU TO KNOW: {} destroys {} overnight!",
        ]
        topics = ["vaccines","5G","climate change","elections","COVID",
                  "AI","water supply","food","medicine","energy"]
        subjs  = ["health","economy","politics","environment","technology",
                  "security","education","rights","society","finance"]
        rows = []
        for _ in range(4000):
            lbl  = np.random.randint(0, 2)
            tmpl = np.random.choice(real_t if lbl else fake_t)
            rows.append({'input': tmpl.format(np.random.choice(topics),
                                              np.random.choice(subjs)),
                         'label': lbl})
        df = pd.DataFrame(rows)

    state['total'] = len(df)
    print("  Preprocessing...")
    df['processed'] = df['input'].apply(preprocess)

    # Drop rows that became empty after preprocessing
    df = df[df['processed'].str.strip() != '']

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr   = tfidf.fit_transform(X_train)
    Xte   = tfidf.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(Xtr, y_train)

    # ── FIX 2: Raise NB alpha 0.1 → 1.0 ──────────────────────
    # alpha=0.1 (very low smoothing) causes NB to be extremely
    # sensitive to rare/unseen tokens, which biases short inputs
    # toward the majority class. alpha=1.0 (Laplace smoothing) is
    # the standard default and handles short text far better.
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(Xtr, y_train)

    state['lr_acc'] = round(accuracy_score(y_test, lr_model.predict(Xte)) * 100, 1)
    state['nb_acc'] = round(accuracy_score(y_test, nb_model.predict(Xte)) * 100, 1)
    state['ready']  = True
    print(f"  LR: {state['lr_acc']}% | NB: {state['nb_acc']}%")
    print("  Web app ready at http://localhost:5000\n")

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify(state)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not state['ready']:
        return jsonify({'error': 'Model not ready'}), 503

    data     = request.json
    headline = data.get('headline', '').strip()
    if not headline:
        return jsonify({'error': 'No headline provided'}), 400

    cleaned  = preprocess(headline)
    features = tfidf.transform([cleaned])

    # ── FIX 3: Minimum token guard ────────────────────────────
    # After stemming + stopword removal, very short text produces
    # near-zero TF-IDF vectors. A prediction on an empty/1-token
    # vector is essentially random but looks confident. We flag it.
    token_count  = len(cleaned.split())
    too_short    = token_count < 3

    lr_pred = int(lr_model.predict(features)[0])
    lr_prob = lr_model.predict_proba(features)[0]
    lr_conf = round(float(max(lr_prob)) * 100, 1)

    nb_pred = int(nb_model.predict(features)[0])
    nb_prob = nb_model.predict_proba(features)[0]
    nb_conf = round(float(max(nb_prob)) * 100, 1)

    # ── FIX 4: Majority-vote verdict ──────────────────────────
    # Original code always used LR alone. With 2 models, use a
    # simple vote: if they agree → use that verdict; if not →
    # pick whichever has higher confidence rather than defaulting
    # to LR blindly.
    both_agree = lr_pred == nb_pred
    if both_agree:
        final_pred = lr_pred
        avg_conf   = round((lr_conf + nb_conf) / 2, 1)
    else:
        # Models disagree — pick the one with higher confidence
        if lr_conf >= nb_conf:
            final_pred = lr_pred
            avg_conf   = lr_conf
        else:
            final_pred = nb_pred
            avg_conf   = nb_conf

    # ── FIX 5: Cap confidence for short/ambiguous inputs ──────
    # Short headlines produce sparse vectors → inflated confidence
    # numbers that are statistically unreliable. Cap at 70%.
    if too_short:
        avg_conf = min(avg_conf, 70.0)
        lr_conf  = min(lr_conf,  70.0)
        nb_conf  = min(nb_conf,  70.0)

    # Warning signals (unchanged)
    signals = []
    h = headline.upper()
    if any(w in h for w in ['SHOCKING','EXPOSED','BANNED','HIDDEN','SECRET','COVER']):
        signals.append('Sensationalist language')
    if headline.count('!') > 1:
        signals.append('Excessive exclamation marks')
    if sum(1 for c in headline if c.isupper()) > len(headline) * 0.4:
        signals.append('Excessive capitalization')
    if any(w in h for w in ["WON'T BELIEVE","THEY DON'T WANT","DOCTORS HATE"]):
        signals.append('Clickbait phrasing')
    if too_short:
        signals.append('Headline too short for reliable analysis')

    result = {
        'headline':    headline,
        'verdict':     'REAL' if final_pred == 1 else 'FAKE',
        'confidence':  avg_conf,
        'lr_verdict':  'REAL' if lr_pred == 1 else 'FAKE',
        'lr_conf':     lr_conf,
        'nb_verdict':  'REAL' if nb_pred == 1 else 'FAKE',
        'nb_conf':     nb_conf,
        'both_agree':  both_agree,
        'signals':     signals,
        'timestamp':   datetime.now().strftime('%H:%M:%S'),
        'token_count': token_count,
    }

    state['history'].insert(0, result)
    state['history'] = state['history'][:20]
    return jsonify(result)

@app.route('/api/history')
def history():
    return jsonify(state['history'])

if __name__ == '__main__':
    print("=" * 55)
    print("  TRUTHLENS — WEB DASHBOARD  (FIXED)")
    print("=" * 55)
    train()
    app.run(debug=False, host='0.0.0.0', port=8000)
