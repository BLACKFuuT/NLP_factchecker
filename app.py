# app.py (redesigned UI: news portal dark theme + tabs) — optimized version
import os
import io
import csv
import time
import random
import logging
from typing import Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote_plus
from ftfy import fix_text

# ML & NLP libs
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from scipy import sparse as sp_sparse

# ---------------------------
# Config & constants (tuned for speed)
# ---------------------------
SCRAPED_DATA_PATH = "politifact_scraped.csv"
N_SPLITS = 3                    # fewer CV folds -> faster runs
MAX_PAGES = 20                  # limit scraping by default
REQUEST_RETRIES = 1             # avoid long retry stalls
REQUEST_BACKOFF = 1             # shorter backoff
DEFAULT_TIMEOUT = 8             # shorter HTTP timeout (seconds)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prefer secrets; fallback to None (app warns if missing)
API_KEY = st.secrets.get("GOOGLE_FACTCHECK_API_KEY") if hasattr(st, "secrets") else None

# ---------------------------
# Utility helpers
# ---------------------------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()

def safe_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    backoff = REQUEST_BACKOFF
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{REQUEST_RETRIES}) for {url}: {e}")
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
    return None

# ---------------------------
# Fact Check integration
# ---------------------------
def get_fact_check_results(query: str):
    """Fetch fact-check results for the given query from Google Fact Check Tools API.
    Returns list of dicts with keys: publisher, title, rating, url"""
    if not API_KEY:
        return [{"publisher": "Error", "title": "No API key configured (add GOOGLE_FACTCHECK_API_KEY to Streamlit secrets).", "rating": "", "url": ""}]
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        claims = data.get("claims", [])
        results = []
        for claim in claims:
            for r in claim.get("claimReview", []):
                results.append({
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "title": r.get("title", ""),
                    "rating": r.get("textualRating", "No Rating"),
                    "url": r.get("url", "")
                })
        if not results:
            return []
        return results
    except requests.exceptions.HTTPError as he:
        try:
            err = he.response.json().get("error", {}).get("message", str(he))
        except Exception:
            err = str(he)
        return [{"publisher": "Error", "title": err, "rating": "", "url": ""}]
    except Exception as e:
        return [{"publisher": "Error", "title": str(e), "rating": "", "url": ""}]

# ---------------------------
# spaCy lazy loader (so app doesn't stop if model missing)
# ---------------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def get_nlp_model():
    try:
        return load_spacy_model()
    except OSError:
        # warn but do not stop the app
        st.warning(
            "spaCy model 'en_core_web_sm' not found. Install it to use Scraper / Model Showdown features.\n\n"
            "Run:\n"
            "pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )
        return None

# ---------------------------
# Scraper & features
# ---------------------------
@st.cache_data(ttl=60*60*24)
def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    seen_urls = set()
    rows = []
    page_count = 0

    while current_url and page_count < MAX_PAGES:
        page_count += 1
        if current_url in seen_urls:
            logger.info("Detected repeated page, stopping to avoid infinite loop.")
            break
        seen_urls.add(current_url)

        resp = safe_get(current_url, timeout=DEFAULT_TIMEOUT)
        if resp is None:
            st.warning(f"Failed to fetch {current_url} after retries; stopping scraper.")
            break

        try:
            resp.encoding = resp.apparent_encoding
        except Exception:
            pass

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all("li", class_="o-listicle__item")
        if not items:
            logger.info("No items found on page; stopping.")
            break

        stop_if_older = False
        for card in items:
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(" ", strip=True) if date_div else ""
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format="%B %d, %Y")
                    except Exception:
                        claim_date = pd.to_datetime(match.group(1), errors='coerce')

            if claim_date is None:
                continue

            if claim_date < start_date:
                stop_if_older = True
                break

            if not (start_date <= claim_date <= end_date):
                continue

            statement = None
            statement_block = card.find("div", class_="m-statement__quote")
            if statement_block:
                a = statement_block.find("a", href=True)
                if a:
                    statement = clean(a.get_text(" ", strip=True))

            source = None
            source_a = card.find("a", class_="m-statement__name")
            if source_a:
                source = clean(source_a.get_text(" ", strip=True))

            author = None
            footer = card.find("footer", class_="m-statement__footer")
            if footer:
                text = footer.get_text(" ", strip=True)
                m = re.search(r"By\s+([^•\n\r]+)", text)
                if m:
                    author = clean(m.group(1).strip())
                else:
                    parts = text.split("•")
                    if parts:
                        author = clean(parts[0].replace("By", "").strip())

            label = None
            label_img = card.find("img", alt=True)
            if label_img and 'alt' in label_img.attrs:
                label = clean(label_img['alt'].replace('-', ' ').title())

            rows.append({
                "author": author,
                "statement": statement,
                "source": source,
                "date": claim_date.strftime("%Y-%m-%d"),
                "label": label
            })

        if stop_if_older:
            break

        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and next_link.get("href"):
            current_url = urljoin(base_url, next_link['href'])
        else:
            break

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["statement", "label"])
    if not df.empty:
        df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ---------------------------
# Feature extraction & modeling helpers (with smaller vectorizers)
# ---------------------------
stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def lexical_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        toks = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        processed.append(" ".join(toks))
    return processed

def syntactic_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        pos = " ".join([token.pos_ for token in doc])
        processed.append(pos)
    return processed

def semantic_features_batch(texts: List[str]) -> pd.DataFrame:
    out = []
    for t in texts:
        b = TextBlob(t)
        out.append([b.sentiment.polarity, b.sentiment.subjectivity])
    return pd.DataFrame(out, columns=["polarity", "subjectivity"])

def discourse_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        sents = [sent.text.strip() for sent in doc.sents]
        first_words = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
        processed.append(f"{len(sents)} {first_words}")
    return processed

def pragmatic_features_batch(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([tl.count(w) for w in pragmatic_words])
    return pd.DataFrame(rows, columns=pragmatic_words)

def apply_feature_extraction(X_series: pd.Series, phase: str, nlp) -> Tuple[np.ndarray, Optional[object]]:
    """Return (feature_matrix, vectorizer_or_None). Feature matrix may be sparse."""
    X_texts = X_series.astype(str).tolist()
    # Note: vectorizer params reduced (max_features/min_df) for speed
    if phase == "Lexical & Morphological":
        X_proc = lexical_features_batch(X_texts, nlp)
        vect = CountVectorizer(binary=True, ngram_range=(1,2), min_df=3, max_features=2000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Syntactic":
        X_proc = syntactic_features_batch(X_texts, nlp)
        vect = TfidfVectorizer(max_features=2000, min_df=3)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Semantic":
        df = semantic_features_batch(X_texts)
        return df.values, None

    if phase == "Discourse":
        X_proc = discourse_features_batch(X_texts, nlp)
        vect = CountVectorizer(ngram_range=(1,2), max_features=2000, min_df=2)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Pragmatic":
        df = pragmatic_features_batch(X_texts)
        return df.values, None

    raise ValueError("Unknown phase")

def get_models_dict():
    return {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', random_state=42, class_weight='balanced', n_jobs=-1),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced', probability=False)
    }

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def map_label(l):
        if pd.isna(l):
            return np.nan
        l = str(l).strip()
        if l in REAL_LABELS:
            return 1
        if l in FAKE_LABELS:
            return 0
        low = l.lower()
        if "true" in low and "mostly" not in low and "half" not in low:
            return 1
        if "false" in low or "pants" in low or "fire" in low:
            return 0
        return np.nan

    df = df.copy()
    df["target_label"] = df["label"].apply(map_label)
    return df

# ---------------------------
# New: precompute & cache features once per phase
# ---------------------------
@st.cache_data(ttl=60*60)
def compute_features_for_phase(df_statements: pd.Series, selected_phase: str):
    """Compute features for the entire dataset once and cache them."""
    nlp = get_nlp_model()
    if nlp is None:
        raise RuntimeError("spaCy model not available")
    t0 = time.time()
    X_feat, vect = apply_feature_extraction(df_statements, selected_phase, nlp)
    logger.info("Computed features for phase '%s' in %.3fs", selected_phase, time.time() - t0)
    return X_feat, vect

def evaluate_models(df: pd.DataFrame, selected_phase: str, nlp) -> pd.DataFrame:
    """Evaluate multiple models using precomputed features + StratifiedKFold slicing."""
    df = create_binary_target(df)
    df = df.dropna(subset=["target_label"])
    df = df[df["statement"].astype(str).str.len() > 10]

    X_raw = df["statement"].astype(str)
    y_raw = df["target_label"].astype(int)

    if len(np.unique(y_raw)) < 2:
        st.error("Only one class present after mapping — adjust data or date range.")
        return pd.DataFrame()

    # Precompute features once for the whole dataset
    try:
        X_full, vectorizer = compute_features_for_phase(X_raw, selected_phase)
    except RuntimeError as rexc:
        st.error(str(rexc))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Feature precomputation failed: {e}")
        return pd.DataFrame()

    models = get_models_dict()
    results = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    n_samples = len(y_raw)

    # Utility to index into X_full whether it's sparse or numpy
    def slice_X(X, idx):
        if sp_sparse.issparse(X):
            return X[idx]
        else:
            return X[idx]

    for name, model in models.items():
        st.caption(f"Training {name}...")
        fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], []
        train_times, infer_times = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples), y_raw)):
            # Slice precomputed features
            try:
                if vectorizer is not None:
                    # X_full is already transformed (sparse)
                    X_train = slice_X(X_full, train_idx)
                    X_test = slice_X(X_full, test_idx)
                else:
                    # numeric features (numpy arrays)
                    X_train = slice_X(X_full, train_idx)
                    X_test = slice_X(X_full, test_idx)

                y_train = y_raw.values[train_idx]
                y_test = y_raw.values[test_idx]

                start_train = time.time()
                if name == "Naive Bayes":
                    # NB expects dense non-negative features — convert if needed
                    if sp_sparse.issparse(X_train):
                        Xt_fit = X_train.toarray().astype(float)
                    else:
                        Xt_fit = np.abs(X_train).astype(float)
                    model.fit(Xt_fit, y_train)
                    clf = model
                else:
                    # Use SMOTE with n_jobs where supported
                    sm = SMOTE(random_state=42, k_neighbors=3, n_jobs=-1)
                    pipeline = ImbPipeline([("smote", sm), ("clf", model)])
                    pipeline.fit(X_train, y_train)
                    clf = pipeline

                train_time = time.time() - start_train
                start_inf = time.time()

                # For dense or sparse, predict should work with pipeline
                if sp_sparse.issparse(X_test) and hasattr(clf, "predict"):
                    y_pred = clf.predict(X_test)
                else:
                    y_pred = clf.predict(X_test)

                infer_time = (time.time() - start_inf) * 1000.0

                fold_acc.append(accuracy_score(y_test, y_pred))
                fold_f1.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_prec.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_rec.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                train_times.append(train_time)
                infer_times.append(infer_time)
            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                fold_acc.append(0); fold_f1.append(0); fold_prec.append(0); fold_rec.append(0)
                train_times.append(0); infer_times.append(9999)

        results.append({
            "Model": name,
            "Accuracy": np.mean(fold_acc) * 100,
            "F1-Score": np.mean(fold_f1),
            "Precision": np.mean(fold_prec),
            "Recall": np.mean(fold_rec),
            "Training Time (s)": round(np.mean(train_times), 3),
            "Inference Latency (ms)": round(np.mean(infer_times), 3)
        })

    return pd.DataFrame(results)

# ---------------------------
# Humorous critiques (unchanged)
# ---------------------------
def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It’s concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It’s the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It’s the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It’s the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))

def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."
    df_results = df_results.copy()
    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_idx = df_results['F1-Score'].idxmax()
    best_model_row = df_results.loc[best_idx]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    headline = f"👑 The Golden Snitch Award goes to the {best_model}!"
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    return summary + roast

# ---------------------------
# STREAMLIT APP UI (tabs + dark theme)
# ---------------------------
def app():
    st.set_page_config(page_title='AI vs. Fact: NLP Comparator', layout='wide')

    # Initialize session state defaults
    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'df_results' not in st.session_state:
        st.session_state['df_results'] = pd.DataFrame()
    if 'selected_phase_run' not in st.session_state:
        st.session_state['selected_phase_run'] = ""

    # CSS for dark news portal look
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg,#05070a,#0b0f12); color: #e9f4ff; }
        .topbar { background: linear-gradient(90deg,#071029,#3b0f34); padding:18px;border-radius:10px;margin-bottom:14px;box-shadow:0 10px 30px rgba(0,0,0,0.6);}
        .topbar h1 { margin:0; font-size:2.2rem; color: #fff; }
        .topbar p { margin:2px 0 0 0; color:#bcd7ef; opacity:0.9; }
        .panel { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
        .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px;border-radius:10px;margin-bottom:12px;border:1px solid rgba(255,255,255,0.03); }
        .small { color:#9fb1c6; font-size:0.85rem; }
        .muted { color:#99aebf; }
        .link-button { display:inline-block; padding:8px 12px; border-radius:8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); color: #dceefc; text-decoration: none; margin-top:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="topbar"><h1>AI vs. Fact — Global FactCheck Network</h1><p>Portal: Scrape, Train, Evaluate, and Cross-check claims with verified sources.</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Home", "Scraper", "Model Showdown", "Fact Check"])

    # Home
    with tabs[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Welcome")
        st.write("This portal helps you build models to identify factual claims and cross-check statements against verified fact-check sources.")
        st.write("Use the tabs to the right to scrape Politifact, run model benchmarks, or quickly check individual claims using Google Fact Check Tools.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Scraper
    with tabs[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Politifact Scraper")
        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
        if st.button("Scrape Politifact Data ⛏️", key="scrape_btn"):
            if start_date > end_date:
                st.error("Error: Start Date must be before or equal to End Date.")
            else:
                with st.spinner("Scraping..."):
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                    if scraped_df.empty:
                        st.warning("No data scraped — try narrowing the date range or check the target site structure.")
                    else:
                        st.session_state['scraped_df'] = scraped_df
                        st.success(f"Scraping complete! {len(scraped_df)} claims harvested.")
                        st.download_button(
                            "Download scraped CSV",
                            scraped_df.to_csv(index=False).encode('utf-8'),
                            file_name="politifact_scraped.csv",
                            mime="text/csv"
                        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Model Showdown
    with tabs[2]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Model Showdown — Train & Evaluate")
        if st.session_state['scraped_df'].empty:
            st.info("No scraped dataset loaded. Scrape data first in the 'Scraper' tab.")
        else:
            df = st.session_state['scraped_df']
            st.write(f"Loaded dataset with {len(df)} rows.")
            phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
            selected_phase = st.selectbox("Choose the Feature Set (NLP Phase):", phases, key="phase_select")
            if st.button("Analyze Model Showdown 🥊", key="analyze_btn"):
                # lazy-load model
                nlp = get_nlp_model()
                if nlp is None:
                    st.error("spaCy model not available. Install 'en_core_web_sm' to run Model Showdown.")
                else:
                    with st.spinner(f"Precomputing features for {selected_phase}..."):
                        # compute_features_for_phase will be used inside evaluate_models; call directly to show progress and cache
                        try:
                            # do one warm compute so the cache entry is created and user sees progress immediately
                            _ = compute_features_for_phase(df["statement"].astype(str), selected_phase)
                        except Exception as e:
                            st.error(f"Feature computation failed: {e}")
                            _ = None

                    with st.spinner(f"Training models using {selected_phase} features..."):
                        df_results = evaluate_models(df, selected_phase, nlp)
                        st.session_state['df_results'] = df_results
                        st.session_state['selected_phase_run'] = selected_phase
                        if not df_results.empty:
                            st.success("Analysis complete! Results ready.")
                        else:
                            st.warning("Analysis returned no results. Check logs or data.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Fact Check tab (search UI moved here)
    with tabs[3]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Cross-Platform Fact Check")
        if not API_KEY:
            st.warning("No Google Fact Check API key configured. Add GOOGLE_FACTCHECK_API_KEY to Streamlit secrets to enable cross-checking.")
        user_query = st.text_input("Enter a claim or statement to fact-check:", key="sidebar_query")
        if st.button("Check Fact Credibility", key="factcheck_btn"):
            if not user_query.strip():
                st.warning("Please enter a statement to check.")
            else:
                with st.spinner("Fetching verified fact-checks..."):
                    results = get_fact_check_results(user_query)
                if not results:
                    st.info("No verified fact-checks found for this claim.")
                else:
                    # If one of the results is an error dict from our function, show it prominently
                    if len(results) == 1 and results[0].get("publisher") == "Error":
                        st.error(results[0].get("title", "Unknown API error"))
                    else:
                        st.success(f"Found {len(results)} fact-check result(s):")
                        for r in results[:10]:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            # safe rendering: use st.markdown with sanitized text inside spans
                            pub_text = r.get('publisher', 'Unknown')
                            rating_text = r.get('rating', 'No Rating')
                            title_text = r.get('title', '')
                            url = r.get("url", "#")
                            st.markdown(f"**Source:** <span class='small'>{st.text(pub_text)}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Verdict:** <span class='muted'>{st.text(rating_text)}</span>", unsafe_allow_html=True)
                            if title_text:
                                st.markdown(f"**{st.text(title_text)}**")
                            st.markdown(f'<a href="{url}" target="_blank" class="link-button">Read article</a>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar quick-check convenience (duplicate of tab but helpful)
    st.sidebar.subheader("🔍 Quick Fact Check")
    sidebar_query = st.sidebar.text_input("Claim or headline", key="sidebar_quick_query")
    if st.sidebar.button("Check (Sidebar)"):
        q = (sidebar_query or "").strip()
        if not q:
            st.sidebar.warning("Please enter a claim or headline.")
        else:
            results = get_fact_check_results(q)
            if len(results) == 1 and results[0].get("publisher") == "Error":
                st.sidebar.error(results[0].get("title", "API error"))
            elif not results:
                st.sidebar.info("No fact-checks found.")
            else:
                for r in results[:5]:
                    st.sidebar.markdown(f"**{r.get('publisher','Unknown')}** — {r.get('rating','No Rating')}")
                    st.sidebar.markdown(f"[Read]({r.get('url','#')})")

    # Metrics & critique area available via session_state
    if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Latest Run — Summary")
        df_results = st.session_state['df_results']
        st.dataframe(df_results[['Model','Accuracy','F1-Score','Training Time (s)','Inference Latency (ms)']], height=220, use_container_width=True)
        # Download results
        st.download_button(
            "Download Results CSV",
            df_results.to_csv(index=False).encode('utf-8'),
            file_name="model_results.csv",
            mime="text/csv"
        )
        # simple plot
        plot_metric = st.selectbox("Metric to Plot:", ['Accuracy','F1-Score','Precision','Recall','Training Time (s)','Inference Latency (ms)'], index=1)
        df_plot = df_results[['Model', plot_metric]].set_index('Model')
        st.bar_chart(df_plot)
        # humorous critique
        if st.button("Generate Humorous Critique"):
            critique_text = generate_humorous_critique(df_results, st.session_state.get('selected_phase_run', 'Unknown'))
            st.markdown(critique_text)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
