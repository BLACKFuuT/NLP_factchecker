# app.py (redesigned UI: news portal dark theme + tabs) — startup-optimized version
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
import importlib

from scipy import sparse as sp_sparse

# ---------------------------
# Config & constants (tuned for speed)
# ---------------------------
SCRAPED_DATA_PATH = "politifact_scraped.csv"
N_SPLITS = 3                    # fewer CV folds -> faster runs
MAX_PAGES = 5                   # <- smaller default to speed scraping by default
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
# spaCy lazy loader (so app doesn't block startup if model missing)
# ---------------------------
@st.cache_resource
def load_spacy_model():
    """
    Load the spaCy model only when requested. This is cached with Streamlit so repeated calls are fast.
    """
    spacy = importlib.import_module("spacy")
    # attempt to load small english model
    return spacy.load("en_core_web_sm")

def get_nlp_model():
    """
    Try to return a spaCy model. If not installed, return None (we have lightweight fallbacks).
    """
    try:
        return load_spacy_model()
    except Exception as exc:
        # Do not block the app — return None and let feature code use faster fallbacks.
        logger.info("spaCy model not available at startup: %s", exc)
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
# Lightweight fallback tokenization & fast heuristics (used when spaCy isn't present)
# ---------------------------
def simple_tokenize_to_lemmas(texts: List[str]) -> List[str]:
    """Very fast fallback tokenization: lowercase, remove non-alpha, simple stemming via suffix stripping."""
    out = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z\s]", " ", t).lower()
        toks = [w for w in t.split() if len(w) > 2]
        # a cheap "stem": drop common suffixes (not a real stemmer but fast)
        toks = [re.sub(r"(ing|ed|ly|s)$", "", w) for w in toks]
        out.append(" ".join(toks))
    return out

def simple_pos_features(texts: List[str]) -> List[str]:
    """Very fast POS-like features by counting uppercase words and punctuation marks as a proxy."""
    out = []
    for t in texts:
        caps = sum(1 for c in t if c.isupper())
        q = t.count("?")
        ex = t.count("!")
        out.append(f"CAPS_{caps}_Q_{q}_EX_{ex}")
    return out

# ---------------------------
# Feature extraction & modeling helpers (lazy imports for faster start)
# ---------------------------
stop_words = None
try:
    # small import attempt (cheap). If spacy isn't present this is fine.
    spacy_mod = importlib.import_module("spacy")
    stop_words = importlib.import_module("spacy.lang.en.stop_words").STOP_WORDS
except Exception:
    stop_words = set()

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def lexical_features_batch(texts: List[str], nlp) -> List[str]:
    if nlp is None:
        return simple_tokenize_to_lemmas(texts)
    processed = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        toks = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        processed.append(" ".join(toks))
    return processed

def syntactic_features_batch(texts: List[str], nlp) -> List[str]:
    if nlp is None:
        return simple_pos_features(texts)
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        pos = " ".join([token.pos_ for token in doc])
        processed.append(pos)
    return processed

def semantic_features_batch(texts: List[str]) -> pd.DataFrame:
    """
    Lazy import TextBlob only when semantic features are required.
    TextBlob can be slow to import; we only import inside this function.
    """
    try:
        TextBlob = importlib.import_module("textblob").TextBlob
    except Exception:
        # fast fallback: polarity & subjectivity via naive heuristics
        out = []
        for t in texts:
            pol = float(t.count("good") - t.count("bad")) / max(1, len(t.split()))
            subj = float(min(1.0, len(set(t.split())) / 50.0))
            out.append([pol, subj])
        return pd.DataFrame(out, columns=["polarity", "subjectivity"])

    out = []
    for t in texts:
        b = TextBlob(t)
        out.append([b.sentiment.polarity, b.sentiment.subjectivity])
    return pd.DataFrame(out, columns=["polarity", "subjectivity"])

def discourse_features_batch(texts: List[str], nlp) -> List[str]:
    if nlp is None:
        # fallback: sentence count via splitting on punctuation
        processed = []
        for t in texts:
            sents = re.split(r'[.!?]+', t)
            sents = [s.strip() for s in sents if s.strip()]
            first_words = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
            processed.append(f"{len(sents)} {first_words}")
        return processed

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
    """
    Return (feature_matrix, vectorizer_or_None). Feature matrix may be sparse.
    Heavy vectorizer imports (CountVectorizer/TfidfVectorizer) are lazy-imported here.
    """
    X_texts = X_series.astype(str).tolist()
    if phase == "Lexical & Morphological":
        X_proc = lexical_features_batch(X_texts, nlp)
        vecmod = importlib.import_module("sklearn.feature_extraction.text")
        CountVectorizer = getattr(vecmod, "CountVectorizer")
        vect = CountVectorizer(binary=True, ngram_range=(1,2), min_df=3, max_features=2000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Syntactic":
        X_proc = syntactic_features_batch(X_texts, nlp)
        vecmod = importlib.import_module("sklearn.feature_extraction.text")
        TfidfVectorizer = getattr(vecmod, "TfidfVectorizer")
        vect = TfidfVectorizer(max_features=2000, min_df=3)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Semantic":
        df = semantic_features_batch(X_texts)
        return df.values, None

    if phase == "Discourse":
        X_proc = discourse_features_batch(X_texts, nlp)
        vecmod = importlib.import_module("sklearn.feature_extraction.text")
        CountVectorizer = getattr(vecmod, "CountVectorizer")
        vect = CountVectorizer(ngram_range=(1,2), max_features=2000, min_df=2)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Pragmatic":
        df = pragmatic_features_batch(X_texts)
        return df.values, None

    raise ValueError("Unknown phase")

def get_models_dict():
    """
    Lazy-import model classes (so app startup is fast).
    """
    model_mods = importlib.import_module("sklearn")
    # import specific algorithms lazily
    nb_mod = importlib.import_module("sklearn.naive_bayes")
    tree_mod = importlib.import_module("sklearn.tree")
    linear_mod = importlib.import_module("sklearn.linear_model")
    svm_mod = importlib.import_module("sklearn.svm")

    return {
        "Naive Bayes": nb_mod.MultinomialNB(),
        "Decision Tree": tree_mod.DecisionTreeClassifie_
