import streamlit as st
import requests
from urllib.parse import quote_plus

# Prefer secret; fallback to placeholder default (you can remove the default if you want)
DEFAULT_API_KEY = ""  # keep empty or set placeholder
API_KEY = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", DEFAULT_API_KEY) if hasattr(st, "secrets") else DEFAULT_API_KEY

st.set_page_config(page_title="Global FactCheck Network", layout="wide", initial_sidebar_state="collapsed")

# Dark theme CSS & news-style layout
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1014 0%, #0e1419 100%);
        color: #E6EEF3;
    }
    .header {
        background: linear-gradient(90deg, rgba(7,54,66,0.9), rgba(139,0,92,0.85));
        padding: 28px;
        border-radius: 10px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        margin-bottom: 18px;
    }
    .header h1 { margin: 0; font-size: 2.4rem; color: #fff; }
    .header p { margin: 4px 0 0 0; color: #d8eaf2; opacity: 0.9; }
    .search-container { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 8px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.04); }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 14px; border-radius: 10px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.03); }
    .card .meta { color: #9fb1c6; font-size: 0.9rem; margin-bottom: 6px; }
    .badge { display:inline-block; padding:6px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; margin-left:8px; }
    .badge.TRUE { background: rgba(46,204,113,0.15); color: #2ee071; border: 1px solid rgba(46,204,113,0.25); }
    .badge.FALSE { background: rgba(231,76,60,0.12); color: #ff6b6b; border: 1px solid rgba(231,76,60,0.18); }
    .badge.NA { background: rgba(255,255,255,0.03); color: #d1dbe6; border: 1px solid rgba(255,255,255,0.03); }
    a.link-button { display:inline-block; padding:8px 12px; border-radius:8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); color: #dceefc; text-decoration: none; margin-top:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <h1>Global FactCheck Network</h1>
        <p>Professional verification search — news portal style. Enter a claim or headline and get verified fact-checks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search container
st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input("", placeholder='Type a claim or headline to verify (e.g. "X country banned Y")', key="fact_query")
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    check_btn = st.button("Check", key="check_btn", help="Query Google Fact Check Tools API")
with col3:
    clear_btn = st.button("Clear", key="clear_btn")
st.markdown("</div>", unsafe_allow_html=True)

if clear_btn:
    st.session_state["fact_query"] = ""

if check_btn:
    if not query or not query.strip():
        st.warning("Please enter a claim or headline to search.")
    else:
        q = query.strip()
        if not API_KEY:
            st.error("No API key configured. Add GOOGLE_FACTCHECK_API_KEY to Streamlit secrets.")
        else:
            url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={quote_plus(q)}&key={API_KEY}"
            with st.spinner("Searching verified fact-checks..."):
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.HTTPError as he:
                    try:
                        err = he.response.json().get("error", {}).get("message", str(he))
                    except Exception:
                        err = str(he)
                    st.error(f"API error: {err}")
                    data = {}
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    data = {}

            if "claims" in data and data["claims"]:
                st.success(f"Found {len(data['claims'])} related claim(s).")
                for claim in data["claims"]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    text = claim.get("text", "—")
                    st.markdown(f"### {text}", unsafe_allow_html=True)
                    for review in claim.get("claimReview", []):
                        pub = review.get("publisher", {}).get("name", "Unknown")
                        rating = review.get("textualRating", None)
                        rtitle = review.get("title", review.get("url", "")) or ""
                        rurl = review.get("url", "#")
                        badge_class = "NA"
                        if rating:
                            low = rating.upper()
                            if "TRUE" in low:
                                badge_class = "TRUE"
                            elif "FALSE" in low or "PANTS" in low or "FLIP" in low:
                                badge_class = "FALSE"
                            else:
                                badge_class = "NA"
                        st.markdown(f'<div class="meta"><strong>{pub}</strong> <span class="badge {badge_class}">{rating or "No Rating"}</span></div>', unsafe_allow_html=True)
                        if rtitle:
                            st.markdown(f"**{rtitle}**", unsafe_allow_html=True)
                        st.markdown(f'<a class="link-button" target="_blank" href="{rurl}">Read full article</a>', unsafe_allow_html=True)
                        st.markdown("---", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No fact-checks found for this query.")
