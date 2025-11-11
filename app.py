# app.py
import time
import requests
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------- PAGE & THEME ----------------------------------
st.set_page_config(page_title="AI Multilingual Book Recommender", layout="wide")
st.markdown("""
<style>
/* Clean dark cards */
.card {
  padding: 16px;
  border-radius: 14px;
  background: #111418;
  color: #e8edf3;
  border: 1px solid #1f2937;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.title { font-size: 1.1rem; font-weight: 700; margin: 0 0 6px 0; }
.meta  { font-size: 0.92rem; opacity: 0.85; margin: 0 0 8px 0; }
.summary { font-size: 0.95rem; line-height: 1.45; opacity: 0.95; }
hr { border: none; border-top: 1px solid #273141; margin: 14px 0; }
.badge { display:inline-block; padding:4px 8px; border-radius:999px; background:#1b2533; margin-right:6px; margin-top:6px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------- MODEL ----------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Multilingual: Hindi, English, +90 languages
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# ---------------------------------- API LAYERS ----------------------------------
def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

@st.cache_data(show_spinner=True, ttl=1800)
def fetch_google_books(query: str, max_results: int = 40, lang_restrict: str = "hi,en") -> List[Dict[str, Any]]:
    """Fetch raw books from Google Books for a query."""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query.strip() or "books",
        "maxResults": min(max_results, 40),
        "printType": "books",
        "langRestrict": lang_restrict
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
    except Exception:
        return []

    items = data.get("items", []) or []
    out = []
    for it in items:
        info = it.get("volumeInfo", {}) or {}
        # Pick first ISBN if exists (for better OpenLibrary resolution)
        isbn = None
        for ident in info.get("industryIdentifiers", []) or []:
            t = (ident.get("type") or "").upper()
            if t in ("ISBN_13", "ISBN_10"):
                isbn = ident.get("identifier")
                break
        out.append({
            "title": info.get("title") or "Untitled",
            "authors": info.get("authors") or ["Unknown"],
            "summary": info.get("description") or "",
            "categories": info.get("categories") or [],
            "language": info.get("language") or "",
            "thumbnail": _safe_get(info, ["imageLinks", "thumbnail"], "")
                        or _safe_get(info, ["imageLinks", "smallThumbnail"], ""),
            "isbn": isbn
        })
    return out

def _ol_search(title: str, author: str | None) -> dict:
    """OpenLibrary search; return first doc or {}."""
    try:
        params = {"title": title}
        if author:
            params["author"] = author
        r = requests.get("https://openlibrary.org/search.json", params=params, timeout=10)
        js = r.json()
        docs = js.get("docs") or []
        return docs[0] if docs else {}
    except Exception:
        return {}

def _ol_ratings(work_key: str) -> Tuple[float | None, int | None]:
    """Fetch ratings for a work (/works/OL.../ratings.json)."""
    try:
        r = requests.get(f"https://openlibrary.org{work_key}/ratings.json", timeout=10)
        js = r.json()
        avg = js.get("summary", {}).get("average", None)
        count = js.get("summary", {}).get("count", None)
        return (float(avg) if avg is not None else None, int(count) if count is not None else None)
    except Exception:
        return (None, None)

def _ol_subjects(work_key: str) -> List[str]:
    """Fetch subjects for a work (/works/OL....json)."""
    try:
        r = requests.get(f"https://openlibrary.org{work_key}.json", timeout=10)
        js = r.json()
        subs = js.get("subjects") or []
        # Normalize to list of strings
        return [s if isinstance(s, str) else "" for s in subs][:20]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def enrich_with_openlibrary(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich Google books with OpenLibrary work metadata: ratings & subjects."""
    enriched = []
    for b in books:
        title = b.get("title", "")
        author0 = (b.get("authors") or ["Unknown"])[0]
        ol_doc = _ol_search(title, author0)
        work_key = ol_doc.get("key", "")
        ratings_avg, ratings_count = (None, None)
        subjects = []
        if work_key.startswith("/works/"):
            ratings_avg, ratings_count = _ol_ratings(work_key)
            subjects = _ol_subjects(work_key)

        b_en = dict(b)
        b_en["openlibrary"] = {
            "work_key": work_key,
            "ratings_average": ratings_avg,    # Goodreads-style rating via OL
            "ratings_count": ratings_count,
            "subjects": subjects,
            "first_publish_year": ol_doc.get("first_publish_year", None)
        }
        enriched.append(b_en)
    return enriched

# ---------------------------------- RECOMMENDER CORE ----------------------------------
def build_embeddings(texts: List[str]) -> np.ndarray:
    # If summary is missing, fall back to title; keep things robust.
    safe_texts = [(t if (isinstance(t, str) and t.strip()) else " ") for t in texts]
    return model.encode(safe_texts, show_progress_bar=False)

def compute_semantic_scores(query: str, doc_emb: np.ndarray) -> np.ndarray:
    q_emb = model.encode([query], show_progress_bar=False)
    return cosine_similarity(q_emb, doc_emb)[0]

def hybrid_rank(
        books: List[Dict[str, Any]],
        query: str,
        w_semantic: float = 1.0,
        w_rating: float = 0.15,
        w_genre: float = 0.05,
        pref_author: str | None = None,
        boost_author: float = 1.08,
        genre_filter: List[str] | None = None
    ) -> List[Tuple[int, float]]:
    """
    Rank items by semantic relevance + gentle boosts from ratings & metadata.
    """
    # Embeddings on the book descriptions
    summaries = [b.get("summary") or b.get("title", "") for b in books]
    doc_emb = build_embeddings(summaries)
    sem = compute_semantic_scores(query, doc_emb)
    # Normalize semantic to 0..1
    sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)

    # Compute hybrid score
    scores = sem_norm * w_semantic

    for i, b in enumerate(books):
        ol = b.get("openlibrary") or {}
        rating = ol.get("ratings_average", None)  # OpenLibrary exposes Goodreads-style agg
        # Rating gentle additive boost (assume 1..5 scale)
        if isinstance(rating, (float, int)):
            scores[i] += (float(rating) / 5.0) * w_rating

        # Genre “soft” boost if any category exists
        cats = b.get("categories") or []
        if cats:
            scores[i] += w_genre

        # Author preference multiplicative bonus
        if pref_author and pref_author != "None":
            if pref_author.lower() in ", ".join(b.get("authors") or []).lower():
                scores[i] *= boost_author

        # Apply genre filter (HARD filter) if provided
        if genre_filter and "All" not in genre_filter:
            if not any(c in genre_filter for c in cats):
                # Penalize heavily (don't zero out to keep sort stable)
                scores[i] *= 0.0001

    # Return ranked indices and scores
    order = np.argsort(scores)[::-1]
    return [(int(idx), float(scores[idx])) for idx in order]

# ---------------------------------- UI ----------------------------------
st.title("📚 AI Multilingual Book Recommender (Live, API-Powered)")
st.caption("Real-time data from Google Books + OpenLibrary • Hindi + English • Hybrid semantic & ratings scoring")

with st.container():
    qcol, ccol = st.columns([3, 1])
    with qcol:
        user_query = st.text_input(
            "🔎 Search by title / Hindi / English / description:",
            placeholder="उदा.: 'महाभारत जैसी पौराणिक कथा' या 'mythology adventure epic quest'"
        )
    with ccol:
        max_results = st.slider("Fetch size", min_value=10, max_value=40, value=30, step=5)

# Sidebar controls (appear always; populate choices after first fetch)
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Results to show", 5, 20, 8)
w_sem = st.sidebar.slider("Weight: Semantic", 0.1, 1.5, 1.0, 0.05)
w_rat = st.sidebar.slider("Weight: Ratings", 0.0, 0.5, 0.15, 0.01)
w_gen = st.sidebar.slider("Weight: Genre Presence", 0.0, 0.2, 0.05, 0.01)

# For dynamic options, we’ll fill after fetching
if "dynamic_genres" not in st.session_state: st.session_state.dynamic_genres = ["All"]
if "dynamic_authors" not in st.session_state: st.session_state.dynamic_authors = ["None"]

pref_author = st.sidebar.selectbox("Prefer Author", options=st.session_state.dynamic_authors)
genre_filter = st.sidebar.multiselect("Filter Genres", options=st.session_state.dynamic_genres, default=["All"])

# Action
if user_query.strip():
    with st.spinner("🔄 Fetching live books & enriching…"):
        t0 = time.time()
        raw_books = fetch_google_books(user_query, max_results=max_results)
        books = enrich_with_openlibrary(raw_books)
        t1 = time.time()

    if not books:
        st.error("No results from APIs. Try a different query.")
        st.stop()

    # Build dynamic filters from current result-set
    # Genres (flatten unique categories)
    genres_set = set()
    for b in books:
        for c in (b.get("categories") or []):
            if isinstance(c, str) and c.strip():
                genres_set.add(c.strip())
    genres_list = ["All"] + sorted(genres_set)
    st.session_state.dynamic_genres = genres_list

    # Authors (common)
    authors_set = set()
    for b in books:
        for a in (b.get("authors") or []):
            if isinstance(a, str) and a.strip():
                authors_set.add(a.strip())
    authors_list = ["None"] + sorted(authors_set)
    st.session_state.dynamic_authors = authors_list

    # Re-render sidebar with new options on next run; continue ranking now
    with st.spinner("🧠 Ranking with multilingual embeddings…"):
        ranked = hybrid_rank(
            books=books,
            query=user_query,
            w_semantic=w_sem,
            w_rating=w_rat,
            w_genre=w_gen,
            pref_author=pref_author,
            genre_filter=genre_filter,
        )
        t2 = time.time()

    st.subheader(f"🔮 Top Recommendations ({min(top_k, len(ranked))})")
    st.caption(f"Fetched in {(t1 - t0):.2f}s • Ranked in {(t2 - t1):.2f}s • Total {(t2 - t0):.2f}s")

    shown = 0
    for idx, score in ranked:
        if shown >= top_k:
            break

        b = books[idx]
        cats = b.get("categories") or []
        ol = b.get("openlibrary") or {}
        rating = ol.get("ratings_average", None)
        rating_count = ol.get("ratings_count", None)
        subjects = ol.get("subjects") or []

        # HARD filter already applied in ranking; still skip if genre filter excludes
        if genre_filter and "All" not in genre_filter:
            if not any(c in genre_filter for c in cats):
                continue

        colA, colB = st.columns([1, 3], vertical_alignment="top")
        with colA:
            thumb = b.get("thumbnail") or "https://via.placeholder.com/128x180?text=No+Cover"
            st.image(thumb, use_container_width=True)
        with colB:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='title'>{b.get('title','Untitled')}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='meta'>"
                f"Authors: {', '.join(b.get('authors') or ['Unknown'])}"
                f" • Language: {b.get('language','') or '—'}"
                f" • Similarity: {score:.3f}"
                f"</div>", unsafe_allow_html=True
            )

            if rating is not None:
                stars = "⭐" * int(round(float(rating)))
                st.markdown(
                    f"<div class='meta'>Goodreads-style Rating (via OpenLibrary): "
                    f"<b>{float(rating):.2f}</b> {stars} "
                    f"{'('+str(rating_count)+' ratings)' if rating_count else ''}"
                    f"</div>", unsafe_allow_html=True
                )

            # Categories as badges
            if cats:
                st.markdown(
                    " ".join([f"<span class='badge'>{c}</span>" for c in cats[:6]]),
                    unsafe_allow_html=True
                )

            # Subjects (a few)
            if subjects:
                st.markdown("<div class='meta'>Subjects: " + ", ".join(subjects[:8]) + "</div>", unsafe_allow_html=True)

            # Summary
            summary = (b.get("summary") or "").strip()
            if summary:
                st.markdown(f"<div class='summary'>{summary[:600]}{'...' if len(summary)>600 else ''}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        shown += 1

else:
    st.info("Type a query above (Hindi/English/mixed). Example: **महाभारत जैसी पौराणिक कथा** or **mythology adventure epic**")
