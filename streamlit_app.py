"""WatchNext in Streamlit: live demo of the Netflix recommender.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_rec import NetflixRecommender  # noqa: E402

st.set_page_config(page_title="WatchNext", page_icon="🍿", layout="wide")


@st.cache_resource
def get_recommender() -> NetflixRecommender:
    return NetflixRecommender.from_csv()


rec = get_recommender()
titles = sorted(rec.df["title"].unique().tolist(), key=str.lower)

st.title("🍿 WatchNext: what to watch on Netflix next")
st.caption(f"Content-based recommendations across {len(rec.df):,} movies and series. Similarity = 25% plot description + 75% tags (genres, top cast, directors, countries).")

mode = st.radio("Search by", ["A title I watched", "A description"], horizontal=True)
c1, c2, c3 = st.columns([3, 1.2, 1])
with c1:
    if mode == "A title I watched":
        default = titles.index("The Irishman") if "The Irishman" in titles else 0
        query = st.selectbox("Title", titles, index=default)
    else:
        query = st.text_input("Describe it", "a heist crew plans to rob a bank")
with c2:
    kind_label = st.selectbox("Show", ["Movies and series", "Movies only", "Series only"])
with c3:
    k = st.slider("Results", 3, 24, 9)

kind = {"Movies and series": None, "Movies only": "Movie", "Series only": "TV Show"}[kind_label]
if mode == "A title I watched":
    matched, results = rec.recommend(query, k=k, kind=kind)
    st.subheader(f"Because you watched {matched}")
else:
    results = rec.recommend_from_text(query, k=k, kind=kind) if query.strip() else []
    st.subheader(f"Matches for “{query}”" if query.strip() else "Type a description above")

cols = st.columns(3)
for i, r in enumerate(results):
    with cols[i % 3].container(border=True):
        st.caption(f"{'SERIES' if r.type == 'TV Show' else 'FILM'} · {r.release_year or ''} · {r.rating} · {r.duration}")
        st.markdown(f"**{r.title}**")
        st.write(r.description)
        if r.shared:
            st.caption("In common: " + ", ".join(r.shared[:5]))
        st.progress(min(r.similarity * 2, 1.0))

with st.expander("How the model was evaluated"):
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    st.write(
        "With the director tags hidden, a good recommender should still find a director's other work from the plot, "
        "cast and genres. A description weight of 0.25 scored best on that check and on genre coherence."
    )
    st.dataframe(pd.DataFrame(m["sweep"]), hide_index=True, use_container_width=True)
    st.image(str(ROOT / "results" / "figures" / "weight_sweep.png"))

st.markdown("[Source code and README](https://github.com/Collinsthe4th/netflix-recommender)")
