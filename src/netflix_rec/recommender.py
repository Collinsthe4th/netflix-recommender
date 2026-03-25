"""Content-based recommender over 8,800 Netflix titles.

Two text representations are combined:

* **Description**: TF-IDF of the plot description (words and two-word phrases).
* **Metadata tags**: genres, the first 5 cast members, directors and countries,
  each turned into a single token (for example ``cast_tom_hanks``) so that
  "Tom Hanks" never matches "Tom Cruise". These tags are weighted with TF-IDF
  too, so rare tags (a niche director) count more than common ones ("Dramas").

The similarity is ``w * description + (1 - w) * tags`` with ``w = 0.25`` by default. With 8,800 titles a full
similarity matrix would need about 600 MB, so scores are computed for one query
at a time against the sparse matrices (a few milliseconds).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import load

CAST_LIMIT = 5


def tag(prefix: str, value: str) -> str:
    return prefix + "_" + re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_tags(row: pd.Series, use_directors: bool = True) -> str:
    tags = [tag("genre", g) for g in row["genres"]]
    tags += [tag("cast", c) for c in row["cast_list"][:CAST_LIMIT]]
    if use_directors:
        tags += [tag("dir", d) for d in row["director_list"]]
    tags += [tag("country", c) for c in row["country_list"]]
    tags.append(tag("type", row["type"]))
    return " ".join(tags)


@dataclass(frozen=True)
class Recommendation:
    show_id: str
    title: str
    type: str
    release_year: int | None
    rating: str
    duration: str
    genres: list[str]
    similarity: float
    shared: list[str]
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


class NetflixRecommender:
    """Recommend titles similar to a title or a free-text description.

    Args:
        catalogue: output of :func:`netflix_rec.data.load`.
        description_weight: share of the similarity from the plot description.
            0.25 scored best on both offline checks in scripts/evaluate.py.
        use_directors: include director tags (turned off in the evaluation that
            tries to find other films by the same director).
    """

    def __init__(self, catalogue: pd.DataFrame, description_weight: float = 0.25, use_directors: bool = True) -> None:
        if not 0 <= description_weight <= 1:
            raise ValueError("description_weight must be between 0 and 1")
        self.df = catalogue.reset_index(drop=True)
        self.w = description_weight
        self.desc_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.desc = normalize(self.desc_vec.fit_transform(self.df["description"]))
        tags = self.df.apply(build_tags, axis=1, use_directors=use_directors)
        self.tag_vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
        self.tags = normalize(self.tag_vec.fit_transform(tags))
        self._by_lower: dict[str, int] = {}
        for i, t in enumerate(self.df["title"]):
            self._by_lower.setdefault(t.lower(), i)

    @classmethod
    def from_csv(cls, **kwargs) -> "NetflixRecommender":
        return cls(load(), **kwargs)

    # ------------------------------------------------------------------ lookup
    def find_title(self, query: str) -> int | None:
        q = query.strip().lower()
        if not q:
            return None
        if q in self._by_lower:
            return self._by_lower[q]
        starts = [t for t in self._by_lower if t.startswith(q)]
        if starts:
            return self._by_lower[min(starts, key=len)]
        close = difflib.get_close_matches(q, list(self._by_lower), n=1, cutoff=0.75)
        return self._by_lower[close[0]] if close else None

    def search(self, query: str, limit: int = 8) -> list[str]:
        q = query.strip().lower()
        if len(q) < 2:
            return []
        titles = self.df["title"]
        starts = titles[titles.str.lower().str.startswith(q)].tolist()
        contains = titles[titles.str.lower().str.contains(q, regex=False)].tolist()
        out = list(dict.fromkeys(starts + contains))
        return out[:limit]

    # ----------------------------------------------------------------- scoring
    def scores_for_index(self, idx: int) -> np.ndarray:
        d = (self.desc @ self.desc[idx].T).toarray().ravel()
        t = (self.tags @ self.tags[idx].T).toarray().ravel()
        return self.w * d + (1 - self.w) * t

    def _shared(self, i: int, j: int) -> list[str]:
        a, b = self.df.iloc[i], self.df.iloc[j]
        shared = [g for g in a["genres"] if g in b["genres"]]
        shared += [c for c in a["cast_list"][:CAST_LIMIT] if c in b["cast_list"][:CAST_LIMIT]]
        shared += [f"dir. {d}" for d in a["director_list"] if d in b["director_list"]]
        return shared

    def _results(self, scores: np.ndarray, k: int, exclude: int | None, kind: str | None, source: int | None) -> list[Recommendation]:
        order = np.argsort(-scores)
        out: list[Recommendation] = []
        for j in order:
            if j == exclude:
                continue
            row = self.df.iloc[j]
            if kind and row["type"] != kind:
                continue
            out.append(
                Recommendation(
                    show_id=row["show_id"],
                    title=row["title"],
                    type=row["type"],
                    release_year=None if pd.isna(row["release_year"]) else int(row["release_year"]),
                    rating=row["rating"],
                    duration=row["duration"],
                    genres=row["genres"],
                    similarity=round(float(scores[j]), 4),
                    shared=self._shared(source, j) if source is not None else [],
                    description=row["description"],
                )
            )
            if len(out) == k:
                break
        return out

    def recommend(self, title: str, k: int = 10, kind: str | None = None) -> tuple[str, list[Recommendation]]:
        """Return the matched title and its ``k`` most similar titles.

        Args:
            kind: ``"Movie"`` or ``"TV Show"`` to restrict the results.

        Raises:
            KeyError: if no title matches.
        """
        idx = self.find_title(title)
        if idx is None:
            raise KeyError(f"'{title}' is not in the catalogue")
        return self.df["title"].iloc[idx], self._results(self.scores_for_index(idx), k, idx, kind, idx)

    def recommend_from_text(self, text: str, k: int = 10, kind: str | None = None) -> list[Recommendation]:
        q = normalize(self.desc_vec.transform([text]))
        scores = (self.desc @ q.T).toarray().ravel()
        return self._results(scores, k, None, kind, None)
