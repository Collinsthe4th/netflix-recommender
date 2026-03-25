"""Offline evaluation without user ratings.

1. **Same-director retrieval.** For films whose director made at least one other
   title in the catalogue, the director tag is removed from the model. A good
   recommender should still rank that director's other work highly, from the
   plot, cast and genre alone. Reported as hit rate @10 and MRR.
2. **Genre coherence.** Mean Jaccard overlap of Netflix genres between a title
   and its top 10, against random picks.

The description weight is swept from 0 (tags only) to 1 (description only).
A fixed random sample of 1,000 query titles keeps the run under a minute.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netflix_rec.data import load  # noqa: E402
from netflix_rec.recommender import NetflixRecommender  # noqa: E402

RESULTS = ROOT / "results"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SAMPLE = 1000


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def main() -> None:
    df = load()
    rng = np.random.default_rng(42)
    n = len(df)
    directors = df["director_list"]
    by_director: dict[str, set[int]] = {}
    for i, ds in enumerate(directors):
        for d in ds:
            by_director.setdefault(d, set()).add(i)

    def siblings(i: int) -> set[int]:
        out: set[int] = set()
        for d in directors.iloc[i]:
            out |= by_director[d]
        out.discard(i)
        return out

    eligible = [i for i in range(n) if siblings(i)]
    dir_queries = rng.choice(eligible, size=min(SAMPLE, len(eligible)), replace=False)
    genre_queries = rng.choice(n, size=SAMPLE, replace=False)
    genres = df["genres"].tolist()

    random_coh = float(np.mean([np.mean([jaccard(genres[i], genres[j]) for j in rng.choice(n, 10)]) for i in genre_queries]))

    sweep = []
    for w in WEIGHTS:
        rec = NetflixRecommender(df, description_weight=w, use_directors=False)
        hits, rr = 0, []
        for i in dir_queries:
            s = rec.scores_for_index(int(i))
            s[i] = -np.inf
            order = np.argsort(-s)
            sib = siblings(int(i))
            rank = next(r for r, j in enumerate(order, 1) if j in sib)
            hits += rank <= 10
            rr.append(1 / rank)
        coh = []
        for i in genre_queries:
            s = rec.scores_for_index(int(i))
            s[i] = -np.inf
            top = np.argpartition(-s, 10)[:10]
            coh.append(np.mean([jaccard(genres[i], genres[j]) for j in top]))
        row = {
            "description_weight": w,
            "director_hit_rate_at_10": round(hits / len(dir_queries), 4),
            "director_mrr": round(float(np.mean(rr)), 4),
            "genre_coherence_top10": round(float(np.mean(coh)), 4),
        }
        sweep.append(row)
        print(row)

    out = {
        "titles": n,
        "director_queries": int(len(dir_queries)),
        "titles_with_director_sibling": len(eligible),
        "genre_queries": SAMPLE,
        "random_genre_coherence": round(random_coh, 4),
        "random_director_hit_rate_at_10_approx": round(10 / n, 4),
        "sweep": sweep,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = [r["description_weight"] for r in sweep]
    ax.plot(x, [r["director_hit_rate_at_10"] for r in sweep], marker="o", label="Same-director hit rate @10")
    ax.plot(x, [r["director_mrr"] for r in sweep], marker="o", label="Same-director MRR")
    ax.plot(x, [r["genre_coherence_top10"] for r in sweep], marker="o", label="Genre coherence (top 10)")
    ax.axhline(random_coh, color="grey", ls=":", label="Genre coherence, random picks")
    ax.set_xlabel("Description weight (0 = tags only, 1 = description only)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Offline evaluation (director tags removed)")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "weight_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
