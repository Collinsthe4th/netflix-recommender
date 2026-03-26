"""Flask web app: search page, JSON API and health check.

Run locally:
    flask --app netflix_rec.app run
Production:
    gunicorn "netflix_rec.app:create_app()"
"""

from __future__ import annotations

from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from .recommender import NetflixRecommender

KINDS = {"Movie", "TV Show"}
MAX_QUERY = 300


@lru_cache(maxsize=1)
def get_recommender() -> NetflixRecommender:
    return NetflixRecommender.from_csv()


def clamp_k(raw: str | None, default: int = 10) -> int:
    try:
        return max(1, min(30, int(raw))) if raw else default
    except ValueError:
        return default


def create_app(recommender: NetflixRecommender | None = None) -> Flask:
    """Application factory. Tests can pass a small recommender."""
    app = Flask(__name__)
    get = (lambda: recommender) if recommender is not None else get_recommender

    def params():
        kind = request.args.get("type", "")
        return (
            request.args.get("title", "").strip()[:MAX_QUERY],
            request.args.get("describe", "").strip()[:MAX_QUERY],
            kind if kind in KINDS else None,
            clamp_k(request.args.get("k")),
        )

    @app.get("/")
    def home():
        rec = get()
        title, describe, kind, k = params()
        ctx = {"title": title, "describe": describe, "kind": kind or "", "k": k, "results": None,
               "matched": None, "error": None, "suggestions": [], "count": len(rec.df)}
        status = 200
        if title:
            try:
                ctx["matched"], ctx["results"] = rec.recommend(title, k=k, kind=kind)
            except KeyError:
                ctx["error"] = f"No Netflix title matches “{title}”."
                ctx["suggestions"] = rec.search(title[:5])
                status = 404
        elif describe:
            ctx["results"] = rec.recommend_from_text(describe, k=k, kind=kind)
        return render_template("index.html", **ctx), status

    @app.get("/api/recommend")
    def api_recommend():
        rec = get()
        title, describe, kind, k = params()
        if title:
            try:
                matched, results = rec.recommend(title, k=k, kind=kind)
            except KeyError:
                return jsonify(error=f"'{title}' is not in the catalogue"), 404
            return jsonify(query=matched, results=[r.to_dict() for r in results])
        if describe:
            return jsonify(query=describe, results=[r.to_dict() for r in rec.recommend_from_text(describe, k=k, kind=kind)])
        return jsonify(error="Provide a 'title' or 'describe' parameter"), 400

    @app.get("/api/search")
    def api_search():
        q = request.args.get("q", "")[:MAX_QUERY]
        return jsonify(q=q, titles=get().search(q))

    @app.get("/health")
    def health():
        return jsonify(status="ok", titles=len(get().df))

    return app
