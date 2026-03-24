"""Loading the Netflix titles file.

The Kaggle export has two problems that this module fixes:

* 14 empty trailing columns (",,,,," at the end of every line).
* One row (cast of a Czech production) saved in Windows-1252 inside an
  otherwise UTF-8 file. A strict UTF-8 read fails on it, and a lenient read
  turns the names into "?". Those bytes are decoded as Windows-1252 instead.
"""

from __future__ import annotations

import codecs
import io
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "netflix_titles.csv"

COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
    "description",
]


def _cp1252_fallback(err: UnicodeDecodeError) -> tuple[str, int]:
    bad = err.object[err.start : err.end]
    return bad.decode("cp1252", errors="replace"), err.end


codecs.register_error("cp1252_fallback", _cp1252_fallback)


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="cp1252_fallback")


def split_list(value) -> list[str]:
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    """Return the catalogue with clean columns and list-valued helper columns."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See the README for the data source.")
    df = pd.read_csv(io.StringIO(read_text(path)), dtype=str)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[COLUMNS].copy()
    df = df.dropna(subset=["title", "description"]).drop_duplicates("show_id").reset_index(drop=True)
    for col in ("type", "title", "description", "rating", "duration"):
        df[col] = df[col].fillna("").str.strip()
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["genres"] = df["listed_in"].map(split_list)
    df["cast_list"] = df["cast"].map(split_list)
    df["director_list"] = df["director"].map(split_list)
    df["country_list"] = df["country"].map(split_list)
    return df
