import pandas as pd
import pytest

from netflix_rec.app import clamp_k, create_app
from netflix_rec.data import read_text, split_list
from netflix_rec.recommender import NetflixRecommender, build_tags, tag


def catalogue() -> pd.DataFrame:
    rows = [
        ("s1", "Movie", "Bank Job", "Ann Lee", "Tom Ray, Kim Bo", "United States", "Thrillers", "A crew of thieves plans to rob a bank vault."),
        ("s2", "Movie", "Vault Night", "Ann Lee", "Tom Ray", "United States", "Thrillers", "Thieves try to rob a bank vault at night."),
        ("s3", "TV Show", "Kitchen Life", None, "Sam Oak", "France", "Reality TV", "Young chefs compete to cook the best meal."),
        ("s4", "Movie", "Space Run", "Bo Chu", "Kim Bo", "Japan", "Sci-Fi & Fantasy", "An astronaut escapes a failing space station."),
    ]
    df = pd.DataFrame(rows, columns=["show_id", "type", "title", "director", "cast", "country", "listed_in", "description"])
    df["release_year"] = pd.array([2001, 2003, 2019, 2020], dtype="Int64")
    df["rating"], df["duration"] = "PG-13", "100 min"
    df["genres"] = df["listed_in"].map(split_list)
    df["cast_list"] = df["cast"].map(split_list)
    df["director_list"] = df["director"].map(split_list)
    df["country_list"] = df["country"].map(split_list)
    return df


@pytest.fixture(scope="module")
def rec():
    # min_df=2 in the real model; tiny catalogues need every word.
    r = NetflixRecommender.__new__(NetflixRecommender)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    r.df, r.w = catalogue(), 0.5
    r.desc_vec = TfidfVectorizer(stop_words="english")
    r.desc = normalize(r.desc_vec.fit_transform(r.df["description"]))
    r.tag_vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    r.tags = normalize(r.tag_vec.fit_transform(r.df.apply(build_tags, axis=1)))
    r._by_lower = {t.lower(): i for i, t in enumerate(r.df["title"])}
    return r


def test_tag_and_split_helpers():
    assert tag("cast", "Tom Hanks") == "cast_tom_hanks"
    assert split_list("A, B ,") == ["A", "B"]
    assert split_list(None) == []


def test_read_text_repairs_cp1252_bytes(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes("Zoë,".encode("utf-8") + b"J\xf8rgensen")
    assert read_text(p) == "Zoë,Jørgensen"


def test_recommend_finds_the_twin_and_explains_it(rec):
    matched, res = rec.recommend("bank job", k=2)
    assert matched == "Bank Job"
    assert res[0].title == "Vault Night"
    assert "dir. Ann Lee" in res[0].shared and "Tom Ray" in res[0].shared


def test_type_filter_and_unknown_title(rec):
    _, res = rec.recommend("Bank Job", k=5, kind="TV Show")
    assert [r.title for r in res] == ["Kitchen Life"]
    with pytest.raises(KeyError):
        rec.recommend("zzz unknown")


def test_free_text(rec):
    assert rec.recommend_from_text("chefs cook a meal", k=1)[0].title == "Kitchen Life"


def test_invalid_weight():
    with pytest.raises(ValueError):
        NetflixRecommender(catalogue(), description_weight=1.5)


def test_flask_routes(rec):
    client = create_app(rec).test_client()
    assert client.get("/").status_code == 200
    page = client.get("/?title=bank+job")
    assert page.status_code == 200 and b"Vault Night" in page.data
    assert client.get("/?title=zzz").status_code == 404
    assert b"&lt;b&gt;" in client.get("/?describe=<b>bank</b>").data  # user input is escaped
    api = client.get("/api/recommend?title=Bank+Job&k=1").get_json()
    assert api["results"][0]["title"] == "Vault Night"
    assert client.get("/api/recommend").status_code == 400
    assert client.get("/api/search?q=va").get_json()["titles"] == ["Vault Night"]
    assert client.get("/health").get_json()["titles"] == 4


def test_clamp_k():
    assert clamp_k(None) == 10 and clamp_k("x") == 10 and clamp_k("0") == 1 and clamp_k("99") == 30
