import os
from fastapi.testclient import TestClient
from .app.server import app

os.chdir("app")
client = TestClient(app)

"""
This part is optional.

We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time.
It would be nice to write some tests to ensure we don't regress!

  1. `Pytest` is a popular testing framework for Python. If you haven't used it before, take a look at
  https://docs.pytest.org/en/7.1.x/getting-started.html to get started and familiarize yourself with this library.

  2. How do we test FastAPI applications with Pytest? Glad you asked, here's two resources to help you get started:
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/

"""


def test_root():
    """
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_empty():
    """
    Test the "/predict" endpoint, with an empty request body
    """
    response = client.post("/predict", json={})
    assert (response.status_code >= 400) and (response.status_code < 500)


def test_predict_en_lang():
    """
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    """
    sample = {
        "source": "The Washington Post",
        "url": "http://www.washingtonpost.com/wp-dyn/articles/A46063-2005Jan3.html?nav=rss_sports",
        "title": "Ruffin Fills Key Role",
        "description": "With power forward Etan Thomas having missed the entire season, reserve forward Michael Ruffin has done well in taking his place.",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["label"] == "Sports"


def test_predict_es_lang():
    """
    Test the "/predict" endpoint, with an input text in Spanish.
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    sample = {
        "source": "The Washington Post",
        "url": "some/url",
        "title": "Some title",
        "description": "Hola me llamo Charles.",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200


def test_predict_non_ascii():
    """
    Test the "/predict" endpoint, with an input text that has non-ASCII characters.
    Does the tokenizer and classifier handle this case correctly? Does it return an error?
    """
    sample = {
        "source": "The Washington Post",
        "url": "some/url",
        "title": "Some title",
        "description": "ålphan¨merˆc",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
