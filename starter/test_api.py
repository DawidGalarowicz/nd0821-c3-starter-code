from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello World! Welcome to my prediction app!"


def test_post_above_50k():
    data_send = {
        "": 0,
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital_gain": 217400,
        "capital_loss": 41,
        "hours_per_week": 50,
        "native_country": "United-States",
    }

    r = client.post("/predict", json=data_send)
    assert r.status_code == 200
    assert r.json() == "Person will earn at least $50,000"


def test_post_below_50k():
    data_send = {
        "": 0,
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 41,
        "hours_per_week": 50,
        "native_country": "United-States",
    }

    r = client.post("/predict", json=data_send)
    assert r.status_code == 200
    assert r.json() == "Person will earn below $50,000"
