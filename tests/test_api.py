import pytest

from api import create_app


class StubModel:
    def predict(self, features):
        return [features.loc[0, "hour"] + features.loc[0, "day"]]


@pytest.fixture
def client():
    app = create_app(model=StubModel())
    app.config.update(TESTING=True)
    return app.test_client()


def test_predict_returns_energy_for_valid_input(client):
    response = client.post("/predict", json={"hour": 14, "day": 2})

    assert response.status_code == 200
    assert response.get_json() == {"predicted_energy_kwh": 16.0}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "Request body must be a JSON object."),
        ({"hour": -1, "day": 0}, "`hour` must be an integer from 0 to 23."),
        ({"hour": 24, "day": 0}, "`hour` must be an integer from 0 to 23."),
        ({"hour": True, "day": 0}, "`hour` must be an integer from 0 to 23."),
        ({"hour": 1.5, "day": 0}, "`hour` must be an integer from 0 to 23."),
        (
            {"hour": 1, "day": 7},
            "`day` must be an integer from 0 (Monday) to 6 (Sunday).",
        ),
        (
            {"hour": 1, "day": False},
            "`day` must be an integer from 0 (Monday) to 6 (Sunday).",
        ),
    ],
)
def test_predict_rejects_invalid_input(client, payload, message):
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": message}
