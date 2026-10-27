import unittest
from unittest.mock import patch

from app import app


class TestIrisPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    @patch("app.joblib.load")
    def test_prediction_endpoint(self, mock_load):
        mock_model = mock_load.return_value
        mock_model.predict.return_value = ["Iris-setosa"]

        response = self.client.post(
            "/predict",
            json={
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "Iris-setosa"
        )

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"SepalLengthCm": 5.1}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
