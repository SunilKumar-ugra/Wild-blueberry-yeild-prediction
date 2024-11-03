import pytest
import numpy as np
from flask import Flask
from app import app  # Importing the Flask app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"index.html" in response.data  # Assuming "index.html" renders correctly

def test_training_route(client):
    """Test the training route."""
    response = client.get('/train')
    assert response.status_code == 200
    assert b"Training Successful!" in response.data

def test_prediction_route(client, monkeypatch):
    """Test the prediction route with mock input data."""

    class MockPredictionPipeline:
        def predict(self, data):
            return 42  # Mock prediction output for testing

    # Replace the PredictionPipeline with the mock class
    monkeypatch.setattr('app.PredictionPipeline', MockPredictionPipeline)

    data = {
        'clonesize': '1.0', 'honeybee': '1.0', 'bumbles': '1.0', 'andrena': '1.0', 'osmia': '1.0',
        'MaxOfUpperTRange': '1.0', 'MinOfUpperTRange': '1.0', 'AverageOfUpperTRange': '1.0',
        'MaxOfLowerTRange': '1.0', 'MinOfLowerTRange': '1.0', 'AverageOfLowerTRange': '1.0',
        'RainingDays': '1.0', 'AverageRainingDays': '1.0', 'fruitset': '1.0', 'fruitmass': '1.0', 'seeds': '1.0'
    }
    response = client.post('/predict', data=data)
    
    assert response.status_code == 200
    assert b"42" in response.data  # Check if the prediction is correct

def test_predict_route_get_request(client):
    """Test the predict route with a GET request."""
    response = client.get('/predict')
    assert response.status_code == 200
    assert b"index.html" in response.data  # Ensure the form loads on GET request
