"""
Tests for cache functionality in the heart disease prediction API.
"""

import time
import unittest
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.models.predict_model import PredictionCache


class TestPredictionCache(unittest.TestCase):
    """Test the PredictionCache class directly."""

    def setUp(self):
        """Set up test instance."""
        self.cache = PredictionCache(max_size=10, ttl=60)

    def test_cache_put_get(self):
        """Test basic put and get operations."""
        test_data = {"age": 61, "sex": 1}
        test_model = "test_model"
        test_result = {"prediction": 1, "probability": 0.8}

        # Cache should be empty initially
        assert self.cache.get_stats()["entries"] == 0
        assert self.cache.get_stats()["hits"] == 0
        assert self.cache.get_stats()["misses"] == 0

        # Put item in cache
        self.cache.put(test_data, test_model, test_result)
        assert self.cache.get_stats()["entries"] == 1

        # Get item from cache
        cached_result = self.cache.get(test_data, test_model)
        assert cached_result == test_result
        assert self.cache.get_stats()["hits"] == 1

        # Try to get non-existent item
        non_existent = self.cache.get({"different": "data"}, test_model)
        assert non_existent is None
        assert self.cache.get_stats()["misses"] == 1

    def test_cache_lru_behavior(self):
        """Test LRU (Least Recently Used) eviction policy."""
        # Fill cache to capacity
        for i in range(10):
            self.cache.put({"id": i}, "test_model", {"result": i})

        # Cache should be full
        assert self.cache.get_stats()["entries"] == 10
        assert self.cache.get_stats()["evictions"] == 0

        # Access item to make it most recently used
        self.cache.get({"id": 0}, "test_model")

        # Add one more item to trigger eviction (should evict second oldest, id=1)
        self.cache.put({"id": 10}, "test_model", {"result": 10})

        # Check that the cache size stays at max_size
        assert self.cache.get_stats()["entries"] == 10
        assert self.cache.get_stats()["evictions"] == 1

        # Check that id=0 is still in cache (was accessed recently)
        assert self.cache.get({"id": 0}, "test_model") is not None

        # Check that id=1 was evicted
        assert self.cache.get({"id": 1}, "test_model") is None

    def test_cache_ttl(self):
        """Test time-to-live (TTL) expiration."""
        test_data = {"age": 61, "sex": 1}
        test_model = "test_model"
        test_result = {"prediction": 1, "probability": 0.8}

        # Put item in cache
        self.cache.put(test_data, test_model, test_result)

        # Mock time to simulate TTL expiration
        with mock.patch("time.time", return_value=time.time() + 61):  # 61 > TTL of 60
            # Get should return None as item has expired
            assert self.cache.get(test_data, test_model) is None
            # Stats should reflect the miss
            assert self.cache.get_stats()["entries"] == 0  # Item should be removed
            assert self.cache.get_stats()["misses"] == 1

    def test_cache_clear(self):
        """Test cache clear functionality."""
        # Put some items in cache
        for i in range(5):
            self.cache.put({"id": i}, "test_model", {"result": i})

        # Cache should have items
        assert self.cache.get_stats()["entries"] == 5

        # Clear cache
        self.cache.clear()

        # Cache should be empty
        assert self.cache.get_stats()["entries"] == 0
        assert self.cache.get_stats()["hits"] == 0
        assert self.cache.get_stats()["misses"] == 0
        assert self.cache.get_stats()["evictions"] == 0


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_cache_stats_endpoint(client):
    """Test the /cache/stats endpoint."""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    # Check that response contains expected fields
    assert "enabled" in data
    assert "max_size" in data
    assert "ttl_seconds" in data
    assert "entries" in data
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data
    assert "evictions" in data
    assert "created_at" in data


def test_cache_config_endpoint(client):
    """Test the /cache/config endpoint."""
    # Configure cache
    response = client.post("/cache/config", json={"enabled": True, "max_size": 2000, "ttl": 7200})
    assert response.status_code == 200
    data = response.json()
    # Check that configuration was applied
    assert data["enabled"] is True
    assert data["max_size"] == 2000
    assert data["ttl_seconds"] == 7200


def test_cache_clear_endpoint(client):
    """Test the /cache/clear endpoint."""
    response = client.post("/cache/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "message" in data


def test_prediction_with_cache(client, sample_patient_data):
    """Test that predictions are cached."""
    # Make initial prediction
    initial_response = client.post("/predict", json=sample_patient_data)
    assert initial_response.status_code == 200

    # Get cache stats to verify the prediction was cached
    stats_response = client.get("/cache/stats")
    stats = stats_response.json()
    initial_hits = stats["hits"]
    initial_entries = stats["entries"]

    # Make the same prediction again
    cached_response = client.post("/predict", json=sample_patient_data)
    assert cached_response.status_code == 200

    # Results should be identical
    assert cached_response.json() == initial_response.json()

    # Get updated cache stats
    updated_stats_response = client.get("/cache/stats")
    updated_stats = updated_stats_response.json()

    # Verify hits increased
    assert updated_stats["hits"] > initial_hits
    # Entries should remain the same since we reused an existing entry
    assert updated_stats["entries"] == initial_entries


if __name__ == "__main__":
    unittest.main()
