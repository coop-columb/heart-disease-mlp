#!/usr/bin/env python
"""
Script to load test the Heart Disease Prediction API.
This script simulates multiple concurrent requests to test API performance.
"""

import concurrent.futures
import statistics
import time
from typing import Dict, List, Tuple

import requests

# Configuration
BASE_URL = "http://localhost:8000"
NUM_CONCURRENT = 10  # Number of concurrent requests
NUM_REQUESTS = 100  # Total number of requests to make
TIMEOUT = 10  # Request timeout in seconds


def get_sample_patient() -> Dict:
    """Get sample patient data for testing."""
    return {
        "age": 61,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 240,
        "fbs": 1,
        "restecg": 1,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 2.4,
        "slope": 2,
        "ca": 1,
        "thal": 3,
    }


def make_prediction_request() -> Tuple[bool, float]:
    """Make a prediction request and return success status and time taken."""
    start_time = time.time()
    success = False

    try:
        response = requests.post(
            f"{BASE_URL}/predict", json=get_sample_patient(), timeout=TIMEOUT
        )
        success = response.status_code == 200
    except Exception as e:
        print(f"Request error: {str(e)}")

    time_taken = time.time() - start_time
    return success, time_taken


def run_batch_prediction_test(batch_size: int = 10) -> Tuple[bool, float]:
    """Test batch prediction endpoint with specified batch size."""
    patients = [get_sample_patient() for _ in range(batch_size)]

    start_time = time.time()
    success = False

    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch", json=patients, timeout=TIMEOUT
        )
        success = response.status_code == 200
    except Exception as e:
        print(f"Batch request error: {str(e)}")

    time_taken = time.time() - start_time
    return success, time_taken


def run_concurrent_tests(
    num_concurrent: int, num_requests: int
) -> List[Tuple[bool, float]]:
    """Run concurrent tests and return results."""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        future_to_request = {
            executor.submit(make_prediction_request): i for i in range(num_requests)
        }

        for future in concurrent.futures.as_completed(future_to_request):
            request_num = future_to_request[future]
            try:
                success, time_taken = future.result()
                results.append((success, time_taken))
                status = "Success" if success else "Failed"
                print(
                    f"Request {request_num+1}/{num_requests}: {status} in {time_taken:.4f}s"
                )
            except Exception as e:
                print(
                    f"Request {request_num+1}/{num_requests} generated an exception: {str(e)}"
                )

    return results


def analyze_results(results: List[Tuple[bool, float]]) -> Dict:
    """Analyze test results and return statistics."""
    if not results:
        return {"error": "No results to analyze"}

    success_count = sum(1 for success, _ in results if success)
    success_rate = success_count / len(results) * 100

    times = [time_taken for _, time_taken in results]

    return {
        "total_requests": len(results),
        "successful_requests": success_count,
        "failed_requests": len(results) - success_count,
        "success_rate": success_rate,
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "p95_time": statistics.quantiles(times, n=20)[18],  # 95th percentile
        "throughput": len(results) / sum(times),
    }


def test_health_endpoint() -> Tuple[bool, float]:
    """Test health endpoint performance."""
    start_time = time.time()
    success = False

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        success = response.status_code == 200
    except Exception as e:
        print(f"Health endpoint error: {str(e)}")

    time_taken = time.time() - start_time
    return success, time_taken


def test_batch_sizes() -> Dict:
    """Test different batch sizes and analyze performance."""
    batch_sizes = [1, 5, 10, 20, 50, 100]
    results = {}

    for size in batch_sizes:
        print(f"\nTesting batch size: {size}")
        success, time_taken = run_batch_prediction_test(size)
        results[size] = {
            "success": success,
            "time": time_taken,
            "time_per_prediction": time_taken / size if size > 0 else 0,
        }
        print(
            f"Batch size {size}: {'Success' if success else 'Failed'} in {time_taken:.4f}s"
        )
        print(f"Time per prediction: {time_taken/size:.4f}s")

    return results


def main():
    """Run the load test."""
    print("Starting Heart Disease Prediction API load test")
    print(
        f"Testing {NUM_REQUESTS} requests with {NUM_CONCURRENT} concurrent connections\n"
    )

    # First check if API is up
    health_success, health_time = test_health_endpoint()
    if not health_success:
        print("ERROR: API health check failed. Is the API running?")
        return

    print(f"API health check: OK ({health_time:.4f}s)\n")

    # Run concurrent tests
    print("Running load test...")
    results = run_concurrent_tests(NUM_CONCURRENT, NUM_REQUESTS)

    # Analyze and display results
    stats = analyze_results(results)

    print("\nLoad Test Results:")
    print("==================")
    print(f"Total Requests:      {stats['total_requests']}")
    print(f"Successful Requests: {stats['successful_requests']}")
    print(f"Failed Requests:     {stats['failed_requests']}")
    print(f"Success Rate:        {stats['success_rate']:.2f}%")
    print("\nPerformance Metrics:")
    print(f"Min Response Time:   {stats['min_time']*1000:.2f}ms")
    print(f"Max Response Time:   {stats['max_time']*1000:.2f}ms")
    print(f"Avg Response Time:   {stats['avg_time']*1000:.2f}ms")
    print(f"Median Response Time: {stats['median_time']*1000:.2f}ms")
    print(f"95th Percentile:     {stats['p95_time']*1000:.2f}ms")
    print(f"Throughput:          {stats['throughput']:.2f} requests/second")

    # Optionally test batch sizes
    print("\nWould you like to test different batch sizes? (y/n)")
    choice = input().lower().strip()
    if choice == "y":
        batch_results = test_batch_sizes()

        print("\nBatch Performance Results:")
        print("=========================")
        for size, result in batch_results.items():
            print(f"Batch Size: {size}")
            print(f"  Success:            {'Yes' if result['success'] else 'No'}")
            print(f"  Total Time:         {result['time']*1000:.2f}ms")
            print(f"  Time Per Prediction: {result['time_per_prediction']*1000:.2f}ms")
            print()


if __name__ == "__main__":
    main()
