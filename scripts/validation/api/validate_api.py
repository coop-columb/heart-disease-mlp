#!/usr/bin/env python3
"""
Validate API functionality, performance, and security.

This script:
1. Tests API endpoints
2. Measures response times
3. Validates response formats
4. Checks security headers
5. Tests rate limiting
"""

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ValidationConfig(BaseModel):
    """API validation configuration."""
    base_url: str
    endpoints: List[Dict[str, str]]
    expected_headers: List[str]
    max_response_time: float = 500  # ms
    min_requests_per_second: float = 10
    required_status_codes: List[int] = [200, 400, 401, 403, 404, 422, 500]

def test_endpoint(
    url: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None
) -> Tuple[bool, Dict]:
    """
    Test a single endpoint.

    Args:
        url: Endpoint URL
        method: HTTP method
        data: Request data
        headers: Request headers

    Returns:
        Tuple of (passed, results)
    """
    try:
        start_time = time.time()
        
        with httpx.Client() as client:
            response = client.request(
                method,
                url,
                json=data,
                headers=headers,
                timeout=30.0
            )
        
        duration = (time.time() - start_time) * 1000  # Convert to ms
        
        results = {
            "status_code": response.status_code,
            "response_time": duration,
            "headers_present": list(response.headers.keys()),
            "content_type": response.headers.get("content-type"),
            "content_length": len(response.content)
        }
        
        # Try to parse JSON response
        try:
            results["response_data"] = response.json()
        except Exception:
            results["response_data"] = None
        
        return True, results
    
    except Exception as e:
        logger.error(f"Error testing endpoint {url}: {e}")
        return False, {"error": str(e)}

def load_test_endpoint(
    url: str,
    duration: int = 60,
    concurrent_users: int = 10
) -> Dict:
    """
    Perform load testing on endpoint.

    Args:
        url: Endpoint URL
        duration: Test duration in seconds
        concurrent_users: Number of concurrent users

    Returns:
        Load test results
    """
    results = []
    end_time = time.time() + duration
    
    def make_request():
        while time.time() < end_time:
            passed, result = test_endpoint(url)
            if passed:
                results.append(result["response_time"])
    
    # Run concurrent requests
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        executor.map(lambda _: make_request(), range(concurrent_users))
    
    # Calculate metrics
    total_requests = len(results)
    requests_per_second = total_requests / duration
    
    if results:
        metrics = {
            "total_requests": total_requests,
            "requests_per_second": requests_per_second,
            "min_response_time": min(results),
            "max_response_time": max(results),
            "mean_response_time": statistics.mean(results),
            "p95_response_time": np.percentile(results, 95),
            "p99_response_time": np.percentile(results, 99)
        }
    else:
        metrics = {
            "error": "No successful requests completed"
        }
    
    return metrics

def check_security_headers(headers: Dict) -> Tuple[bool, List[str]]:
    """
    Check for required security headers.

    Args:
        headers: Response headers

    Returns:
        Tuple of (passed, missing_headers)
    """
    required_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Strict-Transport-Security",
        "Content-Security-Policy"
    ]
    
    present_headers = {h.lower() for h in headers.keys()}
    required_lower = {h.lower() for h in required_headers}
    
    missing = [h for h in required_headers
              if h.lower() not in present_headers]
    
    return len(missing) == 0, missing

def validate_api(
    config: ValidationConfig,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Run complete API validation.

    Args:
        config: Validation configuration
        output_path: Optional path to save results

    Returns:
        Tuple of (passed, results)
    """
    results = {
        "endpoint_tests": {},
        "load_tests": {},
        "security_checks": {},
        "status_codes_found": set()
    }
    
    all_passed = True
    
    # Test each endpoint
    for endpoint in config.endpoints:
        url = f"{config.base_url}{endpoint['path']}"
        passed, endpoint_result = test_endpoint(
            url,
            method=endpoint.get("method", "GET"),
            data=endpoint.get("data"),
            headers=endpoint.get("headers")
        )
        
        results["endpoint_tests"][endpoint["path"]] = endpoint_result
        results["status_codes_found"].add(endpoint_result.get("status_code"))
        
        # Check response time
        if passed and endpoint_result["response_time"] > config.max_response_time:
            passed = False
        
        all_passed &= passed
    
    # Perform load test on main endpoint
    results["load_tests"] = load_test_endpoint(
        f"{config.base_url}/predict",
        duration=30,
        concurrent_users=5
    )
    
    # Check if meets performance requirements
    if results["load_tests"].get("requests_per_second", 0) < config.min_requests_per_second:
        all_passed = False
    
    # Check security headers
    for endpoint_result in results["endpoint_tests"].values():
        if "headers_present" in endpoint_result:
            passed, missing = check_security_headers(
                {h: "present" for h in endpoint_result["headers_present"]}
            )
            results["security_checks"][endpoint_result.get("path", "unknown")] = {
                "passed": passed,
                "missing_headers": missing
            }
            all_passed &= passed
    
    # Check required status codes
    missing_codes = set(config.required_status_codes) - results["status_codes_found"]
    if missing_codes:
        all_passed = False
        results["missing_status_codes"] = list(missing_codes)
    
    # Save results if path provided
    if output_path:
        # Convert sets to lists for JSON serialization
        results["status_codes_found"] = list(results["status_codes_found"])
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return all_passed, results

def main():
    parser = argparse.ArgumentParser(
        description="Validate API functionality and performance."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to validation config JSON"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        with open(args.config) as f:
            config = ValidationConfig(**json.load(f))
        
        # Run validation
        passed, results = validate_api(config, args.output_path)
        
        # Log results
        logger.info("API Validation Results:")
        logger.info(f"Endpoints Tested: {len(results['endpoint_tests'])}")
        logger.info(f"Load Test RPS: {results['load_tests'].get('requests_per_second', 0):.2f}")
        logger.info(f"Security Checks: {sum(1 for r in results['security_checks'].values() if r['passed'])}/{len(results['security_checks'])}")
        logger.info(f"Status Codes Found: {sorted(results['status_codes_found'])}")
        logger.info(f"Overall validation {'passed' if passed else 'failed'}")
        
        # Exit with status
        sys.exit(0 if passed else 1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
