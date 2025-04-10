#!/usr/bin/env python3
"""
Test API rate limiting implementation.

This script:
1. Tests API rate limit thresholds
2. Validates correct 429 responses
3. Checks rate limit header information
4. Tests rate limit bypass protections
5. Verifies rate limit reset behavior
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def send_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    data: Optional[Dict] = None,
    auth_token: Optional[str] = None,
    timeout: float = 10.0
) -> Dict:
    """
    Send a request to the API and return timing and response information.
    
    Args:
        url: URL to send request to
        method: HTTP method (GET, POST, etc.)
        headers: Optional request headers
        data: Optional request data (for POST/PUT)
        auth_token: Optional authentication token
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with response information
    """
    # Prepare headers
    request_headers = headers.copy() if headers else {}
    if auth_token:
        request_headers["Authorization"] = f"Bearer {auth_token}"
    
    start_time = time.time()
    result = {
        "status_code": None,
        "response_time": None,
        "rate_limit_headers": {},
        "error": None
    }
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=request_headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, headers=request_headers, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        result["status_code"] = response.status_code
        result["response_time"] = time.time() - start_time
        
        # Extract rate limiting headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-RateLimit-Reset", 
            "Retry-After",
            "RateLimit-Limit",
            "RateLimit-Remaining",
            "RateLimit-Reset"
        ]
        
        for header in rate_limit_headers:
            if header in response.headers:
                result["rate_limit_headers"][header] = response.headers[header]
        
        # Try to parse response body
        try:
            result["response_body"] = response.json()
        except:
            result["response_body"] = response.text[:200] if response.text else None
            
    except Exception as e:
        result["error"] = str(e)
    
    return result

def find_rate_limit(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    data: Optional[Dict] = None,
    auth_token: Optional[str] = None,
    max_requests: int = 300,
    rps: int = 10,
    timeout: float = 10.0
) -> Tuple[bool, Dict]:
    """
    Find the rate limit by sending requests until receiving a 429 response.
    
    Args:
        url: URL to send request to
        method: HTTP method to use
        headers: Optional request headers
        data: Optional request data
        auth_token: Optional authentication token
        max_requests: Maximum number of requests to send
        rps: Requests per second to send
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (limit_found, details)
    """
    results = {
        "requests_sent": 0,
        "requests_per_second": rps,
        "first_429_at": None,
        "rate_limit_found": False,
        "rate_limit_value": None,
        "rate_limit_window": None,
        "rate_limit_reset": None,
        "rate_limit_headers": {},
        "response_status_counts": {},
        "issues": []
    }
    
    # Send requests until we get a 429 or reach max_requests
    for i in range(max_requests):
        result = send_request(url, method, headers, data, auth_token, timeout)
        results["requests_sent"] += 1
        
        # Count response statuses
        status = result["status_code"]
        results["response_status_counts"][str(status)] = results["response_status_counts"].get(str(status), 0) + 1
        
        # Check if we hit a rate limit
        if status == 429:
            if results["first_429_at"] is None:
                results["first_429_at"] = results["requests_sent"]
                results["rate_limit_found"] = True
                
                # Extract rate limit information from headers
                if result["rate_limit_headers"]:
                    results["rate_limit_headers"] = result["rate_limit_headers"]
                    
                    # Try to determine rate limit value
                    limit_header = None
                    for header in ["X-RateLimit-Limit", "RateLimit-Limit"]:
                        if header in result["rate_limit_headers"]:
                            limit_header = result["rate_limit_headers"][header]
                            break
                    
                    if limit_header:
                        try:
                            results["rate_limit_value"] = int(limit_header)
                        except:
                            pass
                    
                    # Try to determine reset time
                    reset_header = None
                    for header in ["X-RateLimit-Reset", "RateLimit-Reset", "Retry-After"]:
                        if header in result["rate_limit_headers"]:
                            reset_header = result["rate_limit_headers"][header]
                            break
                    
                    if reset_header:
                        try:
                            results["rate_limit_reset"] = int(reset_header)
                            # If we have both limit and reset, estimate window
                            if results["rate_limit_value"]:
                                results["rate_limit_window"] = results["rate_limit_reset"]
                        except:
                            pass
            
            # Continue sending requests to validate consistency of rate limiting
            if i >= results["first_429_at"] + 10:
                break
        
        # Sleep to maintain request rate
        if i < max_requests - 1:
            time.sleep(1.0 / rps)
    
    # Verify rate limit implementation
    if results["rate_limit_found"]:
        if not results["rate_limit_headers"]:
            results["issues"].append("Rate limit headers not provided in 429 response")
        
        if results["first_429_at"] is not None:
            if results["first_429_at"] < 10:
                results["issues"].append(f"Rate limit threshold seems too low: {results['first_429_at']} requests")
    else:
        results["issues"].append(f"No rate limit detected after {max_requests} requests")
    
    passed = len(results["issues"]) == 0
    
    return passed, results

def test_rate_limit_bypasses(
    url: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    auth_token: Optional[str] = None,
    rate_limit: Optional[int] = None
) -> Tuple[bool, Dict]:
    """
    Test various rate limit bypass techniques.
    
    Args:
        url: URL to send request to
        method: HTTP method to use
        data: Optional request data
        auth_token: Optional authentication token
        rate_limit: Number of requests to send for each bypass technique
        
    Returns:
        Tuple of (passed, details)
    """
    if not rate_limit:
        rate_limit = 50  # Default if not provided
    
    results = {
        "bypass_attempts": {},
        "issues": []
    }
    
    # Define bypass techniques to test
    bypass_techniques = {
        "different_user_agents": [
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"},
            {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            {"User-Agent": "Python Requests/2.28.1"},
            {"User-Agent": "curl/7.79.1"}
        ],
        "different_ip_headers": [
            {"X-Forwarded-For": "192.168.1.1"},
            {"X-Forwarded-For": "10.0.0.1"},
            {"X-Real-IP": "172.16.0.1"},
            {"CF-Connecting-IP": "8.8.8.8"}
        ],
        "cache_bypass": [
            {"Cache-Control": "no-cache", "Pragma": "no-cache"},
            {"Cache-Control": "no-store"},
            {}  # No cache headers as control
        ]
    }
    
    # Test each bypass technique
    for technique_name, technique_headers in bypass_techniques.items():
        technique_results = []
        success_count = 0
        
        # Send requests for each header variation
        for headers in technique_headers:
            # Send requests rapidly
            for _ in range(min(20, rate_limit // len(technique_headers))):
                result = send_request(url, method, headers, data, auth_token)
                technique_results.append(result)
                
                # Count successful requests (non-429)
                if result["status_code"] != 429:
                    success_count += 1
        
        # Store results
        results["bypass_attempts"][technique_name] = {
            "success_count": success_count,
            "total_requests": len(technique_results),
            "success_rate": success_count / len(technique_results) if technique_results else 0
        }
        
        # Check if bypass was successful
        success_rate = results["bypass_attempts"][technique_name]["success_rate"]
        if success_rate > 0.8:  # If more than 80% of requests succeeded after rate limit should be hit
            results["issues"].append(f"Rate limit bypass possible using {technique_name}: {success_rate:.2%} success rate")
    
    passed = len(results["issues"]) == 0
    
    return passed, results

def test_rate_limit_reset(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    data: Optional[Dict] = None,
    auth_token: Optional[str] = None,
    rate_limit: Optional[int] = None
) -> Tuple[bool, Dict]:
    """
    Test if rate limit properly resets after specified window.
    
    Args:
        url: URL to send request to
        method: HTTP method to use
        headers: Optional request headers
        data: Optional request data
        auth_token: Optional authentication token
        rate_limit: Optional rate limit value (if known)
        
    Returns:
        Tuple of (passed, details)
    """
    results = {
        "reset_tested": False,
        "reset_time_found": False,
        "estimated_reset_time": None,
        "reset_successful": False,
        "issues": []
    }
    
    try:
        # First, hit the rate limit to get reset information
        _, limit_results = find_rate_limit(
            url, method, headers, data, auth_token,
            max_requests=rate_limit * 2 if rate_limit else 100
        )
        
        if not limit_results["rate_limit_found"]:
            results["issues"].append("Could not trigger rate limit to test reset")
            return False, results
        
        # Extract reset time information
        reset_time = limit_results.get("rate_limit_reset")
        
        if reset_time:
            results["reset_time_found"] = True
            results["estimated_reset_time"] = reset_time
            
            # Wait for reset time plus a small buffer
            wait_time = reset_time + 1
            logger.info(f"Waiting {wait_time} seconds for rate limit to reset...")
            time.sleep(wait_time)
            
            # Test if we can make requests again
            result = send_request(url, method, headers, data, auth_token)
            results["reset_tested"] = True
            results["reset_successful"] = result["status_code"] != 429
            
            if not results["reset_successful"]:
                results["issues"].append(f"Rate limit did not reset after waiting {wait_time} seconds")
        else:
            # If we don't have reset time, make an educated guess (usually 1 hour or 1 minute)
            for wait_time in [60, 65]:  # Try 1 minute + buffer
                logger.info(f"No reset time found. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
                result = send_request(url, method, headers, data, auth_token)
                results["reset_tested"] = True
                
                if result["status_code"] != 429:
                    results["reset_successful"] = True
                    results["estimated_reset_time"] = wait_time
                    break
            
            if not results["reset_successful"]:
                results["issues"].append("Rate limit did not reset after waiting up to 65 seconds")
    
    except Exception as e:
        logger.error(f"Error testing rate limit reset: {e}")
        results["issues"].append(f"Error testing rate limit reset: {str(e)}")
    
    passed = len(results["issues"]) == 0
    
    return passed, results

def test_concurrent_limits(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    data: Optional[Dict] = None,
    auth_token: Optional[str] = None,
    concurrent_users: int = 5,
    requests_per_user: int = 20
) -> Tuple[bool, Dict]:
    """
    Test if rate limits are properly applied under concurrent load.
    
    Args:
        url: URL to send request to
        method: HTTP method to use
        headers: Optional request headers
        data: Optional request data
        auth_token: Optional authentication token
        concurrent_users: Number of concurrent users to simulate
        requests_per_user: Number of requests per simulated user
        
    Returns:
        Tuple of (passed, details)
    """
    results = {
        "concurrent_users": concurrent_users,
        "requests_per_user": requests_per_user,
        "total_requests": concurrent_users * requests_per_user,
        "successful_requests": 0,
        "rate_limited_requests": 0,
        "error_requests": 0,
        "user_results": {},
        "issues": []
    }
    
    rate_limit_lock = threading.Lock()
    success_lock = threading.Lock()
    error_lock = threading.Lock()
    
    def worker(user_id):
        """Worker function for each simulated user"""
        user_results = {
            "requests_sent": 0,
            "successful_requests": 0,
            "rate_limited_requests": 0,
            "error_requests": 0
        }
        
        for i in range(requests_per_user):
            # Send request
            result = send_request(url, method, headers, data, auth_token)
            user_results["requests_sent"] += 1
            
            # Track result
            if result["status_code"] == 429:
                user_results["rate_limited_requests"] += 1
                with rate_limit_lock:
                    results["rate_limited_requests"] += 1
            elif result["error"]:
                user_results["error_requests"] += 1
                with error_lock:
                    results["error_requests"] += 1
            elif result["status_code"] and 200 <= result["status_code"] < 300:
                user_results["successful_requests"] += 1
                with success_lock:
                    results["successful_requests"] += 1
            else:
                user_results["error_requests"] += 1
                with error_lock:
                    results["error_requests"] += 1
            
            # Small delay between requests for same user
            time.sleep(0.1)
        
        return user_id, user_results
    
    # Create and start worker threads
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = {executor.submit(worker, i): i for i in range(concurrent_users)}
        
        for future in futures:
            user_id, user_result = future.result()
            results["user_results"][str(user_id)] = user_result
    
    # Analyze results
    if results["rate_limited_requests"] == 0:
        results["issues"].append("No rate limiting detected under concurrent load")
    
    # Check if rate limiting was consistent across users
    rate_limited_users = sum(1 for user_data in results["user_results"].values() 
                            if user_data["rate_limited_requests"] > 0)
    
    if 0 < rate_limited_users < concurrent_users:
        # Only some users were rate limited
        results["issues"].append(
            f"Inconsistent rate limiting: only {rate_limited_users}/{concurrent_users} users were rate limited"
        )
    
    passed = len(results["issues"]) == 0
    
    return passed, results

def test_rate_limiting(
    base_url: str,
    endpoint: str,
    method: str = "GET",
    auth_token: Optional[str] = None,
    data: Optional[Dict] = None,
    max_requests: int = 300,
    rps: int = 10,
    concurrent_users: int = 5,
    run_reset_test: bool = True,
    run_bypass_test: bool = True,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Perform comprehensive rate limiting tests.
    
    Args:
        base_url: Base URL of the API
        endpoint: Endpoint to test
        method: HTTP method to use
        auth_token: Optional authentication token
        data: Optional request data for POST requests
        max_requests: Maximum number of requests to send
        rps: Requests per second to send
        concurrent_users: Number of concurrent users to simulate
        run_reset_test: Whether to test rate limit reset
        run_bypass_test: Whether to test rate limit bypass techniques
        output_path: Optional path to save results
        
    Returns:
        Tuple of (passed, results)
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {}
    
    try:
        # 1. Find rate limit threshold
        threshold_passed, threshold_results = find_rate_limit(
            url, method, headers, data, auth_token, max_requests, rps
        )
        
        # Extract rate limit value if found
        rate_limit = threshold_results.get("rate_limit_value")
        
        # 2. Test concurrent rate limiting
        concurrent_passed, concurrent_results = test_concurrent_limits(
            url, method, headers, data, auth_token, concurrent_users, 
            max(20, rate_limit // 2 if rate_limit else 50)
        )
        
        # 3. Test rate limit bypass techniques
        bypass_passed = True
        bypass_results = {"bypass_attempts": {}, "issues": []}
        
        if run_bypass_test and threshold_results["rate_limit_found"]:
            bypass_passed, bypass_results = test_rate_limit_bypasses(
                url, method, data, auth_token, rate_limit
            )
        
        # 4. Test rate limit reset
        reset_passed = True
        reset_results = {"reset_tested": False, "issues": []}
        
        if run_reset_test and threshold_results["rate_limit_found"]:
            reset_passed, reset_results = test_rate_limit_reset(
                url, method, headers, data, auth_token, rate_limit
            )
        
        # Combine results
        results = {
            "threshold_detection": {
                "passed": threshold_passed,
                "details": threshold_results
            },
            "concurrent_testing": {
                "passed": concurrent_passed,
                "details": concurrent_results
            },
            "bypass_testing": {
                "passed": bypass_passed,
                "details": bypass_results
            },
            "reset_testing": {
                "passed": reset_passed,
                "details": reset_results
            }
        }
        
        # Overall pass/fail
        passed = threshold_passed and concurrent_passed and bypass_passed and reset_passed
        
        # Save results if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        return passed, results
    
    except Exception as e:
        logger.error(f"Error testing rate limiting: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Test API rate limiting implementation."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the API"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Endpoint to test"
    )
    parser.add_argument(
        "--method",
        default="GET",
        help="HTTP method to use"
    )
    parser.add_argument(
        "--auth-token",
        help="Authentication token"
    )
    parser.add_argument(
        "--data-file",
        help="JSON file with request data"
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=300,
        help="Maximum number of requests to send"
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=10,
        help="Requests per second to send"
    )
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=5,
        help="Number of concurrent users to simulate"
    )
    parser.add_argument(
        "--no-reset-test",
        action="store_true",
        help="Skip rate limit reset test"
    )
    parser.add_argument(
        "--no-bypass-test",
        action="store_true",
        help="Skip rate limit bypass test"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Load data file if provided
    data = None
    if args.data_file:
        with open(args.data_file) as f:
            data = json.load(f)
    
    try:
        passed, results = test_rate_limiting(
            args.base_url,
            args.endpoint,
            args.method,
            args.auth_token,
            data,
            args.max_requests,
            args.rps,
            args.concurrent_users,
            not args.no_reset_test,
            not args.no_bypass_test,
            args.output_path
        )
        
        # Log results
        logger.info("Rate Limiting Test Results:")
        
        # Log threshold detection results
        threshold_results = results["threshold_detection"]
        logger.info(f"Rate Limit Threshold Detection: {'passed' if threshold_results['passed'] else 'failed'}")
        
        if threshold_results['details']['rate_limit_found']:
            logger.info(f"  Found rate limit after {threshold_results['details']['first_429_at']} requests")
            if threshold_results['details']['rate_limit_value']:
                logger.info(f"  Rate limit value: {threshold_results['details']['rate_limit_value']}")
            if threshold_results['details']['rate_limit_reset']:
                logger.info(f"  Reset time: {threshold_results['details']['rate_limit_reset']} seconds")
        else:
            logger.warning("  No rate limit detected")
        
        # Log issues
        if not threshold_results['passed']:
            for issue in threshold_results['details']['issues']:
                logger.warning(f"  - {issue}")
        
        # Log concurrent testing results
        concurrent_results = results["concurrent_testing"]
        logger.info(f"Concurrent Rate Limit Testing: {'passed' if concurrent_results['passed'] else 'failed'}")
        logger.info(f"  Successful requests: {concurrent_results['details']['successful_requests']}")
        logger.info(f"  Rate limited requests: {concurrent_results['details']['rate_limited_requests']}")
        
        if not concurrent_results['passed']:
            for issue in concurrent_results['details']['issues']:
                logger.warning(f"  - {issue}")
        
        # Log bypass testing results if performed
        bypass_results = results["bypass_testing"]
        if bypass_results['details']['bypass_attempts']:
            logger.info(f"Rate Limit Bypass Testing: {'passed' if bypass_results['passed'] else 'failed'}")
            
            for technique, technique_results in bypass_results['details']['bypass_attempts'].items():
                logger.info(f"  {technique}: {technique_results['success_rate']:.1%} success rate")
            
            if not bypass_results['passed']:
                for issue in bypass_results['details']['issues']:
                    logger.warning(f"  - {issue}")
        
        # Log reset testing results if performed
        reset_results = results["reset_testing"]
        if reset_results['details']['reset_tested']:
            logger.info(f"Rate Limit Reset Testing: {'passed' if reset_results['passed'] else 'failed'}")
            
            if reset_results['details']['reset_successful']:
                logger.info(f"  Reset successful after {reset_results['details']['estimated_reset_time']} seconds")
            else:
                logger.warning("  Reset did not occur as expected")
            
            if not reset_results['passed']:
                for issue in reset_results['details']['issues']:
                    logger.warning(f"  - {issue}")
        
        # Overall result
        logger.info(f"Overall validation {'passed' if passed else 'failed'}")
        
        # Exit with status
        sys.exit(0 if passed else 1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
