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
        results["response_status_counts"][str(status)] = results["response_status_counts"].get(str(status), 0) + a
        
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
    success_lock =

