#!/usr/bin/env python3
"""
Check API authentication security implementation.

This script:
1. Validates JWT implementation security
2. Tests authentication flow
3. Checks token handling mechanisms
4. Verifies protection of sensitive endpoints
5. Tests response to invalid credentials
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
import jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def validate_jwt_security(token: str) -> Tuple[bool, Dict]:
    """
    Validate JWT token implementation for security best practices.
    
    Args:
        token: JWT token to analyze
        
    Returns:
        Tuple of (passed, details)
    """
    issues = []
    details = {
        "algorithm": None,
        "claims": [],
        "has_expiration": False,
        "has_issued_at": False,
        "exp_duration_seconds": None,
        "issues": []
    }
    
    try:
        # Decode header without verification to check algorithm
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg", "none")
        details["algorithm"] = algorithm
        
        # Check for weak algorithms
        if algorithm in ["none", "HS256"]:
            issues.append(f"Insecure algorithm detected: {algorithm}")
        
        # Decode payload without verification to check claims
        payload = jwt.decode(token, options={"verify_signature": False})
        details["claims"] = list(payload.keys())
        
        # Check for required claims
        details["has_expiration"] = "exp" in payload
        details["has_issued_at"] = "iat" in payload
        
        if not details["has_expiration"]:
            issues.append("Token missing 'exp' claim (expiration)")
        
        if not details["has_issued_at"]:
            issues.append("Token missing 'iat' claim (issued at)")
        
        # Check for token lifetime if both exp and iat are present
        if details["has_expiration"] and details["has_issued_at"]:
            exp_time = payload["exp"]
            iat_time = payload["iat"]
            token_lifetime = exp_time - iat_time
            details["exp_duration_seconds"] = token_lifetime
            
            # Check if token lifetime is too long (>24 hours) or too short (<1 minute)
            if token_lifetime > 86400:  # 24 hours in seconds
                issues.append(f"Token lifetime too long: {token_lifetime} seconds (>24 hours)")
            elif token_lifetime < 60:  # 1 minute in seconds
                issues.append(f"Token lifetime too short: {token_lifetime} seconds (<1 minute)")
        
        # Check for proper subject claim
        if "sub" not in payload:
            issues.append("Token missing 'sub' claim (subject)")
        
        # Check for sensitive data in token
        sensitive_keys = ["password", "secret", "key", "credential"]
        for key in payload.keys():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                issues.append(f"Token contains potentially sensitive data: {key}")
        
        details["issues"] = issues
        passed = len(issues) == 0
        
        return passed, details
    
    except Exception as e:
        logger.error(f"Error validating JWT security: {e}")
        issues.append(f"Failed to parse JWT: {str(e)}")
        details["issues"] = issues
        return False, details

def test_authentication_flow(
    base_url: str,
    login_endpoint: str,
    protected_endpoint: str,
    refresh_endpoint: Optional[str],
    username: str,
    password: str
) -> Tuple[bool, Dict]:
    """
    Test the complete authentication flow.
    
    Args:
        base_url: Base URL of the API
        login_endpoint: Endpoint for login
        protected_endpoint: Protected endpoint to test with token
        refresh_endpoint: Optional endpoint for token refresh
        username: Valid username for testing
        password: Valid password for testing
        
    Returns:
        Tuple of (passed, details)
    """
    issues = []
    details = {
        "login_status": None,
        "protected_access": False,
        "refresh_token_provided": False,
        "refresh_successful": None,
        "jwt_security": None,
        "invalid_login_response": None,
        "invalid_token_response": None,
        "issues": []
    }
    
    try:
        # Create session to maintain cookies if used
        session = requests.Session()
        
        # 1. Test login with valid credentials
        login_url = f"{base_url.rstrip('/')}/{login_endpoint.lstrip('/')}"
        login_data = {"username": username, "password": password}
        
        login_response = session.post(login_url, json=login_data)
        details["login_status"] = login_response.status_code
        
        if login_response.status_code != 200:
            issues.append(f"Login failed with status {login_response.status_code}")
            details["issues"] = issues
            return False, details
        
        # Extract token from response
        login_json = login_response.json()
        access_token = None
        refresh_token = None
        
        # Handle different token response formats
        if "token" in login_json:
            access_token = login_json["token"]
        elif "access_token" in login_json:
            access_token = login_json["access_token"]
        else:
            for key in login_json.keys():
                if "token" in key.lower() and "refresh" not in key.lower():
                    access_token = login_json[key]
                    break
        
        # Check for refresh token
        if "refresh_token" in login_json:
            refresh_token = login_json["refresh_token"]
            details["refresh_token_provided"] = True
        else:
            for key in login_json.keys():
                if "refresh" in key.lower() and "token" in key.lower():
                    refresh_token = login_json[key]
                    details["refresh_token_provided"] = True
                    break
        
        if not access_token:
            issues.append("No access token found in login response")
            details["issues"] = issues
            return False, details
        
        # 2. Validate JWT implementation
        jwt_passed, jwt_details = validate_jwt_security(access_token)
        details["jwt_security"] = jwt_details
        if not jwt_passed:
            issues.extend(jwt_details["issues"])
        
        # 3. Test access to protected endpoint
        protected_url = f"{base_url.rstrip('/')}/{protected_endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        protected_response = session.get(protected_url, headers=headers)
        details["protected_access"] = protected_response.status_code == 200
        
        if not details["protected_access"]:
            issues.append(f"Access to protected endpoint failed with status {protected_response.status_code}")
        
        # 4. Test token refresh if endpoint provided and refresh token available
        if refresh_endpoint and refresh_token:
            refresh_url = f"{base_url.rstrip('/')}/{refresh_endpoint.lstrip('/')}"
            refresh_data = {"refresh_token": refresh_token}
            
            refresh_response = session.post(refresh_url, json=refresh_data)
            details["refresh_successful"] = refresh_response.status_code == 200
            
            if not details["refresh_successful"]:
                issues.append(f"Token refresh failed with status {refresh_response.status_code}")
            else:
                # Extract new access token
                refresh_json = refresh_response.json()
                new_token = None
                
                if "token" in refresh_json:
                    new_token = refresh_json["token"]
                elif "access_token" in refresh_json:
                    new_token = refresh_json["access_token"]
                
                if not new_token:
                    issues.append("No new access token found in refresh response")
                else:
                    # Verify new token works
                    new_headers = {"Authorization": f"Bearer {new_token}"}
                    new_response = session.get(protected_url, headers=new_headers)
                    
                    if new_response.status_code != 200:
                        issues.append(f"Access with refreshed token failed with status {new_response.status_code}")
        
        # 5. Test invalid login credentials
        invalid_login_data = {"username": username, "password": "wrong_password"}
        invalid_login_response = session.post(login_url, json=invalid_login_data)
        details["invalid_login_response"] = invalid_login_response.status_code
        
        if invalid_login_response.status_code < 400:
            issues.append(f"Invalid login accepted with status {invalid_login_response.status_code}")
        
        # 6. Test invalid/expired token
        invalid_token = "invalid.token.format"
        invalid_headers = {"Authorization": f"Bearer {invalid_token}"}
        invalid_token_response = session.get(protected_url, headers=invalid_headers)
        details["invalid_token_response"] = invalid_token_response.status_code
        
        if invalid_token_response.status_code < 400:
            issues.append(f"Invalid token accepted with status {invalid_token_response.status_code}")
        
        details["issues"] = issues
        passed = len(issues) == 0
        
        return passed, details
    
    except Exception as e:
        logger.error(f"Error testing authentication flow: {e}")
        issues.append(f"Error testing authentication flow: {str(e)}")
        details["issues"] = issues
        return False, details

def check_authorization_scopes(
    base_url: str,
    login_endpoint: str,
    endpoints: List[Dict],
    username: str,
    password: str
) -> Tuple[bool, Dict]:
    """
    Check if authorization scopes are properly enforced.
    
    Args:
        base_url: Base URL of the API
        login_endpoint: Endpoint for login
        endpoints: List of endpoints to test, each with expected permission
        username: Valid username for testing
        password: Valid password for testing
        
    Returns:
        Tuple of (passed, details)
    """
    issues = []
    details = {
        "endpoint_results": {},
        "issues": []
    }
    
    try:
        # Login to get token
        login_url = f"{base_url.rstrip('/')}/{login_endpoint.lstrip('/')}"
        login_data = {"username": username, "password": password}
        
        login_response = requests.post(login_url, json=login_data)
        if login_response.status_code != 200:
            issues.append(f"Login failed with status {login_response.status_code}")
            details["issues"] = issues
            return False, details
        
        # Extract token
        login_json = login_response.json()
        token = login_json.get("access_token") or login_json.get("token")
        
        if not token:
            issues.append("No token found in login response")
            details["issues"] = issues
            return False, details
        
        # Test each endpoint
        for endpoint_info in endpoints:
            endpoint = endpoint_info["endpoint"]
            expected_status = endpoint_info.get("expected_status", 200)
            should_have_access = endpoint_info.get("should_have_access", True)
            
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(url, headers=headers)
            status = response.status_code
            
            endpoint_result = {
                "status": status,
                "expected_status": expected_status,
                "should_have_access": should_have_access,
                "passed": False
            }
            
            # Check if access matches expectation
            has_access = 200 <= status < 300
            if has_access == should_have_access and (has_access or status == expected_status):
                endpoint_result["passed"] = True
            else:
                if should_have_access:
                    issues.append(f"Expected access to {endpoint} but got status {status}")
                else:
                    issues.append(f"Expected no access to {endpoint} but got status {status}")
            
            details["endpoint_results"][endpoint] = endpoint_result
        
        details["issues"] = issues
        passed = len(issues) == 0
        
        return passed, details
    
    except Exception as e:
        logger.error(f"Error checking authorization scopes: {e}")
        issues.append(f"Error checking authorization scopes: {str(e)}")
        details["issues"] = issues
        return False, details

def check_auth_security(
    base_url: str,
    username: str,
    password: str,
    login_endpoint: str = "auth/login",
    protected_endpoint: str = "predict",
    refresh_endpoint: Optional[str] = None,
    scope_endpoints: Optional[List[Dict]] = None,
    output_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Perform comprehensive authentication security checks.
    
    Args:
        base_url: Base URL of the API
        username: Valid username for testing
        password: Valid password for testing
        login_endpoint: Endpoint for login
        protected_endpoint: Protected endpoint to test
        refresh_endpoint: Optional endpoint for token refresh
        scope_endpoints: Optional list of endpoints with expected permissions
        output_path: Optional path to save results
        
    Returns:
        Tuple of (passed, results)
    """
    try:
        # Test authentication flow
        auth_passed, auth_details = test_authentication_flow(
            base_url,
            login_endpoint,
            protected_endpoint,
            refresh_endpoint,
            username,
            password
        )
        
        # Test authorization scopes if provided
        scope_passed = True
        scope_details = {"endpoint_results": {}, "issues": []}
        
        if scope_endpoints:
            scope_passed, scope_details = check_authorization_scopes(
                base_url,
                login_endpoint,
                scope_endpoints,
                username,
                password
            )
        
        # Combine results
        results = {
            "authentication_flow": {
                "passed": auth_passed,
                "details": auth_details
            },
            "authorization_scopes": {
                "passed": scope_passed,
                "details": scope_details
            }
        }
        
        # Overall pass/fail
        passed = auth_passed and scope_passed
        
        # Save results if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        return passed, results
    
    except Exception as e:
        logger.error(f"Error checking auth security: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Check

