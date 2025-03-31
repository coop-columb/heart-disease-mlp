"""
Authentication module for the Heart Disease Prediction API.
This module provides JWT token and API key authentication for the API.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str
    expires_in: int
    scope: str = "api"


class APIKey(BaseModel):
    """API key model."""

    key: str
    name: str


class AuthSettings:
    """Authentication settings."""

    def __init__(self, config: Dict):
        """Initialize authentication settings from configuration."""
        auth_config = config.get("api", {}).get("auth", {})
        self.enabled = auth_config.get("enabled", True)
        self.secret_key = auth_config.get("secret_key", "YOUR_SECRET_KEY_HERE")
        self.algorithm = auth_config.get("algorithm", "HS256")
        self.access_token_expire_minutes = auth_config.get("access_token_expire_minutes", 30)
        self.token_url = auth_config.get("token_url", "/auth/token")

        # Load API keys
        self.api_keys: List[APIKey] = []
        for api_key in auth_config.get("api_keys", []):
            self.api_keys.append(APIKey(**api_key))

        # Load public endpoints
        self.public_endpoints = auth_config.get(
            "public_endpoints", ["/health", "/docs", "/redoc", "/openapi.json", "/auth/token"]
        )

        # Log configuration (excluding sensitive info)
        logger.info(f"Authentication enabled: {self.enabled}")
        logger.info(f"Token URL: {self.token_url}")
        logger.info(f"Public endpoints: {self.public_endpoints}")
        logger.info(f"Loaded {len(self.api_keys)} API keys")


class AuthHandler:
    """Authentication handler for JWT tokens and API keys."""

    def __init__(self, settings: AuthSettings):
        """Initialize authentication handler with settings."""
        self.settings = settings
        self.oauth2_scheme = (
            OAuth2PasswordBearer(tokenUrl=settings.token_url.lstrip("/"), auto_error=False)
            if settings.enabled
            else None
        )
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def create_access_token(self) -> Token:
        """Generate a new access token."""
        if not self.settings.enabled:
            # Return dummy token if auth is disabled
            return Token(access_token="disabled", token_type="bearer", expires_in=0)

        expire = datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes)
        to_encode = {"exp": expire}
        encoded_jwt = jwt.encode(
            to_encode, self.settings.secret_key, algorithm=self.settings.algorithm
        )

        return Token(
            access_token=encoded_jwt,
            token_type="bearer",
            expires_in=self.settings.access_token_expire_minutes * 60,
        )

    async def verify_token(self, token: str) -> bool:
        """Verify JWT token."""
        if not token or not self.settings.enabled:
            return False

        try:
            # Decode and validate token
            payload = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.settings.algorithm]
            )

            # Check if token has expired
            expire = payload.get("exp")
            if expire is None:
                return False

            if datetime.utcnow() > datetime.fromtimestamp(expire):
                return False

            return True
        except JWTError:
            return False

    async def verify_api_key(self, api_key: str) -> bool:
        """Verify API key."""
        if not api_key or not self.settings.enabled:
            return False

        for key in self.settings.api_keys:
            if key.key == api_key:
                return True

        return False

    # Authentication verification methods are now moved to app.py for proper dependency injection
