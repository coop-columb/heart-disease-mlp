import secrets
from datetime import datetime, timedelta

import jwt
from pydantic import BaseModel


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = None


class AuthSettings:
    """Authentication settings."""

    def __init__(self, config: dict):
        auth_config = config.get("api", {}).get("auth", {})
        self.enabled = auth_config.get("enabled", True)
        self.secret_key = auth_config.get("secret_key", secrets.token_urlsafe(32))
        self.token_expire_minutes = auth_config.get("access_token_expire_minutes", 30)
        self.token_includes_expires_in = auth_config.get(
            "token_includes_expires_in", False
        )
        self._config = config
        self.public_endpoints = set(
            auth_config.get("public_endpoints", ["/docs", "/openapi.json", "/health"])
        )

    @property
    def api_keys(self):
        """Get the set of valid API keys."""
        if not hasattr(self, "_api_keys"):
            auth_config = self._config.get("api", {}).get("auth", {})
            api_keys_config = auth_config.get("api_keys", [])
            self._api_keys = {item["key"] for item in api_keys_config if "key" in item}
        return self._api_keys


class AuthHandler:
    """Authentication handler."""

    def __init__(self, settings: AuthSettings):
        self.settings = settings
        self.algorithm = "HS256"

    def create_access_token(self) -> Token:
        """Create a new access token."""
        # Convert minutes to seconds for expires_in
        ACCESS_TOKEN_EXPIRE_SECONDS = self.settings.token_expire_minutes * 60
        expires_delta = timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
        expire = datetime.utcnow() + expires_delta
        token_data = {"exp": expire}
        encoded_jwt = jwt.encode(
            token_data, self.settings.secret_key, algorithm=self.algorithm
        )

        # Always include expires_in for consistency
        token_response = Token(
            access_token=encoded_jwt,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
        )

        return token_response

    async def verify_token(self, token: str) -> dict:
        """Verify a JWT token and return decoded payload.

        Args:
            token: JWT token to verify

        Returns:
            dict: Decoded payload if valid, empty dict if invalid
        """
        try:
            payload = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.algorithm]
            )
            return payload
        except jwt.PyJWTError:
            return {}

    async def verify_api_key(self, api_key: str) -> bool:
        """Verify an API key.

        Args:
            api_key: API key to verify

        Returns:
            bool: True if API key is valid, False otherwise
        """
        if not self.settings.enabled:
            return True

        return api_key in self.settings.api_keys
