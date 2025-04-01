import secrets
from datetime import datetime, timedelta

import PyJWT as jwt
from pydantic import BaseModel


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"


class AuthSettings:
    """Authentication settings."""

    def __init__(self, config: dict):
        auth_config = config.get("auth", {})
        self.enabled = auth_config.get("enabled", True)
        self.secret_key = auth_config.get("secret_key", secrets.token_urlsafe(32))
        self.token_expire_minutes = auth_config.get("token_expire_minutes", 30)
        self.api_keys = set(auth_config.get("api_keys", []))
        self.public_endpoints = set(
            auth_config.get("public_endpoints", ["/docs", "/openapi.json", "/health"])
        )


class AuthHandler:
    """Authentication handler."""

    def __init__(self, settings: AuthSettings):
        self.settings = settings
        self.algorithm = "HS256"

    def create_access_token(self) -> Token:
        """Create a new access token."""
        expires_delta = timedelta(minutes=self.settings.token_expire_minutes)
        expire = datetime.utcnow() + expires_delta
        token_data = {"exp": expire}
        encoded_jwt = jwt.encode(
            token_data, self.settings.secret_key, algorithm=self.algorithm
        )
        return Token(access_token=encoded_jwt)

    async def verify_token(self, token: str) -> bool:
        """Verify a JWT token."""
        try:
            jwt.decode(token, self.settings.secret_key, algorithms=[self.algorithm])
            return True
        except jwt.PyJWTError:
            return False

    async def verify_api_key(self, api_key: str) -> bool:
        """Verify an API key."""
        return api_key in self.settings.api_keys
