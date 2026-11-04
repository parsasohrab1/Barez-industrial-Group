"""JWT + RBAC با ۵ نقش صنعتی"""
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from tvdi_industrial.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)


class Role(str, Enum):
    OPERATOR = "operator"
    QUALITY_MANAGER = "quality_manager"
    MAINTENANCE = "maintenance"
    ADMIN = "admin"
    AUDITOR = "auditor"


ROLE_PERMISSIONS = {
    Role.OPERATOR:       {"predict:read", "inspect:create"},
    Role.QUALITY_MANAGER: {"predict:read", "inspect:create", "grading:override",
                          "report:export", "trace:read"},
    Role.MAINTENANCE:    {"predict:read", "machine:calibrate", "reject:test"},
    Role.ADMIN:          {"*"},
    Role.AUDITOR:        {"predict:read", "report:export", "audit:read"},
}


class TokenUser(BaseModel):
    sub: str
    roles: List[Role]
    exp: int


class AuthService:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        self._users: dict[str, dict] = {}  # در تولید: Keycloak / DB

    def register_user(self, username: str, password: str, roles: List[Role]):
        self._users[username] = {
            "password": pwd_context.hash(password),
            "roles": roles,
        }

    def authenticate(self, username: str, password: str) -> Optional[TokenUser]:
        user = self._users.get(username)
        if not user or not pwd_context.verify(password, user["password"]):
            return None
        return self._make_token(username, user["roles"])

    def _make_token(self, username: str, roles: List[Role]) -> TokenUser:
        exp = datetime.now(timezone.utc) + timedelta(minutes=self.s.jwt_expire_minutes)
        return TokenUser(sub=username, roles=roles, exp=int(exp.timestamp()))


def decode_token(token: str) -> TokenUser:
    s = get_settings()
    try:
        data = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
        return TokenUser(
            sub=data["sub"],
            roles=[Role(r) for r in data["roles"]],
            exp=data["exp"],
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> TokenUser:
    if not creds:
        raise HTTPException(status_code=401, detail="missing token")
    return decode_token(creds.credentials)


def require_permission(permission: str):
    def _checker(user: TokenUser = Depends(get_current_user)) -> TokenUser:
        for role in user.roles:
            perms = ROLE_PERMISSIONS.get(role, set())
            if "*" in perms or permission in perms:
                return user
        raise HTTPException(status_code=403, detail=f"missing permission: {permission}")
    return _checker
