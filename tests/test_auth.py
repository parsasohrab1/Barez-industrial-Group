from tvdi_industrial.security.auth import AuthService, Role, require_permission
from fastapi import HTTPException
import pytest


def test_auth_and_permissions():
    svc = AuthService()
    svc.register_user("op", "p", [Role.OPERATOR])
    token = svc.authenticate("op", "p")
    assert token is not None
    assert token.roles == [Role.OPERATOR]


def test_permission_denied():
    svc = AuthService()
    svc.register_user("op", "p", [Role.OPERATOR])
    token = svc.authenticate("op", "p")
    checker = require_permission("report:export")
    with pytest.raises(HTTPException) as exc:
        checker(token)
    assert exc.value.status_code == 403
