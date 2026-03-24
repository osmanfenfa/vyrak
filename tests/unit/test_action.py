"""tests/unit/test_action.py — UserAction unit tests."""

import pytest
from core.action import UserAction


def test_valid_creation():
    a = UserAction("read", "document_42")
    assert a.action_type == "read"
    assert len(a.resource_id) == 16          # truncated SHA-256
    assert a.timestamp > 0


def test_case_normalisation():
    a = UserAction("READ", "res")
    assert a.action_type == "read"


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        UserAction("fly", "res")


def test_invalid_resource_raises():
    with pytest.raises(TypeError):
        UserAction("read", 123)


def test_to_index_all_types():
    for i, t in enumerate(UserAction.ACTION_TYPES):
        a = UserAction(t, "r")
        assert a.to_index() == i


def test_sensitive_types():
    for t in ("delete", "export", "config", "admin"):
        assert UserAction(t, "r").is_sensitive()
    for t in ("read", "write", "search", "login"):
        assert not UserAction(t, "r").is_sensitive()


def test_high_risk_types():
    assert UserAction("admin", "r").is_high_risk()
    assert UserAction("export", "r").is_high_risk()
    assert not UserAction("read", "r").is_high_risk()


def test_resource_hashing_deterministic():
    a1 = UserAction("read", "same_resource")
    a2 = UserAction("read", "same_resource")
    assert a1.resource_id == a2.resource_id


def test_different_resources_different_hashes():
    a1 = UserAction("read", "doc_A")
    a2 = UserAction("read", "doc_B")
    assert a1.resource_id != a2.resource_id


def test_serialisation_round_trip():
    import time
    a = UserAction("write", "doc_99", session_id="s1", timestamp=time.time(),
                   metadata={"ip": "1.2.3.4"})
    d   = a.to_dict()
    a2  = UserAction.from_dict(d)
    assert a2.action_type  == a.action_type
    assert a2.resource_id  == a.resource_id
    assert a2.session_id   == a.session_id


def test_from_index():
    assert UserAction.from_index(0) == "login"
    assert UserAction.from_index(7) == "admin"
    with pytest.raises(IndexError):
        UserAction.from_index(8)


def test_equality():
    import time
    ts = time.time()
    a1 = UserAction("read", "r", session_id="s", timestamp=ts)
    a2 = UserAction("read", "r", session_id="s", timestamp=ts)
    assert a1 == a2


def test_type_count():
    assert UserAction.type_count() == 8
