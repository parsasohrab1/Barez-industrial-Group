from tvdi_industrial.security.audit import AuditLog


def test_audit_hash_chain(tmp_path):
    log = AuditLog(str(tmp_path / "audit.sqlite"))
    log.log("alice", "predict", "TIRE-1", {"grade": "A"})
    log.log("bob", "reject", "TIRE-2", {"grade": "C"})
    assert log.verify_chain() is True
