from aml.agents.privacy_guard import PrivacyGuard


def test_privacy_guard_redacts_single():
    pg = PrivacyGuard()
    sample = {
        "note": "Contact: john.doe@example.com SSN: 123-45-6789 Card: 4532-1234-5678-9010",
        "sender": "USER_001",
    }

    res = pg.process(sample)
    assert "redacted_data" in res
    rd = res["redacted_data"]
    assert "john.doe@example.com" not in rd["note"]
    assert (
        "[REDACTED_EMAIL" in rd["note"]
        or "[REDACTED_SSN" in rd["note"]
        or "[REDACTED_CREDIT_CARD" in rd["note"]
    )

    report = pg.get_redaction_report()
    assert report["total_redactions"] > 0


def test_privacy_guard_redacts_list():
    pg = PrivacyGuard()
    samples = [
        {"note": "Foo 123-45-6789"},
        {"note": "Bar john@doe.com"},
    ]

    res = pg.process(samples)
    assert "redacted_data" in res
    assert isinstance(res["redacted_data"], list)
    assert res["redaction_applied"] is True
