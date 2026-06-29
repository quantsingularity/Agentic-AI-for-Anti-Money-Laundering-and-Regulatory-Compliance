"""Tests for the Agent-as-Judge (alert dismissal and evidence-based escalation)."""

from aml.agents.judge_agent import JudgeAgent


def test_high_confidence_alert_is_upheld():
    judge = JudgeAgent()
    txn = {"amount": 500.0, "sender_country": "US", "receiver_country": "GB"}
    # High model confidence is upheld even without corroborating evidence.
    assert judge.review(txn, model_probability=0.95) is True


def test_borderline_alert_without_evidence_is_dismissed():
    judge = JudgeAgent()
    txn = {"amount": 500.0, "sender_country": "US", "receiver_country": "GB"}
    assert judge.review(txn, model_probability=0.6) is False


def test_borderline_alert_with_high_risk_geography_is_upheld():
    judge = JudgeAgent()
    txn = {"amount": 500.0, "sender_country": "US", "receiver_country": "IR"}
    assert judge.review(txn, model_probability=0.6) is True


def test_structuring_band_counts_as_evidence():
    judge = JudgeAgent()
    txn = {"amount": 9500.0, "sender_country": "US", "receiver_country": "GB"}
    assert judge.review(txn, model_probability=0.6) is True


def test_large_amount_counts_as_evidence():
    judge = JudgeAgent()
    txn = {"amount": 75000.0, "sender_country": "US", "receiver_country": "GB"}
    assert judge.review(txn, model_probability=0.6) is True


def test_escalation_requires_red_flag():
    judge = JudgeAgent()
    # No red flag: an unflagged transaction is not escalated.
    plain = {"amount": 500.0, "sender_country": "US", "receiver_country": "GB"}
    assert judge.escalate(plain, model_probability=0.0) is False
    # With a red flag (high-risk jurisdiction): escalated.
    risky = {"amount": 500.0, "sender_country": "US", "receiver_country": "KP"}
    assert judge.escalate(risky, model_probability=0.0) is True


def test_escalation_probability_gate_when_enabled():
    judge = JudgeAgent(config={"escalation_min_probability": 0.05})
    risky = {"amount": 500.0, "sender_country": "US", "receiver_country": "KP"}
    # Below the configured gate: not escalated despite the red flag.
    assert judge.escalate(risky, model_probability=0.01) is False
    # At/above the gate: escalated.
    assert judge.escalate(risky, model_probability=0.10) is True


def test_process_returns_decision_dict():
    judge = JudgeAgent()
    txn = {"amount": 500.0, "sender_country": "US", "receiver_country": "GB"}
    result = judge.process({"transaction": txn, "model_probability": 0.6})
    assert result["approved"] is False
    assert "rejection_reason" in result
