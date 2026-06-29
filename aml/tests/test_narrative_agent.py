from aml.agents.narrative_agent import NarrativeAgent


def test_narrative_template_generation():
    agent = NarrativeAgent()

    transactions = [
        {
            "transaction_id": "TXN_1",
            "sender_id": "S1",
            "receiver_id": "R1",
            "amount": 9500,
            "sender_country": "US",
            "receiver_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        },
        {
            "transaction_id": "TXN_2",
            "sender_id": "S1",
            "receiver_id": "R2",
            "amount": 9700,
            "sender_country": "US",
            "receiver_country": "US",
            "timestamp": "2024-01-01T12:00:00",
        },
    ]

    res = agent.process(
        {
            "subject_id": "S1",
            "transactions": transactions,
            "evidence": {},
            "typology": "structuring",
            "risk_score": 0.9,
        }
    )

    assert "narrative" in res
    assert "citations" in res
    assert isinstance(res["citations"], list)
    assert res.get("generation_method") == "template"
    assert res.get("citation_count", 0) == len(res["citations"])
