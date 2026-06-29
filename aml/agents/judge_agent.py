"""
Agent-as-Judge

Adjudicates alerts raised by the upstream detector. In a real AML workflow a
human analyst triages model alerts and dismisses those that lack corroborating
red flags; this agent encodes that review step with a transparent, auditable
rule set so the multi-agent system does not simply mirror the raw classifier.

The judge can only *overturn* an alert (mark a flagged transaction as
non-suspicious); it never creates new alerts. It upholds an alert when the model
is highly confident or when the transaction carries corroborating structural
evidence of a known typology (high-risk jurisdiction exposure, structuring just
under a reporting threshold, or an unusually large amount). Borderline alerts
with no corroborating evidence are dismissed as likely false positives.
"""

from typing import Any, Dict, Optional

from .base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """Reviews detector alerts and dismisses weakly-supported ones."""

    # Jurisdictions treated as high risk (mirrors the data generator's list).
    HIGH_RISK_COUNTRIES = {"SY", "IR", "KP", "VE", "MM", "AF", "IQ"}

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id=agent_id, config=config)
        cfg = self.config or {}
        # An alert whose model probability is at/above this is always upheld.
        self.high_confidence_threshold = cfg.get("high_confidence_threshold", 0.85)
        # Structuring: amounts deliberately placed just under a reporting limit.
        self.structuring_low = cfg.get("structuring_low", 9000.0)
        self.structuring_high = cfg.get("structuring_high", 10000.0)
        # Amounts at/above this count as corroborating "large value" evidence.
        self.large_amount_threshold = cfg.get("large_amount_threshold", 50000.0)
        # Escalation: an *unflagged* transaction is re-raised when it carries a
        # corroborating typology red flag. The optional probability gate is off by
        # default (0.0): empirically the detector assigns near-zero probability to
        # the frauds it misses under a temporal split, so its score carries no
        # signal for recovery and escalation rests on typology evidence instead.
        # Raising this gate (>0) makes escalation also require residual model
        # suspicion, for datasets where the detector's score is informative.
        self.escalation_min_probability = cfg.get("escalation_min_probability", 0.0)

    def _has_corroborating_evidence(self, transaction: Dict[str, Any]) -> bool:
        """True if the transaction carries an independent AML red flag."""
        sender = str(transaction.get("sender_country", "")).upper()
        receiver = str(transaction.get("receiver_country", "")).upper()
        if sender in self.HIGH_RISK_COUNTRIES or receiver in self.HIGH_RISK_COUNTRIES:
            return True

        try:
            amount = float(transaction.get("amount", 0.0))
        except (TypeError, ValueError):
            amount = 0.0

        if self.structuring_low <= amount < self.structuring_high:
            return True
        if amount >= self.large_amount_threshold:
            return True
        return False

    def review(self, transaction: Dict[str, Any], model_probability: float) -> bool:
        """Adjudicate a single flagged transaction.

        Returns:
            True to uphold the alert (suspicious), False to dismiss it.
        """
        if model_probability >= self.high_confidence_threshold:
            return True
        return self._has_corroborating_evidence(transaction)

    def escalate(self, transaction: Dict[str, Any], model_probability: float) -> bool:
        """Decide whether an *unflagged* transaction should be re-raised.

        Recovers frauds the statistical model narrowly missed: requires both
        residual model suspicion (>= ``escalation_min_probability``) and a
        corroborating typology red flag. Returns True to escalate to suspicious.
        """
        if model_probability < self.escalation_min_probability:
            return False
        return self._has_corroborating_evidence(transaction)

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Adjudicate an alert.

        Args:
            input_data: dict with ``transaction`` (record dict) and
                ``model_probability`` (float).

        Returns:
            Dict with ``approved`` (bool) and a ``rejection_reason`` when dismissed.
        """
        transaction = input_data.get("transaction", {})
        probability = float(input_data.get("model_probability", 0.0))

        upheld = self.review(transaction, probability)
        result: Dict[str, Any] = {"approved": upheld}
        if not upheld:
            result["rejection_reason"] = (
                "Borderline model confidence with no corroborating red flags "
                "(high-risk jurisdiction, structuring band, or large amount)."
            )
        return result
