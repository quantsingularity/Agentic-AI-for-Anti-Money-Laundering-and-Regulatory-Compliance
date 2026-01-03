"""
Privacy Guard Agent
Implements PII redaction and privacy safeguards before any LLM processing.
"""

import re
from typing import Dict, List, Any
from .base_agent import BaseAgent
from loguru import logger


class PrivacyGuard(BaseAgent):
    """
    Privacy Guard agent that redacts PII from transaction data.
    
    Implements:
    - Pattern-based PII detection (SSN, credit cards, emails, phones)
    - Deterministic redaction with consistent mapping
    - Audit trail of redacted fields
    """
    
    # PII patterns
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    ACCOUNT_PATTERN = r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b'  # IBAN-like
    
    def __init__(self, agent_id=None, config=None):
        super().__init__(agent_id, config)
        self.redaction_map = {}
        self.redaction_counter = {
            'ssn': 0,
            'credit_card': 0,
            'email': 0,
            'phone': 0,
            'account': 0,
            'name': 0
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Redact PII from input data.
        
        Args:
            input_data: Dictionary containing text fields to redact
            
        Returns:
            Dict with redacted data and redaction metadata
        """
        if isinstance(input_data, dict):
            redacted_data = self._redact_dict(input_data)
        elif isinstance(input_data, str):
            redacted_data = self._redact_text(input_data)
        elif isinstance(input_data, list):
            redacted_data = [self._redact_dict(item) if isinstance(item, dict) else self._redact_text(str(item)) 
                           for item in input_data]
        else:
            redacted_data = input_data
        
        return {
            "redacted_data": redacted_data,
            "redaction_count": sum(self.redaction_counter.values()),
            "redaction_types": dict(self.redaction_counter),
            "redaction_applied": sum(self.redaction_counter.values()) > 0
        }
    
    def _redact_dict(self, data: Dict) -> Dict:
        """Recursively redact PII in dictionary."""
        redacted = {}
        for key, value in data.items():
            if isinstance(value, str):
                redacted[key] = self._redact_text(value)
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [self._redact_dict(item) if isinstance(item, dict) else self._redact_text(str(item)) 
                               for item in value]
            else:
                redacted[key] = value
        return redacted
    
    def _redact_text(self, text: str) -> str:
        """Redact PII patterns from text."""
        if not isinstance(text, str):
            return text
        
        # SSN
        text = re.sub(self.SSN_PATTERN, lambda m: self._get_redaction('ssn', m.group()), text)
        
        # Credit Cards
        text = re.sub(self.CREDIT_CARD_PATTERN, lambda m: self._get_redaction('credit_card', m.group()), text)
        
        # Emails
        text = re.sub(self.EMAIL_PATTERN, lambda m: self._get_redaction('email', m.group()), text)
        
        # Phones
        text = re.sub(self.PHONE_PATTERN, lambda m: self._get_redaction('phone', m.group()), text)
        
        # Account numbers
        text = re.sub(self.ACCOUNT_PATTERN, lambda m: self._get_redaction('account', m.group()), text)
        
        return text
    
    def _get_redaction(self, pii_type: str, original: str) -> str:
        """
        Get consistent redaction token for PII.
        
        Args:
            pii_type: Type of PII (ssn, credit_card, etc.)
            original: Original PII value
            
        Returns:
            Redaction token
        """
        key = f"{pii_type}:{original}"
        
        if key not in self.redaction_map:
            self.redaction_counter[pii_type] += 1
            count = self.redaction_counter[pii_type]
            self.redaction_map[key] = f"[REDACTED_{pii_type.upper()}_{count}]"
        
        return self.redaction_map[key]
    
    def get_redaction_report(self) -> Dict[str, Any]:
        """
        Generate report of redactions applied.
        
        Returns:
            Dict with redaction statistics
        """
        return {
            "total_redactions": sum(self.redaction_counter.values()),
            "redaction_types": dict(self.redaction_counter),
            "unique_pii_values": len(self.redaction_map)
        }
