# Regulatory Compliance and Ethics Checklist

## Status: IMPLEMENTED WITH CODE-LEVEL SAFEGUARDS

## 1. Privacy & Data Protection

### 1.1 PII Redaction (IMPLEMENTED)
**Location**: `code/agents/privacy_guard.py`

**Implementation**:
- Deterministic regex patterns for SSN, credit cards, emails, phones, account numbers
- Pre-processing before any LLM calls
- Consistent redaction mapping (same PII → same redaction token)
- Audit trail of redacted fields

**Test**:
```python
from code.agents.privacy_guard import PrivacyGuard

guard = PrivacyGuard()
result = guard.process({
    'text': 'Contact john@example.com, SSN: 123-45-6789'
})
assert '[REDACTED_EMAIL' in str(result)
assert '[REDACTED_SSN' in str(result)
```

**Compliance**: GDPR Art. 5(1)(c) - Data minimization

### 1.2 Data Retention Policy
**Recommendation**: 
- Transaction logs: 5 years (BSA requirement)
- PII redacted: Delete after SAR filing
- Audit logs: 7 years (regulatory examination)

**Implementation Required**: Database TTL configuration (not in research code)

## 2. Regulatory Alignment

### 2.1 FATF Recommendations

**Recommendation 10** (Customer Due Diligence):
- System supports risk scoring
- ✓ Enhanced due diligence for high-risk cases (risk_score > 0.9)

**Recommendation 11** (Record Keeping):
- ✓ Complete audit trail in JSONL format
- ✓ Transaction-level provenance

**Recommendation 20** (Suspicious Transaction Reporting):
- ✓ SAR narratives include required elements
- ✓ Evidence linking to source transactions

**Recommendation 19** (Higher-Risk Countries):
- ✓ Geographic risk detection
- ✓ Enhanced review for high-risk jurisdictions

### 2.2 Bank Secrecy Act (US)

**31 U.S.C. § 5318(g)** - SAR Filing Requirements:
- ✓ Narrative describes suspicious activity
- ✓ Supporting documentation maintained
- ✓ Confidentiality (system logs secure)

**31 CFR § 1020.320** - SAR Filing Procedures:
- ✓ Within 30 days of detection (system supports)
- ✓ No tipping off subject (no user notifications)

### 2.3 GDPR (EU)

**Article 5** - Data Processing Principles:
- ✓ Lawfulness: Legitimate interest (AML obligation)
- ✓ Purpose limitation: Only AML detection
- ✓ Data minimization: PII redacted
- ✓ Accuracy: Validation layers

**Article 22** - Automated Decision-Making:
- ✓ Human-in-loop for high-risk cases
- ✓ Right to explanation (audit logs)

### 2.4 PCI DSS

**Requirement 3** - Protect Stored Cardholder Data:
- ✓ No storage of full card numbers (redacted)
- ✓ Masking when displayed

## 3. Code-Level Safeguards (IMPLEMENTED)

### 3.1 Investigator Gating
**Location**: `code/agents/orchestrator.py` lines 75-85

```python
if risk_score >= self.high_risk_threshold:  # Default 0.9
    requires_human_review = True
```

**Effect**: High-severity SARs cannot be auto-filed

### 3.2 Entity Throttling
**Location**: `code/agents/orchestrator.py` lines 70-75

```python
if current_count >= self.max_sar_per_entity:  # Default 10/month
    logger.warning("Entity exceeded SAR limit. Throttling.")
    return None
```

**Effect**: Prevents SAR bombing attacks

### 3.3 Audit Logging
**Location**: `code/agents/base_agent.py` lines 90-120

Every agent action logged:
```json
{
  "agent_id": "narrative_agent_001",
  "execution_id": "exec_abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "input": {...},
  "output": {...},
  "citations": [...]
}
```

**Effect**: Full replay capability for audits

### 3.4 Citation Validation
**Location**: `code/agents/narrative_agent.py` lines 200-225

Every claim must cite source transaction:
```python
[CITE: TXN_00001234:amount]
```

Validation rejects narratives with uncited claims.

**Effect**: No hallucinations without evidence link

## 4. Human Oversight Requirements

### 4.1 Roles & Responsibilities

**Compliance Officer**:
- Reviews high-risk SARs (score > 0.9)
- Final filing decision authority
- Feedback for system improvement

**System Operator**:
- Monitors system health
- Investigates anomalies
- Does NOT override compliance decisions

**Auditor**:
- Reviews audit logs
- Validates citation accuracy
- Assesses false negative risk

### 4.2 Override Mechanism

Investigators can:
- ✓ Reject system-generated SARs
- ✓ Request regeneration with different parameters
- ✓ Manually edit narratives (logged)
- ✓ Add additional evidence

System cannot:
- ❌ Auto-file without human approval (high-risk cases)
- ❌ Suppress alerts without investigator decision
- ❌ Modify audit logs

## 5. Bias & Fairness

### 5.1 Protected Attributes

System does NOT use:
- Race, ethnicity
- Religion
- Gender
- Age
- Nationality (except for sanctions compliance)

Geography used ONLY for:
- FATF high-risk jurisdictions
- OFAC sanctioned countries

### 5.2 Fairness Auditing (Recommended)

Monitor false positive rates by:
- Transaction type
- Amount range
- Geographic region

Flag if disparate impact detected.

### 5.3 Adverse Action Notices

If system contributes to account closure:
- Customer entitled to explanation
- System provides: risk factors, transaction IDs
- Human investigator makes final decision

## 6. Security

### 6.1 Access Control (Deployment Requirement)

- **Authentication**: MFA for compliance officers
- **Authorization**: Role-based access (read-only, file, admin)
- **Encryption**: TLS for transit, AES-256 for rest

### 6.2 Audit Log Protection

- Immutable: Write-only logs (no deletion)
- Integrity: Cryptographic hashing
- Retention: 7 years minimum

### 6.3 Secrets Management

- No hardcoded credentials ✓
- Environment variables for API keys ✓
- Secrets rotation policy (deployment)

## 7. Adversarial Considerations

### 7.1 Evasion Attacks

**Threat**: Criminals adapt to avoid detection

**Mitigation**:
- Ensemble methods (multiple detectors)
- Regular model retraining
- Adversarial training (future work)
- Red team testing

### 7.2 Poisoning Attacks

**Threat**: Malicious feedback to degrade model

**Mitigation**:
- Human validation before label update
- Anomaly detection on feedback
- Periodic baseline validation

### 7.3 Model Inversion

**Threat**: Reverse-engineer training data

**Mitigation**:
- Differential privacy (future)
- Aggregated statistics only
- No raw transaction exports

## 8. Transparency & Explainability

### 8.1 System Documentation

- ✓ Architecture diagram (Figure 1)
- ✓ Agent specifications (Section 4)
- ✓ Feature engineering (XGBoost classifier)
- ✓ Prompt templates (Appendix B)

### 8.2 Decision Provenance

For each SAR:
- Transaction IDs with citations
- Risk score calculation
- Agent execution trace
- Investigator review notes

### 8.3 Model Cards

Provide for each model:
- Training data characteristics
- Performance metrics by typology
- Known limitations
- Recommended use cases

## 9. Regulatory Approval Process

### 9.1 Pre-Deployment

1. **Internal Review**:
   - Legal team approval
   - Risk committee approval
   - IT security review

2. **Regulator Consultation**:
   - FinCEN (US) / FIU (jurisdiction)
   - Demonstrate compliance with BSA/AML
   - Provide audit logs and documentation

3. **Pilot Study**:
   - Parallel run with existing system
   - Human review of all system SARs
   - Measure false positive reduction
   - Document errors and fixes

### 9.2 Ongoing Monitoring

- Monthly: False positive/negative rates
- Quarterly: Model performance audit
- Annual: Full regulatory examination

## 10. Incident Response

### 10.1 System Errors

If system generates incorrect SAR:
1. Investigator identifies error
2. SAR retracted (if filed)
3. Root cause analysis
4. Model retraining if needed
5. Regulator notification (if material)

### 10.2 Data Breach

If PII exposed:
- Immediate system shutdown
- Forensic investigation
- Customer notification (GDPR 72 hours)
- Regulatory filing

### 10.3 Kill Switch

Emergency stop:
```bash
export SYSTEM_KILL_SWITCH=true
```

Effect: All agents halt, no new SARs generated, existing work preserved.

## 11. Ethical Considerations

### 11.1 Job Displacement

System augments, not replaces:
- Compliance officers review all high-risk SARs
- System handles high-volume low-risk alerts
- Investigators focus on complex cases

### 11.2 Accountability

- System recommendations, not decisions
- Human compliance officer accountable
- Audit trail enables responsibility tracing

### 11.3 Transparency to Regulators

- Full system documentation provided
- Audit logs available for examination
- Model updates disclosed

## 12. Checklist Summary

### Implemented ✓
- [x] PII redaction code
- [x] Audit logging
- [x] Evidence citation validation
- [x] High-risk gating
- [x] Entity throttling
- [x] Deterministic seeds (reproducibility)

### Deployment Required
- [ ] Access control (MFA, RBAC)
- [ ] Secrets management
- [ ] Database encryption
- [ ] Regulatory approval

### Ongoing
- [ ] Bias monitoring
- [ ] Performance audits
- [ ] Model retraining
- [ ] Red team testing

---

**Last Updated**: 2026-01-01
**Reviewed By**: Research Team
**Next Review**: Deployment planning phase
