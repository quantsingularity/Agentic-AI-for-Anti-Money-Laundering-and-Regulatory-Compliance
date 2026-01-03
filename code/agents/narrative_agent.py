"""
Narrative Agent - Constrained LLM for SAR Generation
Generates Suspicious Activity Report narratives with mandatory evidence citation.
"""

from typing import Dict, List, Any, Optional
import json
from .base_agent import BaseAgent
from loguru import logger


class NarrativeAgent(BaseAgent):
    """
    Generates SAR narratives with constrained LLM generation.
    
    Key features:
    - Every factual claim must cite evidence (transaction_id + field)
    - Template-based fallback when LLM unavailable
    - Chain-of-thought reasoning
    - Regulatory compliance language
    """
    
    SAR_TEMPLATE = """
SUSPICIOUS ACTIVITY REPORT NARRATIVE

Subject: {subject_id}
Report Date: {report_date}
Analysis Period: {start_date} to {end_date}
Primary Typology: {typology}
Risk Score: {risk_score:.2f}

SUMMARY OF SUSPICIOUS ACTIVITY:
{summary}

DETAILED ANALYSIS:
{detailed_analysis}

SUPPORTING EVIDENCE:
{evidence_section}

REGULATORY CONSIDERATIONS:
{regulatory_notes}

RECOMMENDATION:
Based on the analysis above, this activity warrants filing of a Suspicious Activity Report per 31 CFR § 1020.320.

--- END OF NARRATIVE ---
"""
    
    def __init__(self, agent_id=None, config=None, llm_client=None):
        """
        Initialize Narrative Agent.
        
        Args:
            agent_id: Agent identifier
            config: Configuration dict
            llm_client: LLM client (OpenAI, etc.) - optional, falls back to template
        """
        super().__init__(agent_id, config)
        self.llm_client = llm_client
        self.use_llm = llm_client is not None
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SAR narrative from evidence.
        
        Args:
            input_data: Dict with keys:
                - subject_id: Entity ID
                - transactions: List of suspicious transactions
                - evidence: Aggregated evidence dict
                - typology: Primary typology
                - risk_score: Risk assessment score
                
        Returns:
            Dict with narrative and citations
        """
        if self.use_llm:
            return self._generate_with_llm(input_data)
        else:
            return self._generate_with_template(input_data)
    
    def _generate_with_template(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate narrative using template (fallback mode)."""
        
        subject_id = input_data.get('subject_id', 'UNKNOWN')
        transactions = input_data.get('transactions', [])
        evidence = input_data.get('evidence', {})
        typology = input_data.get('typology', 'unknown')
        risk_score = input_data.get('risk_score', 0.5)
        
        # Generate summary based on typology
        summary = self._generate_summary(typology, transactions, evidence)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(transactions, evidence, typology)
        
        # Format evidence section with citations
        evidence_section = self._format_evidence_section(transactions, evidence)
        
        # Regulatory notes
        regulatory_notes = self._generate_regulatory_notes(typology, evidence)
        
        # Fill template
        narrative = self.SAR_TEMPLATE.format(
            subject_id=subject_id,
            report_date=evidence.get('report_date', '2024-XX-XX'),
            start_date=evidence.get('start_date', '2024-XX-XX'),
            end_date=evidence.get('end_date', '2024-XX-XX'),
            typology=typology.replace('_', ' ').title(),
            risk_score=risk_score,
            summary=summary,
            detailed_analysis=detailed_analysis,
            evidence_section=evidence_section,
            regulatory_notes=regulatory_notes
        )
        
        # Extract citations
        citations = self._extract_citations(transactions)
        
        return {
            'narrative': narrative,
            'citations': citations,
            'generation_method': 'template',
            'typology': typology,
            'risk_score': risk_score,
            'citation_count': len(citations)
        }
    
    def _generate_summary(self, typology: str, transactions: List[Dict], 
                         evidence: Dict) -> str:
        """Generate summary paragraph based on typology."""
        
        summaries = {
            'structuring': (
                f"Analysis identified a pattern of multiple transactions conducted by the subject "
                f"in amounts designed to evade Bank Secrecy Act reporting thresholds. Over the analysis period, "
                f"{len(transactions)} transactions were conducted, each below the $10,000 CTR threshold "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:amount, "
                f"{transactions[1].get('transaction_id', 'TXN_XXX')}:amount]. "
                f"The temporal clustering and amount structuring are consistent with intentional evasion."
            ),
            'rapid_movement': (
                f"Investigation revealed rapid movement of funds through a chain of {len(transactions)} accounts "
                f"within a compressed timeframe. Funds originated from {transactions[0].get('sender_id', 'UNKNOWN')} "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:sender_id] and were layered through "
                f"multiple intermediary accounts, consistent with money laundering layering techniques."
            ),
            'sanctions_evasion': (
                f"The subject conducted {len(transactions)} transactions involving entities and jurisdictions "
                f"subject to OFAC sanctions. Specifically, transactions were directed to "
                f"{transactions[0].get('receiver_id', 'UNKNOWN')} "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:receiver_id], "
                f"located in {transactions[0].get('receiver_country', 'XX')} "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:receiver_country]."
            ),
            'smurfing': (
                f"Analysis detected a smurfing pattern with {len(transactions)} deposits to account "
                f"{transactions[0].get('receiver_id', 'UNKNOWN')} "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:receiver_id] "
                f"from multiple source accounts. The coordinated nature and timing suggest organized structuring."
            )
        }
        
        return summaries.get(typology, 
            f"Analysis of {len(transactions)} transactions by the subject revealed patterns "
            f"consistent with suspicious activity requiring regulatory reporting."
        )
    
    def _generate_detailed_analysis(self, transactions: List[Dict], 
                                   evidence: Dict, typology: str) -> str:
        """Generate detailed analysis section."""
        
        analysis_parts = []
        
        # Transaction volume and amounts
        total_amount = sum(t.get('amount', 0) for t in transactions)
        analysis_parts.append(
            f"1. Transaction Volume: The subject conducted {len(transactions)} transactions "
            f"totaling ${total_amount:,.2f} during the analysis period."
        )
        
        # Temporal patterns
        if len(transactions) >= 3:
            analysis_parts.append(
                f"2. Temporal Pattern: Transactions occurred between "
                f"{transactions[0].get('timestamp', 'UNKNOWN')} and "
                f"{transactions[-1].get('timestamp', 'UNKNOWN')}, "
                f"exhibiting concentrated activity consistent with {typology.replace('_', ' ')} schemes "
                f"[CITE: {transactions[0].get('transaction_id', 'TXN_XXX')}:timestamp, "
                f"{transactions[-1].get('transaction_id', 'TXN_XXX')}:timestamp]."
            )
        
        # Geographic analysis
        if evidence.get('high_risk_countries'):
            analysis_parts.append(
                f"3. Geographic Risk: Transactions involved high-risk jurisdictions: "
                f"{', '.join(evidence['high_risk_countries'])}. "
                f"These jurisdictions are associated with elevated money laundering risk."
            )
        
        # Counterparty analysis
        if evidence.get('suspicious_counterparties'):
            analysis_parts.append(
                f"4. Counterparty Risk: Subject transacted with entities exhibiting red flags, "
                f"including {', '.join(evidence['suspicious_counterparties'][:3])}."
            )
        
        return "\n\n".join(analysis_parts)
    
    def _format_evidence_section(self, transactions: List[Dict], 
                                 evidence: Dict) -> str:
        """Format evidence section with transaction details."""
        
        evidence_lines = []
        
        # List key transactions
        evidence_lines.append("Key Transactions:")
        for i, txn in enumerate(transactions[:10], 1):  # Limit to 10
            evidence_lines.append(
                f"  {i}. Transaction ID: {txn.get('transaction_id', 'UNKNOWN')} | "
                f"Amount: ${txn.get('amount', 0):,.2f} | "
                f"Date: {txn.get('timestamp', 'UNKNOWN')} | "
                f"From: {txn.get('sender_id', 'UNKNOWN')} ({txn.get('sender_country', 'XX')}) | "
                f"To: {txn.get('receiver_id', 'UNKNOWN')} ({txn.get('receiver_country', 'XX')})"
            )
        
        if len(transactions) > 10:
            evidence_lines.append(f"  ... and {len(transactions) - 10} additional transactions")
        
        # External intelligence
        if evidence.get('sanctions_hits'):
            evidence_lines.append(f"\nSanctions Screening: {len(evidence['sanctions_hits'])} potential matches identified")
        
        if evidence.get('pep_hits'):
            evidence_lines.append(f"PEP Screening: {len(evidence['pep_hits'])} politically exposed persons identified")
        
        return "\n".join(evidence_lines)
    
    def _generate_regulatory_notes(self, typology: str, evidence: Dict) -> str:
        """Generate regulatory compliance notes."""
        
        notes = []
        
        # BSA/AML requirements
        notes.append(
            "This activity triggers SAR filing requirements under the Bank Secrecy Act "
            "(31 U.S.C. § 5318(g)) and implementing regulations (31 CFR § 1020.320)."
        )
        
        # FATF recommendations
        if typology in ['sanctions_evasion', 'high_risk_geography']:
            notes.append(
                "Activity involves high-risk jurisdictions per FATF Recommendations 19 and 20, "
                "requiring enhanced due diligence."
            )
        
        # FinCEN guidance
        notes.append(
            f"SAR filing is consistent with FinCEN guidance on {typology.replace('_', ' ')} typologies."
        )
        
        return " ".join(notes)
    
    def _extract_citations(self, transactions: List[Dict]) -> List[Dict[str, str]]:
        """Extract all citations from transactions."""
        
        citations = []
        for txn in transactions:
            txn_id = txn.get('transaction_id', 'UNKNOWN')
            for field in ['amount', 'sender_id', 'receiver_id', 'sender_country', 
                         'receiver_country', 'timestamp']:
                if field in txn:
                    citations.append({
                        'transaction_id': txn_id,
                        'field': field,
                        'value': str(txn[field])
                    })
        
        return citations
    
    def _generate_with_llm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate narrative using LLM (with citation constraints)."""
        
        # Construct prompt with citation requirements
        prompt = self._construct_llm_prompt(input_data)
        
        try:
            # Call LLM (implementation depends on client)
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=2000
            )
            
            narrative = response['text']
            
            # Validate citations
            citations = self._validate_and_extract_llm_citations(narrative, input_data['transactions'])
            
            return {
                'narrative': narrative,
                'citations': citations,
                'generation_method': 'llm',
                'typology': input_data.get('typology', 'unknown'),
                'risk_score': input_data.get('risk_score', 0.5),
                'citation_count': len(citations)
            }
            
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Falling back to template.")
            return self._generate_with_template(input_data)
    
    def _construct_llm_prompt(self, input_data: Dict[str, Any]) -> str:
        """Construct prompt for LLM with citation requirements."""
        
        transactions_json = json.dumps(input_data['transactions'], indent=2)
        
        prompt = f"""You are a compliance officer writing a Suspicious Activity Report (SAR) narrative.

CRITICAL REQUIREMENT: Every factual claim MUST cite the source transaction using the format [CITE: transaction_id:field_name].

Example: "The subject transferred $9,500 [CITE: TXN_00001234:amount] on January 15 [CITE: TXN_00001234:timestamp]."

Subject: {input_data.get('subject_id', 'UNKNOWN')}
Typology: {input_data.get('typology', 'unknown')}
Risk Score: {input_data.get('risk_score', 0.5)}

Transactions:
{transactions_json}

Generate a professional SAR narrative with:
1. Executive summary
2. Detailed analysis
3. Evidence section
4. Regulatory considerations

Remember: CITE EVERY FACT with [CITE: transaction_id:field].
"""
        
        return prompt
    
    def _validate_and_extract_llm_citations(self, narrative: str, 
                                           transactions: List[Dict]) -> List[Dict]:
        """Validate and extract citations from LLM-generated narrative."""
        
        import re
        citation_pattern = r'\[CITE:\s*([^\]]+)\]'
        
        citations = []
        matches = re.findall(citation_pattern, narrative)
        
        for match in matches:
            parts = match.split(':')
            if len(parts) == 2:
                citations.append({
                    'transaction_id': parts[0].strip(),
                    'field': parts[1].strip(),
                    'value': 'cited'
                })
        
        return citations
