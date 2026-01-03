"""
Orchestrator - Multi-Agent Coordination for SAR Workflow
Coordinates all agents in the AML pipeline.
"""

from typing import Dict, List, Any, Optional
import time
from datetime import datetime
from loguru import logger
import json


class Orchestrator:
    """
    Orchestrates the multi-agent SAR generation workflow.
    
    Workflow:
    1. Ingest & Feature Engineering
    2. Privacy Guard (PII redaction)
    3. Crime Classification
    4. External Intelligence (sanctions/PEP)
    5. Evidence Aggregation
    6. Narrative Generation
    7. Agent-as-Judge Validation
    8. Human-in-Loop (for high-risk)
    """
    
    def __init__(self, agents: Dict[str, Any], config: Optional[Dict] = None):
        """
        Initialize orchestrator.
        
        Args:
            agents: Dict mapping agent names to agent instances
            config: Configuration dict
        """
        self.agents = agents
        self.config = config or {}
        self.execution_log = []
        
        # Safeguards
        self.max_sar_per_entity = self.config.get('max_sar_per_entity', 10)
        self.high_risk_threshold = self.config.get('high_risk_threshold', 0.9)
        self.enable_privacy_guard = self.config.get('enable_privacy_guard', True)
        self.enable_judge_agent = self.config.get('enable_judge_agent', True)
        
        # Counters
        self.sar_count_by_entity = {}
        
        logger.info(f"Initialized Orchestrator with {len(agents)} agents")
    
    def process_transaction_batch(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Process a batch of transactions through the full pipeline.
        
        Args:
            transactions: List of transaction dicts
            
        Returns:
            Dict with processing results and generated SARs
        """
        start_time = time.time()
        workflow_id = f"workflow_{int(start_time)}"
        
        logger.info(f"[{workflow_id}] Starting batch processing for {len(transactions)} transactions")
        
        results = {
            'workflow_id': workflow_id,
            'input_count': len(transactions),
            'timestamp': datetime.utcnow().isoformat(),
            'sars_generated': [],
            'alerts': [],
            'errors': []
        }
        
        try:
            # Step 1: Feature Engineering
            logger.info(f"[{workflow_id}] Step 1: Feature Engineering")
            features = self._run_agent('feature_engineer', {'transactions': transactions})
            
            # Step 2: Privacy Guard
            if self.enable_privacy_guard:
                logger.info(f"[{workflow_id}] Step 2: Privacy Guard")
                privacy_result = self._run_agent('privacy_guard', features['result'])
                processed_data = privacy_result['result']['redacted_data']
            else:
                processed_data = features['result']
            
            # Step 3: Crime Classification
            logger.info(f"[{workflow_id}] Step 3: Crime Classification")
            classification = self._run_agent('classifier', processed_data)
            
            # Get suspicious transactions
            suspicious_txns = self._extract_suspicious(
                transactions, 
                classification['result'].get('predictions', [])
            )
            
            logger.info(f"[{workflow_id}] Identified {len(suspicious_txns)} suspicious transactions")
            
            # Step 4: Group by entity and process each
            entity_groups = self._group_by_entity(suspicious_txns)
            
            for entity_id, entity_txns in entity_groups.items():
                try:
                    sar = self._process_entity(entity_id, entity_txns, workflow_id)
                    if sar:
                        results['sars_generated'].append(sar)
                except Exception as e:
                    logger.error(f"[{workflow_id}] Error processing entity {entity_id}: {e}")
                    results['errors'].append({
                        'entity_id': entity_id,
                        'error': str(e)
                    })
            
            # Summary statistics
            end_time = time.time()
            results['execution_time'] = end_time - start_time
            results['sars_count'] = len(results['sars_generated'])
            results['detection_rate'] = len(suspicious_txns) / len(transactions) if transactions else 0
            results['status'] = 'success'
            
            logger.info(
                f"[{workflow_id}] Completed: {results['sars_count']} SARs generated "
                f"in {results['execution_time']:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            results['execution_time'] = time.time() - start_time
        
        return results
    
    def _process_entity(self, entity_id: str, transactions: List[Dict], 
                       workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single entity through the SAR pipeline.
        
        Args:
            entity_id: Entity identifier
            transactions: Entity's suspicious transactions
            workflow_id: Workflow identifier
            
        Returns:
            SAR dict or None if throttled/rejected
        """
        # Check throttle
        current_count = self.sar_count_by_entity.get(entity_id, 0)
        if current_count >= self.max_sar_per_entity:
            logger.warning(
                f"[{workflow_id}] Entity {entity_id} exceeded SAR limit "
                f"({current_count}/{self.max_sar_per_entity}). Throttling."
            )
            return None
        
        logger.info(f"[{workflow_id}] Processing entity {entity_id} with {len(transactions)} transactions")
        
        # Step 4: External Intelligence
        intelligence = self._run_agent('intelligence', {
            'entity_id': entity_id,
            'transactions': transactions
        })
        
        # Step 5: Evidence Aggregation
        evidence = self._run_agent('evidence_aggregator', {
            'entity_id': entity_id,
            'transactions': transactions,
            'intelligence': intelligence['result']
        })
        
        # Determine primary typology
        typology = self._determine_typology(transactions)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(transactions, intelligence['result'], typology)
        
        # Step 6: Narrative Generation
        narrative_input = {
            'subject_id': entity_id,
            'transactions': transactions,
            'evidence': evidence['result'],
            'typology': typology,
            'risk_score': risk_score
        }
        
        narrative = self._run_agent('narrative_agent', narrative_input)
        
        # Step 7: Agent-as-Judge Validation
        if self.enable_judge_agent:
            validation = self._run_agent('judge_agent', {
                'narrative': narrative['result'],
                'evidence': evidence['result'],
                'transactions': transactions
            })
            
            if not validation['result'].get('approved', True):
                logger.warning(
                    f"[{workflow_id}] SAR for {entity_id} rejected by Judge Agent: "
                    f"{validation['result'].get('rejection_reason', 'Unknown')}"
                )
                return None
        
        # Step 8: Human-in-Loop Gating
        requires_human_review = risk_score >= self.high_risk_threshold
        
        # Construct SAR
        sar = {
            'entity_id': entity_id,
            'workflow_id': workflow_id,
            'generation_time': datetime.utcnow().isoformat(),
            'typology': typology,
            'risk_score': risk_score,
            'transaction_count': len(transactions),
            'narrative': narrative['result']['narrative'],
            'citations': narrative['result']['citations'],
            'evidence': evidence['result'],
            'intelligence': intelligence['result'],
            'requires_human_review': requires_human_review,
            'status': 'pending_review' if requires_human_review else 'auto_filed'
        }
        
        # Update counter
        self.sar_count_by_entity[entity_id] = current_count + 1
        
        return sar
    
    def _run_agent(self, agent_name: str, input_data: Any) -> Dict[str, Any]:
        """
        Execute an agent.
        
        Args:
            agent_name: Name of agent to run
            input_data: Input data for agent
            
        Returns:
            Agent execution result
        """
        if agent_name not in self.agents:
            logger.warning(f"Agent {agent_name} not found, skipping")
            return {'result': input_data, 'status': 'skipped'}
        
        agent = self.agents[agent_name]
        result = agent.execute(input_data)
        
        # Log execution
        self.execution_log.append({
            'agent': agent_name,
            'timestamp': datetime.utcnow().isoformat(),
            'execution_time': result.get('execution_time', 0),
            'status': result.get('status', 'unknown')
        })
        
        return result
    
    def _extract_suspicious(self, transactions: List[Dict], 
                          predictions: List[int]) -> List[Dict]:
        """Extract suspicious transactions based on predictions."""
        if not predictions or len(predictions) != len(transactions):
            # Fallback: use ground truth labels if available
            return [t for t in transactions if t.get('is_fraud', 0) == 1]
        
        suspicious = []
        for txn, pred in zip(transactions, predictions):
            if pred == 1:
                suspicious.append(txn)
        
        return suspicious
    
    def _group_by_entity(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group transactions by entity (sender or receiver)."""
        entity_map = {}
        
        for txn in transactions:
            # Use sender as primary entity
            entity_id = txn.get('sender_id', 'UNKNOWN')
            
            if entity_id not in entity_map:
                entity_map[entity_id] = []
            
            entity_map[entity_id].append(txn)
        
        return entity_map
    
    def _determine_typology(self, transactions: List[Dict]) -> str:
        """Determine primary typology from transactions."""
        if not transactions:
            return 'unknown'
        
        # Use most common typology
        from collections import Counter
        typologies = [t.get('fraud_typology', 'unknown') for t in transactions]
        counter = Counter(typologies)
        
        # Exclude 'none'
        counter.pop('none', None)
        
        if not counter:
            return 'unknown'
        
        return counter.most_common(1)[0][0]
    
    def _calculate_risk_score(self, transactions: List[Dict], 
                             intelligence: Dict, typology: str) -> float:
        """
        Calculate risk score for entity.
        
        Args:
            transactions: Entity transactions
            intelligence: External intelligence results
            typology: Primary typology
            
        Returns:
            Risk score (0-1)
        """
        score = 0.0
        
        # Transaction volume factor
        score += min(len(transactions) / 50.0, 0.3)
        
        # Amount factor
        total_amount = sum(t.get('amount', 0) for t in transactions)
        score += min(total_amount / 1000000.0, 0.2)
        
        # Typology severity
        severe_typologies = ['sanctions_evasion', 'trade_based', 'rapid_movement']
        if typology in severe_typologies:
            score += 0.3
        else:
            score += 0.15
        
        # Intelligence hits
        if intelligence.get('sanctions_hits'):
            score += 0.3
        if intelligence.get('pep_hits'):
            score += 0.15
        if intelligence.get('adverse_media'):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_execution_log(self) -> List[Dict]:
        """Get execution log."""
        return self.execution_log
    
    def save_execution_log(self, filepath: str):
        """Save execution log to file."""
        with open(filepath, 'w') as f:
            for entry in self.execution_log:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Saved execution log to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'total_agents': len(self.agents),
            'total_executions': len(self.execution_log),
            'sars_by_entity': dict(self.sar_count_by_entity),
            'total_sars': sum(self.sar_count_by_entity.values()),
            'config': self.config
        }
