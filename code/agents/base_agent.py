"""
Base Agent Abstract Class
All agents inherit from this base class for consistent interface and logging.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import time
from datetime import datetime
from loguru import logger
import uuid


class BaseAgent(ABC):
    """Abstract base class for all agents in the AML system."""
    
    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            config: Configuration dictionary
        """
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        self.audit_log = []
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {self.agent_id}")
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by subclasses.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dict containing processing results and metadata
        """
        pass
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute agent with logging and error handling.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dict containing results, metadata, and audit trail
        """
        self.start_time = time.time()
        execution_id = uuid.uuid4().hex
        
        try:
            logger.info(f"[{self.agent_id}] Starting execution {execution_id}")
            
            # Log input
            self._log_event("input", input_data, execution_id)
            
            # Process
            result = self.process(input_data)
            
            # Log output
            self._log_event("output", result, execution_id)
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            logger.info(f"[{self.agent_id}] Completed execution {execution_id} in {execution_time:.2f}s")
            
            return {
                "agent_id": self.agent_id,
                "agent_class": self.__class__.__name__,
                "execution_id": execution_id,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time if self.start_time else 0
            
            logger.error(f"[{self.agent_id}] Execution {execution_id} failed: {str(e)}")
            self._log_event("error", {"error": str(e), "type": type(e).__name__}, execution_id)
            
            return {
                "agent_id": self.agent_id,
                "agent_class": self.__class__.__name__,
                "execution_id": execution_id,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error"
            }
    
    def _log_event(self, event_type: str, data: Any, execution_id: str):
        """
        Log an event to the audit trail.
        
        Args:
            event_type: Type of event (input, output, error, etc.)
            data: Event data
            execution_id: Execution identifier
        """
        event = {
            "agent_id": self.agent_id,
            "agent_class": self.__class__.__name__,
            "execution_id": execution_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": self._serialize_data(data)
        }
        self.audit_log.append(event)
    
    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data for JSON logging.
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON-serializable representation
        """
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        else:
            return str(data)
    
    def get_audit_log(self) -> List[Dict]:
        """
        Retrieve the audit log for this agent.
        
        Returns:
            List of audit log events
        """
        return self.audit_log
    
    def save_audit_log(self, filepath: str):
        """
        Save audit log to file.
        
        Args:
            filepath: Path to save the audit log
        """
        with open(filepath, 'w') as f:
            for event in self.audit_log:
                f.write(json.dumps(event) + '\n')
        logger.info(f"[{self.agent_id}] Saved audit log to {filepath}")
