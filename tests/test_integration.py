"""
Integration Test - End-to-End Pipeline
Tests complete workflow from data generation to SAR generation.
"""

import pytest
import sys
from pathlib import Path
import json
import pandas as pd

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.data.synthetic_generator import SyntheticTransactionGenerator
from code.agents.privacy_guard import PrivacyGuard
from code.agents.narrative_agent import NarrativeAgent
from code.models.xgboost_classifier import XGBoostClassifier


@pytest.fixture
def synthetic_data():
    """Generate small synthetic dataset for testing."""
    generator = SyntheticTransactionGenerator(n_transactions=1000, fraud_rate=0.05, seed=42)
    df = generator.generate()
    return df


def test_data_generation(synthetic_data):
    """Test synthetic data generation."""
    assert len(synthetic_data) == 1000
    assert 'transaction_id' in synthetic_data.columns
    assert 'is_fraud' in synthetic_data.columns
    assert synthetic_data['is_fraud'].sum() > 0
    
    # Check fraud rate is approximately correct
    fraud_rate = synthetic_data['is_fraud'].mean()
    assert 0.03 < fraud_rate < 0.07  # Allow some variance


def test_privacy_guard(synthetic_data):
    """Test PII redaction."""
    # Create sample transaction with PII
    txn = {
        'sender_id': 'USER_001234',
        'description': 'Payment from john@example.com, SSN: 123-45-6789, Card: 4532-1234-5678-9010'
    }
    
    privacy_guard = PrivacyGuard()
    result = privacy_guard.execute(txn)
    
    assert result['status'] == 'success'
    redacted = result['result']['redacted_data']
    
    # Check PII was redacted
    assert '[REDACTED_EMAIL' in str(redacted) or 'john@example.com' not in str(redacted)
    assert result['result']['redaction_applied']


def test_xgboost_classifier(synthetic_data):
    """Test XGBoost classifier training and prediction."""
    # Split data
    train_df = synthetic_data.iloc[:700]
    test_df = synthetic_data.iloc[700:]
    
    classifier = XGBoostClassifier(task='binary', seed=42)
    
    # Engineer features
    train_features = classifier.engineer_features(train_df)
    test_features = classifier.engineer_features(test_df)
    
    X_train, y_train = classifier.prepare_data(train_features)
    X_test, y_test = classifier.prepare_data(test_features)
    
    # Train
    classifier.train(X_train, y_train)
    
    # Predict
    predictions = classifier.predict(X_test)
    
    assert len(predictions) == len(test_df)
    assert set(predictions).issubset({0, 1})
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['f1'] <= 1


def test_narrative_agent():
    """Test SAR narrative generation."""
    narrative_agent = NarrativeAgent()
    
    # Sample suspicious transactions
    transactions = [
        {
            'transaction_id': 'TXN_00001234',
            'sender_id': 'USER_001234',
            'receiver_id': 'USER_005678',
            'amount': 9500,
            'sender_country': 'US',
            'receiver_country': 'US',
            'timestamp': '2024-01-15T10:30:00',
            'fraud_typology': 'structuring'
        },
        {
            'transaction_id': 'TXN_00001235',
            'sender_id': 'USER_001234',
            'receiver_id': 'USER_009012',
            'amount': 9700,
            'sender_country': 'US',
            'receiver_country': 'US',
            'timestamp': '2024-01-15T14:30:00',
            'fraud_typology': 'structuring'
        }
    ]
    
    result = narrative_agent.execute({
        'subject_id': 'USER_001234',
        'transactions': transactions,
        'evidence': {},
        'typology': 'structuring',
        'risk_score': 0.85
    })
    
    assert result['status'] == 'success'
    assert 'narrative' in result['result']
    assert 'citations' in result['result']
    assert len(result['result']['narrative']) > 100  # Non-trivial narrative


def test_end_to_end_pipeline(synthetic_data, tmp_path):
    """Test complete end-to-end pipeline."""
    
    # Step 1: Data
    assert len(synthetic_data) == 1000
    
    # Step 2: Privacy Guard
    privacy_guard = PrivacyGuard()
    sample_txns = synthetic_data.head(10).to_dict('records')
    privacy_result = privacy_guard.execute(sample_txns)
    assert privacy_result['status'] == 'success'
    
    # Step 3: Classification
    train_df = synthetic_data.iloc[:700]
    test_df = synthetic_data.iloc[700:]
    
    classifier = XGBoostClassifier(task='binary', seed=42)
    train_features = classifier.engineer_features(train_df)
    X_train, y_train = classifier.prepare_data(train_features)
    classifier.train(X_train, y_train)
    
    test_features = classifier.engineer_features(test_df)
    X_test, y_test = classifier.prepare_data(test_features)
    predictions = classifier.predict(X_test)
    
    # Step 4: SAR generation for suspicious cases
    suspicious_indices = [i for i, pred in enumerate(predictions) if pred == 1]
    
    if len(suspicious_indices) > 0:
        # Generate SAR for first suspicious case
        idx = suspicious_indices[0]
        txn = test_df.iloc[idx].to_dict()
        
        narrative_agent = NarrativeAgent()
        sar = narrative_agent.execute({
            'subject_id': txn['sender_id'],
            'transactions': [txn],
            'evidence': {},
            'typology': txn.get('fraud_typology', 'unknown'),
            'risk_score': 0.75
        })
        
        assert sar['status'] == 'success'
        assert 'narrative' in sar['result']
        
        # Save SAR
        sar_file = tmp_path / 'test_sar.json'
        with open(sar_file, 'w') as f:
            json.dump(sar, f, indent=2)
        
        assert sar_file.exists()
    
    print(f"\nEnd-to-end test completed successfully!")
    print(f"  Data generated: {len(synthetic_data)} transactions")
    print(f"  Fraud cases: {synthetic_data['is_fraud'].sum()}")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Flagged as suspicious: {len(suspicious_indices)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
