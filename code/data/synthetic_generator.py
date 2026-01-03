"""
Deterministic Synthetic Transaction Generator
Generates realistic AML transaction data with configurable fraud typologies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import json


class SyntheticTransactionGenerator:
    """
    Generates deterministic synthetic financial transactions with AML patterns.
    
    Features:
    - Deterministic with fixed seed
    - Realistic distributions (amounts, timestamps, entities)
    - Seven crime typologies: structuring, rapid movement, sanctions, 
      high-risk geography, trade-based laundering, shell companies, smurfing
    - Mixing of benign and suspicious transactions
    """
    
    # Crime typology definitions
    TYPOLOGIES = {
        'structuring': {
            'description': 'Multiple transactions just below reporting threshold',
            'threshold': 10000,
            'pattern': 'burst_below_threshold'
        },
        'rapid_movement': {
            'description': 'Funds moved rapidly through multiple accounts',
            'pattern': 'rapid_transfers'
        },
        'sanctions_evasion': {
            'description': 'Transactions involving sanctioned entities',
            'pattern': 'sanctioned_entity'
        },
        'high_risk_geography': {
            'description': 'Transactions from/to high-risk jurisdictions',
            'pattern': 'risky_country'
        },
        'trade_based': {
            'description': 'Trade-based money laundering indicators',
            'pattern': 'mispriced_goods'
        },
        'shell_company': {
            'description': 'Shell companies with minimal activity',
            'pattern': 'dormant_sudden_activity'
        },
        'smurfing': {
            'description': 'Many small deposits from multiple sources',
            'pattern': 'many_small_deposits'
        }
    }
    
    HIGH_RISK_COUNTRIES = ['SY', 'IR', 'KP', 'VE', 'MM', 'AF', 'IQ']
    SANCTIONED_ENTITIES = [
        'Acme Imports LLC', 'Global Trade Co', 'Eastern Finance Corp',
        'Pacific Resources Ltd', 'Continental Holdings'
    ]
    
    def __init__(self, n_transactions: int = 100000, fraud_rate: float = 0.023, 
                 seed: int = 42):
        """
        Initialize generator.
        
        Args:
            n_transactions: Total number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (default 2.3%)
            seed: Random seed for reproducibility
        """
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate fraud distribution
        self.n_fraud = int(n_transactions * fraud_rate)
        self.n_benign = n_transactions - self.n_fraud
        
    def generate(self) -> pd.DataFrame:
        """
        Generate complete transaction dataset.
        
        Returns:
            DataFrame with transaction features and labels
        """
        # Generate benign transactions
        benign_txns = self._generate_benign(self.n_benign)
        
        # Generate fraudulent transactions (distributed across typologies)
        fraud_txns = self._generate_fraud(self.n_fraud)
        
        # Combine and shuffle
        all_txns = pd.concat([benign_txns, fraud_txns], ignore_index=True)
        
        # Shuffle with fixed seed
        all_txns = all_txns.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Add transaction IDs
        all_txns['transaction_id'] = [f"TXN_{i:08d}" for i in range(len(all_txns))]
        
        # Sort by timestamp
        all_txns = all_txns.sort_values('timestamp').reset_index(drop=True)
        
        return all_txns
    
    def _generate_benign(self, n: int) -> pd.DataFrame:
        """Generate benign transactions."""
        data = {
            'amount': self.rng.lognormal(mean=6.5, sigma=1.5, size=n),
            'sender_id': [f"USER_{self.rng.randint(1, 50000):06d}" for _ in range(n)],
            'receiver_id': [f"USER_{self.rng.randint(1, 50000):06d}" for _ in range(n)],
            'sender_country': self.rng.choice(['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP'], size=n),
            'receiver_country': self.rng.choice(['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP'], size=n),
            'transaction_type': self.rng.choice(['transfer', 'payment', 'withdrawal'], size=n),
            'is_fraud': 0,
            'fraud_typology': 'none'
        }
        
        # Generate timestamps over 90 days
        base_time = datetime(2024, 1, 1)
        data['timestamp'] = [base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60)) 
                            for _ in range(n)]
        
        return pd.DataFrame(data)
    
    def _generate_fraud(self, n: int) -> pd.DataFrame:
        """Generate fraudulent transactions across typologies."""
        fraud_dfs = []
        
        # Distribute across typologies
        typology_names = list(self.TYPOLOGIES.keys())
        n_per_typology = n // len(typology_names)
        
        for typology in typology_names:
            fraud_dfs.append(self._generate_typology(typology, n_per_typology))
        
        # Add remaining to random typologies
        remaining = n - len(fraud_dfs) * n_per_typology
        if remaining > 0:
            fraud_dfs.append(self._generate_typology(
                self.rng.choice(typology_names), remaining
            ))
        
        return pd.concat(fraud_dfs, ignore_index=True)
    
    def _generate_typology(self, typology: str, n: int) -> pd.DataFrame:
        """Generate transactions for specific typology."""
        if typology == 'structuring':
            return self._generate_structuring(n)
        elif typology == 'rapid_movement':
            return self._generate_rapid_movement(n)
        elif typology == 'sanctions_evasion':
            return self._generate_sanctions_evasion(n)
        elif typology == 'high_risk_geography':
            return self._generate_high_risk_geography(n)
        elif typology == 'trade_based':
            return self._generate_trade_based(n)
        elif typology == 'shell_company':
            return self._generate_shell_company(n)
        elif typology == 'smurfing':
            return self._generate_smurfing(n)
        else:
            raise ValueError(f"Unknown typology: {typology}")
    
    def _generate_structuring(self, n: int) -> pd.DataFrame:
        """Structuring: amounts just below $10K threshold."""
        # Generate clusters of transactions
        n_clusters = max(1, n // 5)
        cluster_sizes = self.rng.poisson(5, n_clusters)
        
        data = []
        base_time = datetime(2024, 1, 1)
        
        for cluster_size in cluster_sizes[:n]:
            cluster_sender = f"USER_{self.rng.randint(50000, 60000):06d}"
            cluster_time = base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60))
            
            for i in range(min(cluster_size, n - len(data))):
                data.append({
                    'amount': self.rng.uniform(9000, 9900),
                    'sender_id': cluster_sender,
                    'receiver_id': f"USER_{self.rng.randint(1, 50000):06d}",
                    'sender_country': 'US',
                    'receiver_country': self.rng.choice(['US', 'GB', 'CA']),
                    'transaction_type': 'transfer',
                    'timestamp': cluster_time + timedelta(hours=i),
                    'is_fraud': 1,
                    'fraud_typology': 'structuring'
                })
                
                if len(data) >= n:
                    break
            
            if len(data) >= n:
                break
        
        return pd.DataFrame(data[:n])
    
    def _generate_rapid_movement(self, n: int) -> pd.DataFrame:
        """Rapid movement: funds moved quickly through accounts."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        # Create chains
        n_chains = max(1, n // 4)
        
        for _ in range(n_chains):
            chain_length = self.rng.randint(3, 6)
            amount = self.rng.uniform(50000, 500000)
            start_time = base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60))
            
            accounts = [f"USER_{self.rng.randint(60000, 70000):06d}" for _ in range(chain_length + 1)]
            
            for i in range(min(chain_length, n - len(data))):
                data.append({
                    'amount': amount * self.rng.uniform(0.95, 1.0),
                    'sender_id': accounts[i],
                    'receiver_id': accounts[i + 1],
                    'sender_country': self.rng.choice(['US', 'GB', 'CH', 'LU']),
                    'receiver_country': self.rng.choice(['US', 'GB', 'CH', 'LU']),
                    'transaction_type': 'transfer',
                    'timestamp': start_time + timedelta(minutes=i * 15),
                    'is_fraud': 1,
                    'fraud_typology': 'rapid_movement'
                })
                
                if len(data) >= n:
                    break
            
            if len(data) >= n:
                break
        
        return pd.DataFrame(data[:n])
    
    def _generate_sanctions_evasion(self, n: int) -> pd.DataFrame:
        """Sanctions evasion: transactions with sanctioned entities."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        for _ in range(n):
            data.append({
                'amount': self.rng.lognormal(mean=8.0, sigma=1.5),
                'sender_id': f"USER_{self.rng.randint(1, 50000):06d}",
                'receiver_id': self.rng.choice(self.SANCTIONED_ENTITIES),
                'sender_country': self.rng.choice(['US', 'GB', 'DE']),
                'receiver_country': self.rng.choice(self.HIGH_RISK_COUNTRIES),
                'transaction_type': 'transfer',
                'timestamp': base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60)),
                'is_fraud': 1,
                'fraud_typology': 'sanctions_evasion'
            })
        
        return pd.DataFrame(data)
    
    def _generate_high_risk_geography(self, n: int) -> pd.DataFrame:
        """High-risk geography: transactions to/from risky countries."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        for _ in range(n):
            data.append({
                'amount': self.rng.lognormal(mean=7.5, sigma=1.8),
                'sender_id': f"USER_{self.rng.randint(1, 50000):06d}",
                'receiver_id': f"USER_{self.rng.randint(1, 50000):06d}",
                'sender_country': self.rng.choice(['US', 'GB']),
                'receiver_country': self.rng.choice(self.HIGH_RISK_COUNTRIES),
                'transaction_type': 'transfer',
                'timestamp': base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60)),
                'is_fraud': 1,
                'fraud_typology': 'high_risk_geography'
            })
        
        return pd.DataFrame(data)
    
    def _generate_trade_based(self, n: int) -> pd.DataFrame:
        """Trade-based ML: mispriced goods."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        for _ in range(n):
            data.append({
                'amount': self.rng.uniform(100000, 2000000),  # Unusually large
                'sender_id': f"CORP_{self.rng.randint(1000, 2000):04d}",
                'receiver_id': f"CORP_{self.rng.randint(1000, 2000):04d}",
                'sender_country': self.rng.choice(['US', 'CN', 'HK']),
                'receiver_country': self.rng.choice(['US', 'CN', 'HK']),
                'transaction_type': 'trade_payment',
                'timestamp': base_time + timedelta(seconds=self.rng.randint(0, 90*24*60*60)),
                'is_fraud': 1,
                'fraud_typology': 'trade_based'
            })
        
        return pd.DataFrame(data)
    
    def _generate_shell_company(self, n: int) -> pd.DataFrame:
        """Shell company: dormant account sudden activity."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        n_shell_companies = max(1, n // 10)
        
        for _ in range(n_shell_companies):
            company_id = f"SHELL_{self.rng.randint(5000, 6000):04d}"
            burst_time = base_time + timedelta(days=self.rng.randint(60, 89))  # Late activity
            burst_size = min(10, n - len(data))
            
            for i in range(burst_size):
                data.append({
                    'amount': self.rng.lognormal(mean=8.5, sigma=1.0),
                    'sender_id': company_id,
                    'receiver_id': f"USER_{self.rng.randint(1, 50000):06d}",
                    'sender_country': self.rng.choice(['PA', 'KY', 'BZ', 'VG']),  # Tax havens
                    'receiver_country': self.rng.choice(['US', 'GB', 'CA']),
                    'transaction_type': 'transfer',
                    'timestamp': burst_time + timedelta(hours=i),
                    'is_fraud': 1,
                    'fraud_typology': 'shell_company'
                })
                
                if len(data) >= n:
                    break
            
            if len(data) >= n:
                break
        
        return pd.DataFrame(data[:n])
    
    def _generate_smurfing(self, n: int) -> pd.DataFrame:
        """Smurfing: many small deposits to single account."""
        data = []
        base_time = datetime(2024, 1, 1)
        
        n_target_accounts = max(1, n // 20)
        
        for _ in range(n_target_accounts):
            target_account = f"USER_{self.rng.randint(70000, 80000):06d}"
            campaign_time = base_time + timedelta(seconds=self.rng.randint(0, 80*24*60*60))
            n_smurfs = min(20, n - len(data))
            
            for i in range(n_smurfs):
                data.append({
                    'amount': self.rng.uniform(1000, 3000),
                    'sender_id': f"USER_{self.rng.randint(1, 50000):06d}",
                    'receiver_id': target_account,
                    'sender_country': 'US',
                    'receiver_country': 'US',
                    'transaction_type': 'deposit',
                    'timestamp': campaign_time + timedelta(hours=i * 2),
                    'is_fraud': 1,
                    'fraud_typology': 'smurfing'
                })
                
                if len(data) >= n:
                    break
            
            if len(data) >= n:
                break
        
        return pd.DataFrame(data[:n])
    
    def validate_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate generated data against expected distributions.
        
        Args:
            df: Generated transaction DataFrame
            
        Returns:
            Dict with validation statistics
        """
        stats = {
            'total_transactions': len(df),
            'fraud_rate': df['is_fraud'].mean(),
            'fraud_count': df['is_fraud'].sum(),
            'typology_distribution': df['fraud_typology'].value_counts().to_dict(),
            'amount_stats': {
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            },
            'country_distribution': {
                'sender': df['sender_country'].value_counts().head(10).to_dict(),
                'receiver': df['receiver_country'].value_counts().head(10).to_dict()
            },
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            }
        }
        
        return stats


def generate_and_save(output_path: str = 'data/synthetic/transactions.csv',
                     n_transactions: int = 100000,
                     fraud_rate: float = 0.023,
                     seed: int = 42) -> pd.DataFrame:
    """
    Generate and save synthetic transaction data.
    
    Args:
        output_path: Path to save CSV
        n_transactions: Number of transactions
        fraud_rate: Fraud proportion
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    generator = SyntheticTransactionGenerator(n_transactions, fraud_rate, seed)
    df = generator.generate()
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions, saved to {output_path}")
    
    # Validate and save stats
    stats = generator.validate_distribution(df)
    stats_path = output_path.replace('.csv', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Validation statistics saved to {stats_path}")
    
    return df


if __name__ == '__main__':
    import os
    os.makedirs('data/synthetic', exist_ok=True)
    df = generate_and_save()
    print("\nFirst 5 transactions:")
    print(df.head())
    print(f"\nFraud rate: {df['is_fraud'].mean():.3f}")
    print(f"Typology distribution:\n{df['fraud_typology'].value_counts()}")
