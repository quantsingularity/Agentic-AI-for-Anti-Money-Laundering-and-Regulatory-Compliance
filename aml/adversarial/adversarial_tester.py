"""
Adversarial Testing Framework for AML System
Simulates sophisticated money laundering evasion techniques
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class AdversarialTester:
    """
    Framework for testing AML system against adversarial evasion techniques.
    Simulates sophisticated money laundering typologies.
    """

    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        """
        Initialize adversarial testing framework.

        """
        self.config = config or {}
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.evasion_techniques = self._load_evasion_techniques()
        self.test_results = []

        logger.info(f"Initialized Adversarial Testing Framework with seed={seed}")

    def _load_evasion_techniques(self) -> Dict:
        """Define sophisticated evasion techniques."""
        return {
            "structuring": {
                "description": "Breaking large transactions into smaller amounts below reporting thresholds",
                "implementation": self._generate_structuring_attack,
            },
            "layering": {
                "description": "Complex chain of transactions to obscure origin",
                "implementation": self._generate_layering_attack,
            },
            "rapid_movement": {
                "description": "Rapid cross-border fund movements",
                "implementation": self._generate_rapid_movement_attack,
            },
            "shell_company_network": {
                "description": "Network of shell companies for fund routing",
                "implementation": self._generate_shell_network_attack,
            },
            "trade_based_laundering": {
                "description": "Over/under-invoicing in trade transactions",
                "implementation": self._generate_trade_based_attack,
            },
            "crypto_mixing": {
                "description": "Cryptocurrency mixing and chain-hopping",
                "implementation": self._generate_crypto_mixing_attack,
            },
            "smurfing": {
                "description": "Using multiple individuals to make small deposits",
                "implementation": self._generate_smurfing_attack,
            },
            "timing_manipulation": {
                "description": "Strategic timing to avoid detection windows",
                "implementation": self._generate_timing_attack,
            },
            "geographic_arbitrage": {
                "description": "Exploiting jurisdictional gaps",
                "implementation": self._generate_geographic_attack,
            },
            "velocity_variation": {
                "description": "Varying transaction velocity to avoid patterns",
                "implementation": self._generate_velocity_attack,
            },
        }

    def run_adversarial_test_suite(
        self, aml_system, baseline_transactions: pd.DataFrame, num_attacks: int = 100
    ) -> Dict:
        """
        Run comprehensive adversarial test suite.

        Args:
            aml_system: AML detection system to test
            baseline_transactions: Clean transaction dataset
            num_attacks: Number of attack scenarios per technique

        Returns:
            Dict with test results and detection rates
        """
        logger.info(
            f"Starting adversarial test suite with {num_attacks} attacks per technique"
        )

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_attacks": 0,
            "detected": 0,
            "missed": 0,
            "techniques": {},
        }

        for technique_name, technique_info in self.evasion_techniques.items():
            logger.info(f"Testing technique: {technique_name}")

            technique_results = self._test_technique(
                technique_name,
                technique_info,
                aml_system,
                baseline_transactions,
                num_attacks,
            )

            results["techniques"][technique_name] = technique_results
            results["total_attacks"] += technique_results["total"]
            results["detected"] += technique_results["detected"]
            results["missed"] += technique_results["missed"]

        # Calculate overall metrics
        results["detection_rate"] = results["detected"] / max(
            1, results["total_attacks"]
        )
        results["evasion_rate"] = results["missed"] / max(1, results["total_attacks"])

        self._generate_adversarial_report(results)

        return results

    def _test_technique(
        self,
        technique_name: str,
        technique_info: Dict,
        aml_system,
        baseline_data: pd.DataFrame,
        num_attacks: int,
    ) -> Dict:
        """Test a specific evasion technique."""

        detected = 0
        missed = 0
        attack_scores = []

        for i in range(num_attacks):
            # Generate adversarial transactions
            attack_txns = technique_info["implementation"](baseline_data)

            # Run through AML system
            try:
                predictions = aml_system.predict(attack_txns)

                # Check if attack was detected
                if self._is_detected(predictions):
                    detected += 1
                else:
                    missed += 1

                # Store risk scores
                attack_scores.extend(predictions.get("risk_scores", []))

            except Exception as e:
                logger.error(f"Error testing {technique_name}: {e}")
                missed += 1

        return {
            "technique": technique_name,
            "description": technique_info["description"],
            "total": num_attacks,
            "detected": detected,
            "missed": missed,
            "detection_rate": detected / max(1, num_attacks),
            "avg_risk_score": float(np.mean(attack_scores)) if attack_scores else 0.0,
            "max_risk_score": float(np.max(attack_scores)) if attack_scores else 0.0,
        }

    def _is_detected(self, predictions: Dict, threshold: float = 0.5) -> bool:
        """Check if attack was detected."""
        risk_scores = predictions.get("risk_scores", [])
        if not risk_scores:
            return False
        return max(risk_scores) >= threshold

    # Evasion technique implementations

    def _generate_structuring_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate structuring attack (smurfing).
        Split large transaction into multiple below threshold.
        """
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        large_amount = np.random.uniform(15000, 50000)
        num_splits = np.random.randint(5, 15)
        split_amounts = self._split_amount_randomly(large_amount, num_splits)

        attack_txns = []
        base_time = datetime.now()

        for i, amount in enumerate(split_amounts):
            txn = sample.copy()
            txn["amount"] = amount
            txn["timestamp"] = base_time + timedelta(hours=i * np.random.uniform(2, 8))
            txn["is_adversarial"] = True
            txn["attack_type"] = "structuring"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_layering_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate layering attack with complex transaction chains.
        """
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        # Create chain of 8-15 transactions
        chain_length = np.random.randint(8, 15)
        initial_amount = np.random.uniform(50000, 200000)

        attack_txns = []
        current_amount = initial_amount
        base_time = datetime.now()

        for i in range(chain_length):
            txn = sample.copy()
            # Random fee/split at each layer
            current_amount *= np.random.uniform(0.85, 0.98)
            txn["amount"] = current_amount
            txn["timestamp"] = base_time + timedelta(
                minutes=i * np.random.uniform(15, 60)
            )
            txn["sender_id"] = f"layer_{i}"
            txn["receiver_id"] = f"layer_{i+1}"
            txn["is_adversarial"] = True
            txn["attack_type"] = "layering"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_rapid_movement_attack(
        self, baseline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate rapid cross-border movement attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        countries = ["US", "UK", "CH", "LU", "SG", "HK", "PA", "BM"]
        num_movements = np.random.randint(5, 12)

        attack_txns = []
        base_time = datetime.now()
        amount = np.random.uniform(100000, 500000)

        for i in range(num_movements):
            txn = sample.copy()
            txn["amount"] = amount * np.random.uniform(0.95, 1.0)
            txn["timestamp"] = base_time + timedelta(hours=i * 0.5)
            txn["sender_country"] = countries[i % len(countries)]
            txn["receiver_country"] = countries[(i + 1) % len(countries)]
            txn["is_adversarial"] = True
            txn["attack_type"] = "rapid_movement"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_shell_network_attack(
        self, baseline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate shell company network attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        num_shells = np.random.randint(6, 12)
        amount = np.random.uniform(200000, 1000000)

        attack_txns = []
        base_time = datetime.now()

        for i in range(num_shells):
            txn = sample.copy()
            txn["amount"] = amount / num_shells * np.random.uniform(0.8, 1.2)
            txn["timestamp"] = base_time + timedelta(
                days=i, hours=np.random.randint(0, 24)
            )
            txn["sender_id"] = f"shell_company_{i}"
            txn["receiver_id"] = f"shell_company_{(i+1) % num_shells}"
            txn["business_type"] = random.choice(
                ["import_export", "consulting", "real_estate"]
            )
            txn["is_adversarial"] = True
            txn["attack_type"] = "shell_network"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_trade_based_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trade-based money laundering attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        num_trades = np.random.randint(3, 8)
        attack_txns = []
        base_time = datetime.now()

        for i in range(num_trades):
            txn = sample.copy()
            # Over/under invoice by 200-400%
            inflation_factor = np.random.uniform(2.0, 4.0)
            txn["amount"] = np.random.uniform(50000, 300000) * inflation_factor
            txn["timestamp"] = base_time + timedelta(days=i * 30)
            txn["transaction_type"] = "trade"
            txn["goods_description"] = random.choice(
                ["electronics", "textiles", "machinery"]
            )
            txn["is_adversarial"] = True
            txn["attack_type"] = "trade_based"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_crypto_mixing_attack(
        self, baseline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate cryptocurrency mixing attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        num_hops = np.random.randint(8, 20)
        attack_txns = []
        base_time = datetime.now()
        amount = np.random.uniform(10000, 100000)

        for i in range(num_hops):
            txn = sample.copy()
            txn["amount"] = amount * np.random.uniform(0.98, 1.0)
            txn["timestamp"] = base_time + timedelta(
                minutes=i * np.random.uniform(5, 30)
            )
            txn["currency"] = random.choice(["BTC", "ETH", "XMR", "USDT"])
            txn["wallet_address"] = f"0x{np.random.randint(1000000, 9999999)}"
            txn["mixer_used"] = random.choice([True, False])
            txn["is_adversarial"] = True
            txn["attack_type"] = "crypto_mixing"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_smurfing_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """Generate smurfing attack (multiple individuals)."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        num_smurfs = np.random.randint(10, 30)
        total_amount = np.random.uniform(100000, 500000)

        attack_txns = []
        base_time = datetime.now()

        for i in range(num_smurfs):
            txn = sample.copy()
            txn["amount"] = total_amount / num_smurfs * np.random.uniform(0.8, 1.2)
            txn["timestamp"] = base_time + timedelta(
                hours=i * np.random.uniform(0.5, 4)
            )
            txn["sender_id"] = f"smurf_{i}"
            txn["ip_address"] = (
                f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            )
            txn["is_adversarial"] = True
            txn["attack_type"] = "smurfing"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_timing_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """Generate timing manipulation attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        # Transactions timed to avoid detection windows (e.g., weekends, holidays)
        attack_txns = []
        base_time = datetime.now()

        # Find weekend/holiday times
        for i in range(5):
            txn = sample.copy()
            # Saturday/Sunday
            offset_days = i * 7 + (5 if i % 2 == 0 else 6)
            txn["timestamp"] = base_time + timedelta(
                days=offset_days, hours=np.random.randint(20, 24)
            )
            txn["amount"] = np.random.uniform(8000, 14000)
            txn["is_adversarial"] = True
            txn["attack_type"] = "timing_manipulation"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_geographic_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """Generate geographic arbitrage attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        # Use jurisdictions with weaker AML enforcement
        weak_jurisdictions = ["VG", "KY", "PA", "SC", "LI", "MC"]

        attack_txns = []
        base_time = datetime.now()

        for i in range(6):
            txn = sample.copy()
            txn["amount"] = np.random.uniform(50000, 200000)
            txn["timestamp"] = base_time + timedelta(days=i * 7)
            txn["sender_country"] = weak_jurisdictions[i % len(weak_jurisdictions)]
            txn["receiver_country"] = weak_jurisdictions[
                (i + 1) % len(weak_jurisdictions)
            ]
            txn["is_adversarial"] = True
            txn["attack_type"] = "geographic_arbitrage"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    def _generate_velocity_attack(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """Generate velocity variation attack."""
        sample = baseline_data.sample(n=1).iloc[0].to_dict()

        attack_txns = []
        base_time = datetime.now()

        # Vary velocity randomly to avoid pattern detection
        velocities = [0.5, 2, 8, 0.1, 24, 1, 12, 0.25]  # hours between txns

        for i, velocity in enumerate(velocities):
            txn = sample.copy()
            txn["amount"] = np.random.uniform(5000, 15000)
            base_time += timedelta(hours=velocity)
            txn["timestamp"] = base_time
            txn["is_adversarial"] = True
            txn["attack_type"] = "velocity_variation"
            attack_txns.append(txn)

        return pd.DataFrame(attack_txns)

    # Helper methods

    def _split_amount_randomly(self, total: float, num_splits: int) -> List[float]:
        """Split amount into random parts that sum to total."""
        # Generate random splits
        splits = np.random.dirichlet(np.ones(num_splits)) * total
        # Add some variation
        splits = splits * np.random.uniform(0.9, 1.1, num_splits)
        # Normalize to exact total
        splits = splits * (total / splits.sum())
        return splits.tolist()

    def _generate_adversarial_report(self, results: Dict):
        """Generate comprehensive adversarial testing report."""
        logger.info("\n" + "=" * 80)
        logger.info("ADVERSARIAL TESTING REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Attacks: {results['total_attacks']}")
        logger.info(
            f"Detected: {results['detected']} ({results['detection_rate']*100:.1f}%)"
        )
        logger.info(f"Missed: {results['missed']} ({results['evasion_rate']*100:.1f}%)")
        logger.info("\nPer-Technique Results:")
        logger.info("-" * 80)

        for technique_name, tech_results in results["techniques"].items():
            logger.info(f"\n{technique_name.upper()}")
            logger.info(f"  Detection Rate: {tech_results['detection_rate']*100:.1f}%")
            logger.info(f"  Avg Risk Score: {tech_results['avg_risk_score']:.3f}")
            logger.info(f"  Max Risk Score: {tech_results['max_risk_score']:.3f}")

        logger.info("\n" + "=" * 80)


# Missing import for List (used in type hint of _split_amount_randomly)
from typing import List  # noqa: E402  (placed after class body intentionally)


class AdaptiveLearner:
    """
    Adaptive learning system that learns from adversarial attacks.
    Updates detection models with new typologies.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.adversarial_memory = []
        self.adaptation_history = []

        logger.info("Initialized Adaptive Learning System")

    def learn_from_attack(self, attack_data: pd.DataFrame, was_detected: bool):
        """
        Learn from adversarial attack.

        Args:
            attack_data: Adversarial transaction data
            was_detected: Whether attack was detected
        """
        self.adversarial_memory.append(
            {
                "attack_data": attack_data,
                "detected": was_detected,
                "timestamp": datetime.utcnow(),
            }
        )

        # Trigger retraining if enough new attacks accumulated
        if len(self.adversarial_memory) >= 100:
            self._retrain_model()

    def _retrain_model(self):
        """Retrain model with adversarial examples."""
        logger.info("Retraining model with adversarial examples...")

        # Combine adversarial examples
        adversarial_df = pd.concat(
            [ex["attack_data"] for ex in self.adversarial_memory]
        )

        # Add to training data and retrain
        # Implementation depends on base model type

        self.adaptation_history.append(
            {
                "timestamp": datetime.utcnow(),
                "num_examples": len(adversarial_df),
                "metrics": {},  # Would contain new performance metrics
            }
        )

        # Clear memory
        self.adversarial_memory = []

        logger.info("Model adaptation complete")
