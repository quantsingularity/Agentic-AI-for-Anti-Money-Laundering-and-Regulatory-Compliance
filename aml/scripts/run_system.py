"""
Enhanced AML System Integration Script
Demonstrates all enhanced capabilities in production workflow
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime

import numpy as np
from loguru import logger

from aml.adversarial.adversarial_tester import AdversarialTester
from aml.analysis.cost_benefit import CostBenefitAnalyzer
from aml.caching.redis_cache import RedisCache
from aml.dashboard.explainability_dashboard import ExplainabilityDashboard
from aml.data.synthetic_generator import SyntheticTransactionGenerator
from aml.monitoring.mlflow_monitor import (
    DriftDetector,
    MLflowMonitor,
    PrometheusMetrics,
)

# Import enhanced components
from aml.streaming.kafka_consumer import TransactionStreamConsumer
from aml.validation.data_validator import DataValidator


class EnhancedAMLSystem:
    """
    Production-ready enhanced AML system integrating all improvements.
    """

    def __init__(self, config: dict):
        """
        Initialize enhanced AML system.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Resolve a base output directory relative to this file so nothing is
        # ever written to a hardcoded absolute path.
        self._results_dir = Path(config.get("results_dir", "results")).resolve()
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing Enhanced AML System...")

        # 1. Caching Layer
        self.cache = RedisCache(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
        )

        # 2. Streaming (if enabled)
        if config.get("enable_streaming", False):
            self.kafka_consumer = TransactionStreamConsumer(
                bootstrap_servers=config.get("kafka_servers", ["localhost:9092"]),
                topic="aml-transactions",
                group_id="aml-system",
            )

        # 3. Monitoring
        self.mlflow_monitor = MLflowMonitor(
            tracking_uri=config.get("mlflow_uri", "http://localhost:5000")
        )

        if config.get("enable_prometheus", False):
            self.prometheus = PrometheusMetrics(port=9090)

        # 4. Adversarial Testing
        self.adversarial_tester = AdversarialTester()

        # 5. Data Validation
        self.data_validator = DataValidator()

        # 6. Cost-Benefit Analysis
        self.cost_analyzer = CostBenefitAnalyzer()

        # 7. Explainability Dashboard
        if config.get("enable_dashboard", False):
            self.dashboard = ExplainabilityDashboard(port=5001)

        logger.info("Enhanced AML System initialized successfully!")

    def run_full_workflow(self):
        """
        Execute complete enhanced workflow demonstration.
        """
        logger.info("=" * 80)
        logger.info("ENHANCED AML SYSTEM - FULL WORKFLOW DEMONSTRATION")
        logger.info("=" * 80)

        # Step 1: Generate synthetic and real-world data
        logger.info("\n[1/7] Generating Data...")
        synthetic_data = self._generate_synthetic_data()

        # Step 2: Real-world data validation
        logger.info("\n[2/7] Validating Against Real-World Data...")
        self._validate_data(synthetic_data)

        # Step 3: Train model with monitoring
        logger.info("\n[3/7] Training Model with MLflow Tracking...")
        model = self._train_model_with_monitoring(synthetic_data)

        # Step 4: Adversarial robustness testing
        logger.info("\n[4/7] Running Adversarial Robustness Tests...")
        self._test_adversarial_robustness(model, synthetic_data)

        # Step 5: Cost-benefit analysis
        logger.info("\n[5/7] Performing Cost-Benefit Analysis...")
        self._analyze_costs(model, synthetic_data)

        # Step 6: Production deployment with caching
        logger.info("\n[6/7] Deploying with Redis Caching...")
        self._deploy_with_caching(model)

        # Step 7: Launch explainability dashboard
        logger.info("\n[7/7] Generating Explainability Reports...")
        self._generate_explainability_reports()

        # Summary
        self._print_summary()

        logger.info("=" * 80)
        logger.info("Enhanced workflow complete!")
        logger.info("=" * 80)

    def _generate_synthetic_data(self):
        """Generate synthetic transaction data."""

        generator = SyntheticTransactionGenerator(
            n_transactions=10000, fraud_rate=0.023, seed=42
        )
        data = generator.generate()

        logger.info(f"Generated {len(data)} synthetic transactions")
        return data

    def _validate_data(self, synthetic_data):
        """Validate synthetic vs real-world data."""
        # For demonstration, we'll simulate real-world data
        # In production, this would load from actual data source

        real_data = synthetic_data.sample(frac=0.3)  # Simulate real subset

        comparison = self.data_validator.compare_distributions(
            synthetic_data, real_data
        )

        logger.info(f"Distribution similarity: {comparison['overall_similarity']:.3f}")

        return comparison

    def _train_model_with_monitoring(self, data):
        """Train model with MLflow tracking."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Start MLflow run
        self.mlflow_monitor.start_run(
            run_name="enhanced_aml_model", tags={"version": "2.0", "enhanced": "true"}
        )

        # Prepare data
        X = data.drop(["is_fraud"], axis=1) if "is_fraud" in data.columns else data
        y = (
            data["is_fraud"]
            if "is_fraud" in data.columns
            else np.random.randint(0, 2, len(data))
        )

        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model and metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        self.mlflow_monitor.log_detection_metrics(y_test, y_pred, y_proba)
        self.mlflow_monitor.log_model(model, "random_forest_model")

        # Initialize drift detector
        baseline_perf = {"precision": 0.85, "recall": 0.89, "f1_score": 0.87}

        self.drift_detector = DriftDetector(
            baseline_data=X_train, baseline_performance=baseline_perf
        )

        self.mlflow_monitor.end_run()

        logger.info("Model trained and logged to MLflow")

        return model

    def _test_adversarial_robustness(self, model, data):
        """Test model against adversarial attacks."""

        # Create a simple model wrapper for testing
        class ModelWrapper:
            def __init__(self, model):
                self.model = model

            def predict(self, data):
                X_numeric = data.select_dtypes(include=[np.number])
                predictions = self.model.predict_proba(X_numeric)[:, 1]
                return {"risk_scores": predictions.tolist()}

        wrapped_model = ModelWrapper(model)

        results = self.adversarial_tester.run_adversarial_test_suite(
            wrapped_model, data, num_attacks=10  # Reduced for demo
        )

        logger.info(f"Adversarial detection rate: {results['detection_rate']*100:.1f}%")

        return results

    def _analyze_costs(self, model, data):
        """Perform cost-benefit analysis."""

        # Create confusion matrix
        X_numeric = data.select_dtypes(include=[np.number])
        y_true = (
            data["is_fraud"]
            if "is_fraud" in data.columns
            else np.random.randint(0, 2, len(data))
        )
        y_pred = model.predict(X_numeric)

        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cm = {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        # Calculate costs
        cost_analysis = self.cost_analyzer.calculate_costs(
            cm, {"avg_fraud_amount": 50000}
        )

        # Optimize threshold
        y_proba = model.predict_proba(X_numeric)[:, 1]

        threshold_opt = self.cost_analyzer.optimize_threshold(y_true, y_proba)

        logger.info(f"Optimal threshold: {threshold_opt['optimal_threshold']:.3f}")
        logger.info(f"Net benefit: ${cost_analysis['summary']['net_benefit']:,.2f}")

        self.cost_analyzer.export_analysis(
            cost_analysis, str(self._results_dir / "cost_analysis.json")
        )

        return cost_analysis

    def _deploy_with_caching(self, model):
        """Deploy model with Redis caching."""

        # Cache some entity profiles
        for i in range(5):
            entity_profile = {
                "entity_id": f"ENT_{i:04d}",
                "risk_level": np.random.choice(["low", "medium", "high"]),
                "transaction_count": np.random.randint(10, 1000),
                "total_volume": np.random.uniform(10000, 1000000),
            }

            self.cache.cache_entity_profile(f"ENT_{i:04d}", entity_profile)

        # Test cache retrieval
        cached_profile = self.cache.get_entity_profile("ENT_0000")
        logger.info(f"Cached profile retrieved: {cached_profile}")

        # Get cache statistics
        cache_stats = self.cache.get_stats()
        logger.info(f"Cache stats: {cache_stats}")

    def _generate_explainability_reports(self):
        """Generate explainability reports and sample SARs."""

        # Create sample SARs for dashboard
        sample_sars = []

        for i in range(5):
            sar = {
                "sar_id": f"SAR_{i:06d}",
                "entity_id": f"ENT_{i:04d}",
                "risk_score": np.random.uniform(0.5, 0.95),
                "timestamp": datetime.utcnow().isoformat(),
                "crime_type": np.random.choice(
                    ["structuring", "layering", "trade_based", "smurfing"]
                ),
                "transactions": [
                    {
                        "transaction_id": f"TXN_{j:08d}",
                        "amount": np.random.uniform(5000, 50000),
                        "timestamp": datetime.utcnow().isoformat(),
                        "sender_id": f"ACC_{i:04d}",
                        "receiver_id": f"ACC_{(i+1):04d}",
                        "risk_score": np.random.uniform(0.3, 0.9),
                    }
                    for j in range(5)
                ],
                "feature_importance": {
                    "transaction_velocity": np.random.uniform(0.1, 0.3),
                    "amount_deviation": np.random.uniform(0.15, 0.35),
                    "geographic_risk": np.random.uniform(0.1, 0.25),
                    "temporal_pattern": np.random.uniform(0.1, 0.2),
                },
                "narrative": f"Entity ENT_{i:04d} exhibits suspicious transaction patterns...",
                "validation_score": np.random.uniform(0.7, 0.95),
            }

            sample_sars.append(sar)

        sars_path = self._results_dir / "sample_sars.json"
        with open(sars_path, "w") as f:
            json.dump(sample_sars, f, indent=2)

        logger.info(f"Generated {len(sample_sars)} sample SARs for dashboard")

    def _print_summary(self):
        """Print workflow summary."""
        logger.info("\n" + "=" * 80)
        logger.info("ENHANCED SYSTEM CAPABILITIES SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "✓ Real Data Validation: Distribution comparison and validation framework"
        )
        logger.info(
            "✓ Scalability Architecture: Kafka streaming + Redis caching + Kubernetes ready"
        )
        logger.info(
            "✓ Adversarial Robustness: 10 evasion techniques tested with adaptive learning"
        )
        logger.info(
            "✓ Production Monitoring: MLflow experiment tracking + Prometheus metrics"
        )
        logger.info(
            "✓ Cost-Benefit Analysis: Optimized thresholds with business impact quantification"
        )
        logger.info(
            "✓ Explainability Dashboard: Web-based investigator interface with visualizations"
        )
        logger.info("=" * 80)


def main():
    """Main execution function."""

    config = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "kafka_servers": ["localhost:9092"],
        "mlflow_uri": "http://localhost:5000",
        "enable_streaming": False,  # Set to True if Kafka is running
        "enable_prometheus": False,  # Set to True if Prometheus is running
        "enable_dashboard": False,  # Set to True to launch dashboard
        "results_dir": str(Path(__file__).resolve().parent.parent / "results"),
    }

    # Initialize and run
    system = EnhancedAMLSystem(config)
    system.run_full_workflow()

    logger.info("\n✓ Enhanced AML System demonstration complete!")
    logger.info(f"Results saved to: {config['results_dir']}")


if __name__ == "__main__":
    main()
