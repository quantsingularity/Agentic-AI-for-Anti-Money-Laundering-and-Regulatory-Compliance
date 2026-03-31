"""
Production Monitoring and Model Drift Detection
Integrates MLflow for experiment tracking and Prometheus for system monitoring
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class MLflowMonitor:
    """
    MLflow integration for model monitoring and experiment tracking.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "aml-detection",
    ):
        """
        Initialize MLflow monitoring.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.current_run_id = None

        logger.info(f"Initialized MLflow monitoring at {tracking_uri}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional run name
            tags: Optional tags dict
        """
        run = mlflow.start_run(run_name=run_name)
        self.current_run_id = run.info.run_id

        if tags:
            mlflow.set_tags(tags)

        # Log system info
        mlflow.log_param("start_time", datetime.utcnow().isoformat())

        logger.info(f"Started MLflow run: {self.current_run_id}")
        return run

    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ):
        """
        Log a trained model to MLflow.

        Args:
            model: Trained model object
            model_name: Model artifact name
            signature: Optional model signature
            input_example: Optional input example
        """
        try:
            # Detect model type and use appropriate logger
            if hasattr(model, "__class__"):
                model_type = model.__class__.__name__

                if "XGB" in model_type:
                    mlflow.xgboost.log_model(
                        model,
                        model_name,
                        signature=signature,
                        input_example=input_example,
                    )
                elif "sklearn" in str(type(model).__module__):
                    mlflow.sklearn.log_model(
                        model,
                        model_name,
                        signature=signature,
                        input_example=input_example,
                    )
                else:
                    mlflow.log_artifact(model_name)

            logger.info(f"Logged model '{model_name}' to MLflow")

        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dict of metric name to value
            step: Optional step number for time series
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
        logger.debug(f"Logged params: {list(params.keys())}")

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Log an artifact file to MLflow.

        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional artifact name
        """
        mlflow.log_artifact(artifact_path, artifact_name)
        logger.debug(f"Logged artifact: {artifact_path}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dict to log
            filename: JSON filename
        """
        mlflow.log_dict(dictionary, filename)

    def log_detection_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ):
        """
        Log standard AML detection metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            step: Optional step for time series
        """
        metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except:
                pass

        # Calculate false positive rate
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

        self.log_metrics(metrics, step=step)

        return metrics

    def end_run(self):
        """End current MLflow run."""
        if self.current_run_id:
            mlflow.log_param("end_time", datetime.utcnow().isoformat())
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.current_run_id}")
            self.current_run_id = None

    def load_model(self, run_id: str, model_name: str = "model"):
        """
        Load a model from MLflow.

        Args:
            run_id: MLflow run ID
            model_name: Model artifact name

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.pyfunc.load_model(model_uri)

    def compare_models(self, run_ids: List[str], metric: str = "f1_score") -> Dict:
        """
        Compare multiple model runs.

        Args:
            run_ids: List of run IDs to compare
            metric: Metric to compare

        Returns:
            Comparison results dict
        """
        client = mlflow.tracking.MlflowClient()

        comparisons = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            metric_value = run.data.metrics.get(metric, None)

            comparisons.append(
                {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                    metric: metric_value,
                    "start_time": run.info.start_time,
                }
            )

        # Sort by metric
        comparisons.sort(key=lambda x: x.get(metric, 0), reverse=True)

        return {
            "metric": metric,
            "runs": comparisons,
            "best_run": comparisons[0] if comparisons else None,
        }


class DriftDetector:
    """
    Model drift detection system.
    Monitors for data drift and model performance degradation.
    """

    def __init__(
        self,
        baseline_data: pd.DataFrame,
        baseline_performance: Dict[str, float],
        drift_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            baseline_data: Baseline dataset for comparison
            baseline_performance: Baseline model performance metrics
            drift_threshold: Threshold for triggering drift alert
        """
        self.baseline_data = baseline_data
        self.baseline_performance = baseline_performance
        self.drift_threshold = drift_threshold

        # Calculate baseline statistics
        self.baseline_stats = self._calculate_statistics(baseline_data)

        self.drift_history = []

        logger.info("Initialized Drift Detector")

    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect data drift using statistical tests.

        Args:
            current_data: Current production data

        Returns:
            Drift detection results
        """
        current_stats = self._calculate_statistics(current_data)

        drift_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": False,
            "features_drifted": [],
            "drift_scores": {},
        }

        # Compare distributions for each feature
        for feature in self.baseline_stats.keys():
            if feature in current_stats:
                drift_score = self._calculate_drift_score(
                    self.baseline_stats[feature], current_stats[feature]
                )

                drift_results["drift_scores"][feature] = drift_score

                if drift_score > self.drift_threshold:
                    drift_results["drift_detected"] = True
                    drift_results["features_drifted"].append(feature)

        self.drift_history.append(drift_results)

        if drift_results["drift_detected"]:
            logger.warning(
                f"Data drift detected! {len(drift_results['features_drifted'])} features drifted"
            )

        return drift_results

    def detect_performance_drift(self, current_performance: Dict[str, float]) -> Dict:
        """
        Detect model performance drift.

        Args:
            current_performance: Current model performance metrics

        Returns:
            Performance drift results
        """
        drift_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": False,
            "metrics_degraded": [],
            "performance_changes": {},
        }

        for metric, baseline_value in self.baseline_performance.items():
            if metric in current_performance:
                current_value = current_performance[metric]
                change = (current_value - baseline_value) / baseline_value

                drift_results["performance_changes"][metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_percent": change * 100,
                }

                # Check if performance degraded significantly
                if change < -self.drift_threshold:
                    drift_results["drift_detected"] = True
                    drift_results["metrics_degraded"].append(metric)

        if drift_results["drift_detected"]:
            logger.warning(
                f"Performance drift detected! {len(drift_results['metrics_degraded'])} metrics degraded"
            )

        return drift_results

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical features of data."""
        stats = {}

        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "median": data[col].median(),
                "q25": data[col].quantile(0.25),
                "q75": data[col].quantile(0.75),
            }

        return stats

    def _calculate_drift_score(
        self, baseline_stats: Dict, current_stats: Dict
    ) -> float:
        """
        Calculate drift score between baseline and current statistics.
        Uses normalized difference of means and standard deviations.
        """
        # Mean shift
        mean_diff = abs(current_stats["mean"] - baseline_stats["mean"])
        mean_shift = mean_diff / (baseline_stats["std"] + 1e-6)

        # Variance shift
        std_ratio = current_stats["std"] / (baseline_stats["std"] + 1e-6)
        variance_shift = abs(std_ratio - 1)

        # Combined drift score
        drift_score = (mean_shift + variance_shift) / 2

        return drift_score

    def get_drift_summary(self, window_days: int = 7) -> Dict:
        """
        Get summary of drift over time window.

        Args:
            window_days: Number of days to analyze

        Returns:
            Drift summary statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=window_days)

        recent_drifts = [
            d
            for d in self.drift_history
            if datetime.fromisoformat(d["timestamp"]) > cutoff
        ]

        total_checks = len(recent_drifts)
        drift_detected = sum(1 for d in recent_drifts if d["drift_detected"])

        return {
            "window_days": window_days,
            "total_checks": total_checks,
            "drifts_detected": drift_detected,
            "drift_rate": drift_detected / max(1, total_checks),
            "recent_drifts": recent_drifts[-10:],  # Last 10
        }


class PrometheusMetrics:
    """
    Prometheus metrics exporter for AML system monitoring.
    """

    def __init__(self, port: int = 9090):
        """
        Initialize Prometheus metrics.

        Args:
            port: Prometheus exporter port
        """
        from prometheus_client import Counter, Gauge, Histogram, start_http_server

        self.port = port

        # Define metrics
        self.transactions_processed = Counter(
            "aml_transactions_processed_total", "Total number of transactions processed"
        )

        self.sars_generated = Counter(
            "aml_sars_generated_total", "Total number of SARs generated"
        )

        self.processing_time = Histogram(
            "aml_processing_seconds",
            "Time to process transaction batch",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.risk_score = Histogram(
            "aml_risk_score",
            "Distribution of risk scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.active_alerts = Gauge("aml_active_alerts", "Number of active alerts")

        self.model_accuracy = Gauge("aml_model_accuracy", "Current model accuracy")

        self.cache_hit_rate = Gauge("aml_cache_hit_rate", "Redis cache hit rate")

        # Start HTTP server
        start_http_server(port)
        logger.info(f"Started Prometheus metrics server on port {port}")

    def record_transaction_batch(self, count: int):
        """Record processed transaction count."""
        self.transactions_processed.inc(count)

    def record_sar_generated(self):
        """Record SAR generation."""
        self.sars_generated.inc()

    def record_processing_time(self, seconds: float):
        """Record processing time."""
        self.processing_time.observe(seconds)

    def record_risk_score(self, score: float):
        """Record risk score."""
        self.risk_score.observe(score)

    def update_active_alerts(self, count: int):
        """Update active alerts gauge."""
        self.active_alerts.set(count)

    def update_model_accuracy(self, accuracy: float):
        """Update model accuracy gauge."""
        self.model_accuracy.set(accuracy)

    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate."""
        self.cache_hit_rate.set(hit_rate)
