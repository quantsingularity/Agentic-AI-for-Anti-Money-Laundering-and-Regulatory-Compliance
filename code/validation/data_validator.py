"""
Real-World Data Validation and Comparison Framework
Compares synthetic vs real-world performance metrics
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class DataValidator:
    """
    Validates AML system performance on real-world data.
    Compares synthetic vs production data characteristics and model performance.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data validator.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.validation_results = {}

        logger.info("Initialized Real-World Data Validator")

    def load_real_data(
        self, data_source: str, file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load real-world transaction data.

        Args:
            data_source: Data source identifier ('csv', 'database', 'api')
            file_path: Path to data file (for CSV source)

        Returns:
            DataFrame with real transaction data
        """
        logger.info(f"Loading real-world data from {data_source}")

        if data_source == "csv" and file_path:
            return pd.read_csv(file_path)
        elif data_source == "database":
            # Database connection implementation
            return self._load_from_database()
        elif data_source == "api":
            # API integration implementation
            return self._load_from_api()
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

    def compare_distributions(
        self,
        synthetic_data: pd.DataFrame,
        real_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compare statistical distributions between synthetic and real data.

        Args:
            synthetic_data: Synthetic transaction dataset
            real_data: Real-world transaction dataset
            features: List of features to compare (None = all numeric)

        Returns:
            Comparison results with statistical tests
        """
        if features is None:
            features = synthetic_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        logger.info(f"Comparing distributions for {len(features)} features")

        comparison_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "features_compared": len(features),
            "feature_comparisons": {},
        }

        for feature in features:
            if feature in synthetic_data.columns and feature in real_data.columns:
                comparison = self._compare_feature_distribution(
                    synthetic_data[feature], real_data[feature], feature
                )
                comparison_results["feature_comparisons"][feature] = comparison

        # Calculate overall similarity score
        similarity_scores = [
            comp["similarity_score"]
            for comp in comparison_results["feature_comparisons"].values()
        ]
        comparison_results["overall_similarity"] = (
            np.mean(similarity_scores) if similarity_scores else 0.0
        )

        logger.info(
            f"Overall distribution similarity: {comparison_results['overall_similarity']:.3f}"
        )

        return comparison_results

    def _compare_feature_distribution(
        self, synthetic_values: pd.Series, real_values: pd.Series, feature_name: str
    ) -> Dict:
        """
        Compare distribution of a single feature.

        Uses Kolmogorov-Smirnov test and statistical moments.
        """
        from scipy import stats

        # Remove NaN values
        synthetic_clean = synthetic_values.dropna()
        real_clean = real_values.dropna()

        # Statistical moments
        synthetic_stats = {
            "mean": synthetic_clean.mean(),
            "std": synthetic_clean.std(),
            "median": synthetic_clean.median(),
            "min": synthetic_clean.min(),
            "max": synthetic_clean.max(),
        }

        real_stats = {
            "mean": real_clean.mean(),
            "std": real_clean.std(),
            "median": real_clean.median(),
            "min": real_clean.min(),
            "max": real_clean.max(),
        }

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(synthetic_clean, real_clean)

        # Calculate similarity score (inverse of KS statistic)
        similarity_score = 1 - ks_statistic

        return {
            "feature": feature_name,
            "synthetic_stats": synthetic_stats,
            "real_stats": real_stats,
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "similarity_score": similarity_score,
            "distributions_similar": ks_pvalue > 0.05,  # 5% significance level
        }

    def validate_model_performance(
        self,
        model,
        synthetic_test: Tuple[pd.DataFrame, np.ndarray],
        real_test: Tuple[pd.DataFrame, np.ndarray],
    ) -> Dict:
        """
        Validate model performance on both synthetic and real data.

        Args:
            model: Trained AML detection model
            synthetic_test: Tuple of (X_test, y_test) from synthetic data
            real_test: Tuple of (X_test, y_test) from real data

        Returns:
            Comprehensive performance comparison
        """
        logger.info("Validating model performance on synthetic vs real data")

        # Evaluate on synthetic data
        synthetic_metrics = self._evaluate_model(
            model, synthetic_test[0], synthetic_test[1], dataset_name="Synthetic"
        )

        # Evaluate on real data
        real_metrics = self._evaluate_model(
            model, real_test[0], real_test[1], dataset_name="Real-World"
        )

        # Compare performance
        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "synthetic_performance": synthetic_metrics,
            "real_world_performance": real_metrics,
            "performance_gaps": self._calculate_performance_gaps(
                synthetic_metrics, real_metrics
            ),
        }

        self.validation_results["model_performance"] = comparison

        return comparison

    def _evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: np.ndarray, dataset_name: str
    ) -> Dict:
        """
        Evaluate model and return comprehensive metrics.
        """
        logger.info(f"Evaluating model on {dataset_name} data...")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        metrics = {
            "dataset": dataset_name,
            "samples": len(y_test),
            "positive_samples": int(np.sum(y_test)),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "accuracy": np.mean(y_test == y_pred),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                metrics["avg_precision"] = average_precision_score(y_test, y_proba)
            except Exception:
                pass

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        logger.info(f"{dataset_name} F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"{dataset_name} Precision: {metrics['precision']:.4f}")
        logger.info(f"{dataset_name} Recall: {metrics['recall']:.4f}")

        return metrics

    def _calculate_performance_gaps(
        self, synthetic_metrics: Dict, real_metrics: Dict
    ) -> Dict:
        """
        Calculate performance gaps between synthetic and real data.
        """
        gaps = {}

        metric_keys = ["precision", "recall", "f1_score", "false_positive_rate"]

        for key in metric_keys:
            if key in synthetic_metrics and key in real_metrics:
                synthetic_val = synthetic_metrics[key]
                real_val = real_metrics[key]

                absolute_gap = real_val - synthetic_val
                relative_gap = (
                    (absolute_gap / synthetic_val * 100) if synthetic_val > 0 else 0
                )

                gaps[key] = {
                    "synthetic": synthetic_val,
                    "real_world": real_val,
                    "absolute_gap": absolute_gap,
                    "relative_gap_percent": relative_gap,
                }

        return gaps

    def generate_validation_report(self, output_path: str = "validation_report.json"):
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save report
        """
        logger.info(f"Generating validation report: {output_path}")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_type": "synthetic_vs_real",
            "results": self.validation_results,
            "summary": self._generate_summary(),
        }

        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

        return report

    def _generate_summary(self) -> Dict:
        """Generate executive summary of validation results."""
        summary = {
            "data_quality": "N/A",
            "model_performance": "N/A",
            "recommendations": [],
        }

        if "model_performance" in self.validation_results:
            perf = self.validation_results["model_performance"]
            gaps = perf.get("performance_gaps", {})

            # Check F1 gap
            if "f1_score" in gaps:
                f1_gap = abs(gaps["f1_score"]["relative_gap_percent"])

                if f1_gap < 5:
                    summary["model_performance"] = "Excellent"
                    summary["recommendations"].append(
                        "Model performs consistently on real-world data"
                    )
                elif f1_gap < 15:
                    summary["model_performance"] = "Good"
                    summary["recommendations"].append(
                        "Minor performance degradation on real data - monitor closely"
                    )
                else:
                    summary["model_performance"] = "Needs Improvement"
                    summary["recommendations"].append(
                        "Significant performance gap - consider retraining with real data"
                    )

        return summary

    def plot_performance_comparison(
        self, save_path: str = "performance_comparison.png"
    ):
        """
        Generate visualization comparing synthetic vs real performance.

        Args:
            save_path: Path to save plot
        """
        if "model_performance" not in self.validation_results:
            logger.warning("No performance data available for plotting")
            return

        perf = self.validation_results["model_performance"]
        synthetic = perf["synthetic_performance"]
        real_world = perf["real_world_performance"]

        # Metrics to plot
        metrics = ["precision", "recall", "f1_score", "false_positive_rate"]

        synthetic_vals = [synthetic.get(m, 0) for m in metrics]
        real_vals = [real_world.get(m, 0) for m in metrics]

        # Create plot
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            x - width / 2,
            synthetic_vals,
            width,
            label="Synthetic Data",
            color="#3498db",
        )
        ax.bar(
            x + width / 2, real_vals, width, label="Real-World Data", color="#e74c3c"
        )

        ax.set_ylabel("Score")
        ax.set_title("Model Performance: Synthetic vs Real-World Data")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance comparison plot saved to {save_path}")

        plt.close()

    def _load_from_database(self) -> pd.DataFrame:
        """
        Load data from database (placeholder for actual implementation).
        """
        logger.warning("Database loading not implemented - returning empty DataFrame")
        return pd.DataFrame()

    def _load_from_api(self) -> pd.DataFrame:
        """
        Load data from API (placeholder for actual implementation).
        """
        logger.warning("API loading not implemented - returning empty DataFrame")
        return pd.DataFrame()


class AnonymizedDataProcessor:
    """
    Processor for anonymized production data.
    Handles PII removal and data sanitization.
    """

    def __init__(self):
        """Initialize anonymized data processor."""

        self._analyzer = None
        self._anonymizer = None

        logger.info("Initialized Anonymized Data Processor")

    def _ensure_presidio(self):
        """Lazily import and initialise presidio engines."""
        if self._analyzer is None or self._anonymizer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
            except ImportError as exc:
                raise ImportError(
                    "The 'presidio-analyzer' and 'presidio-anonymizer' packages are "
                    "required for AnonymizedDataProcessor. "
                    "Install them with: pip install presidio-analyzer presidio-anonymizer"
                ) from exc

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

    def anonymize_dataset(
        self, data: pd.DataFrame, pii_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Anonymize a dataset by removing/masking PII.

        Args:
            data: Input DataFrame
            pii_columns: Columns containing PII (None = auto-detect)

        Returns:
            Anonymized DataFrame
        """
        # Presidio engines are created only when this method is first called.
        self._ensure_presidio()

        logger.info("Anonymizing dataset...")

        anonymized_data = data.copy()

        if pii_columns is None:
            # Auto-detect PII columns
            pii_columns = self._detect_pii_columns(data)

        logger.info(f"Anonymizing {len(pii_columns)} PII columns")

        for col in pii_columns:
            if col in anonymized_data.columns:
                anonymized_data[col] = anonymized_data[col].apply(self._anonymize_value)

        logger.info("Dataset anonymization complete")

        return anonymized_data

    def _detect_pii_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect columns containing PII."""
        pii_indicators = [
            "name",
            "email",
            "phone",
            "address",
            "ssn",
            "account",
            "card",
            "customer",
            "client",
        ]

        pii_columns = []
        for col in data.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in pii_indicators):
                pii_columns.append(col)

        return pii_columns

    def _anonymize_value(self, value):
        """Anonymize a single value."""
        if pd.isna(value):
            return value

        # Simple hash-based anonymization
        import hashlib

        str_value = str(value)
        hashed = hashlib.sha256(str_value.encode()).hexdigest()[:16]
        return f"ANON_{hashed}"
