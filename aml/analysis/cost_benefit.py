"""
Cost-Benefit Analysis Framework for AML Systems
Quantifies business impact of false positives vs false negatives
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class CostBenefitAnalyzer:
    """
    Analyzes cost-benefit trade-offs in AML detection systems.
    Quantifies business impact and optimal decision thresholds.
    """

    def __init__(self, cost_config: Optional[Dict] = None):
        """
        Initialize cost-benefit analyzer.

        Args:
            cost_config: Dict with cost parameters
        """
        # Default cost structure (USD)
        self.costs = cost_config or {
            # False Positive Costs
            "investigation_cost_per_fp": 250,  # Manual investigation
            "compliance_officer_hourly": 75,
            "avg_investigation_hours": 3,
            "customer_friction_cost": 100,  # Customer experience impact
            "false_sar_filing_cost": 500,  # Cost of filing unnecessary SAR
            # False Negative Costs
            "regulatory_fine_per_miss": 50000,  # Average regulatory penalty
            "reputational_damage": 100000,  # Brand/reputation impact
            "legal_costs": 25000,  # Legal defense costs
            "customer_loss": 10000,  # Lost customer value
            # True Positive Benefits
            "prevented_fraud_recovery": 0.3,  # 30% of fraud amount recovered
            "regulatory_compliance_value": 5000,  # Value of demonstrating compliance
            # Operational Costs
            "system_cost_per_transaction": 0.05,  # Processing cost
            "sar_filing_cost": 150,  # Cost to file legitimate SAR
        }

        # Risk appetite configuration
        self.risk_appetite = {
            "max_acceptable_fpr": 0.05,  # 5% false positive rate
            "min_required_recall": 0.85,  # 85% recall minimum
            "regulatory_tolerance": "moderate",  # low, moderate, high
        }

        logger.info("Initialized Cost-Benefit Analyzer")

    def calculate_costs(
        self,
        confusion_matrix: Dict[str, int],
        transaction_volumes: Dict[str, float],
        evaluation_period_days: int = 365,
    ) -> Dict:
        """
        Calculate comprehensive cost analysis.

        Args:
            confusion_matrix: Dict with TP, TN, FP, FN counts
            transaction_volumes: Dict with transaction statistics
            evaluation_period_days: Period for cost calculation

        Returns:
            Detailed cost breakdown
        """
        logger.info("Calculating cost-benefit analysis...")

        tp = confusion_matrix.get("true_positives", 0)
        tn = confusion_matrix.get("true_negatives", 0)
        fp = confusion_matrix.get("false_positives", 0)
        fn = confusion_matrix.get("false_negatives", 0)

        total_transactions = tp + tn + fp + fn
        avg_fraud_amount = transaction_volumes.get("avg_fraud_amount", 50000)

        # False Positive Costs
        fp_investigation_cost = fp * (
            self.costs["investigation_cost_per_fp"]
            + self.costs["compliance_officer_hourly"]
            * self.costs["avg_investigation_hours"]
        )
        fp_customer_friction = fp * self.costs["customer_friction_cost"]
        fp_false_sar_cost = fp * self.costs["false_sar_filing_cost"]
        total_fp_cost = fp_investigation_cost + fp_customer_friction + fp_false_sar_cost

        # False Negative Costs
        fn_regulatory_fines = fn * self.costs["regulatory_fine_per_miss"]
        fn_reputational = fn * self.costs["reputational_damage"]
        fn_legal_costs = fn * self.costs["legal_costs"]
        fn_fraud_loss = fn * avg_fraud_amount  # Undetected fraud
        total_fn_cost = (
            fn_regulatory_fines + fn_reputational + fn_legal_costs + fn_fraud_loss
        )

        # True Positive Benefits
        tp_fraud_prevented = (
            tp * avg_fraud_amount * self.costs["prevented_fraud_recovery"]
        )
        tp_compliance_value = tp * self.costs["regulatory_compliance_value"]
        total_tp_benefit = tp_fraud_prevented + tp_compliance_value

        # Operational Costs
        operational_cost = (
            total_transactions * self.costs["system_cost_per_transaction"]
        )
        sar_filing_cost = tp * self.costs["sar_filing_cost"]
        total_operational = operational_cost + sar_filing_cost

        # Net Analysis
        total_costs = total_fp_cost + total_fn_cost + total_operational
        net_benefit = total_tp_benefit - total_costs
        roi = (net_benefit / total_costs * 100) if total_costs > 0 else 0

        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation_period_days": evaluation_period_days,
            "transaction_metrics": {
                "total_transactions": total_transactions,
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            },
            "cost_breakdown": {
                "false_positive_costs": {
                    "investigation": fp_investigation_cost,
                    "customer_friction": fp_customer_friction,
                    "false_sar_filings": fp_false_sar_cost,
                    "total": total_fp_cost,
                    "per_fp": total_fp_cost / fp if fp > 0 else 0,
                },
                "false_negative_costs": {
                    "regulatory_fines": fn_regulatory_fines,
                    "reputational_damage": fn_reputational,
                    "legal_costs": fn_legal_costs,
                    "fraud_losses": fn_fraud_loss,
                    "total": total_fn_cost,
                    "per_fn": total_fn_cost / fn if fn > 0 else 0,
                },
                "operational_costs": {
                    "system_processing": operational_cost,
                    "sar_filing": sar_filing_cost,
                    "total": total_operational,
                },
            },
            "benefits": {
                "fraud_prevented": tp_fraud_prevented,
                "compliance_value": tp_compliance_value,
                "total": total_tp_benefit,
            },
            "summary": {
                "total_costs": total_costs,
                "total_benefits": total_tp_benefit,
                "net_benefit": net_benefit,
                "roi_percent": roi,
                "cost_per_transaction": (
                    total_costs / total_transactions if total_transactions > 0 else 0
                ),
            },
        }

        self._log_analysis_summary(analysis)

        return analysis

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        transaction_amounts: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Find optimal decision threshold that minimizes total cost.

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            transaction_amounts: Optional array of transaction amounts
            thresholds: Optional array of thresholds to test

        Returns:
            Optimal threshold and cost analysis
        """
        logger.info("Optimizing decision threshold for cost minimization...")

        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)

        if transaction_amounts is None:
            transaction_amounts = np.full(len(y_true), 50000)  # Default amount

        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate confusion matrix
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            cm = {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

            # Calculate costs
            avg_fraud = (
                float(np.mean(transaction_amounts[y_true == 1]))
                if np.sum(y_true) > 0
                else 50000
            )

            cost_analysis = self.calculate_costs(cm, {"avg_fraud_amount": avg_fraud})

            # Check if meets risk appetite
            meets_requirements = self._check_risk_appetite(cost_analysis)

            results.append(
                {
                    "threshold": threshold,
                    "confusion_matrix": cm,
                    "total_cost": cost_analysis["summary"]["total_costs"],
                    "net_benefit": cost_analysis["summary"]["net_benefit"],
                    "precision": cost_analysis["transaction_metrics"]["precision"],
                    "recall": cost_analysis["transaction_metrics"]["recall"],
                    "fpr": cost_analysis["transaction_metrics"]["fpr"],
                    "meets_risk_appetite": meets_requirements,
                }
            )

        # Find optimal threshold
        valid_results = [r for r in results if r["meets_risk_appetite"]]

        if not valid_results:
            logger.warning("No threshold meets risk appetite - using best available")
            valid_results = results

        # Optimize for maximum net benefit
        optimal = max(valid_results, key=lambda x: x["net_benefit"])

        logger.info(f"Optimal threshold: {optimal['threshold']:.3f}")
        logger.info(f"Net benefit: ${optimal['net_benefit']:,.2f}")
        logger.info(
            f"Precision: {optimal['precision']:.3f}, Recall: {optimal['recall']:.3f}"
        )

        return {
            "optimal_threshold": optimal["threshold"],
            "optimal_metrics": optimal,
            "all_thresholds": results,
            "recommendation": self._generate_threshold_recommendation(optimal),
        }

    def _check_risk_appetite(self, cost_analysis: Dict) -> bool:
        """
        Check if metrics meet organizational risk appetite.
        """
        metrics = cost_analysis["transaction_metrics"]

        # Check FPR constraint
        if metrics["fpr"] > self.risk_appetite["max_acceptable_fpr"]:
            return False

        # Check recall constraint
        if metrics["recall"] < self.risk_appetite["min_required_recall"]:
            return False

        return True

    def compare_scenarios(self, scenarios: List[Dict]) -> Dict:
        """
        Compare multiple detection scenarios.

        Args:
            scenarios: List of scenario dicts with confusion matrices

        Returns:
            Comparative analysis
        """
        logger.info(f"Comparing {len(scenarios)} scenarios...")

        scenario_analyses = []

        for i, scenario in enumerate(scenarios):
            analysis = self.calculate_costs(
                scenario["confusion_matrix"], scenario.get("transaction_volumes", {})
            )

            analysis["scenario_name"] = scenario.get("name", f"Scenario {i+1}")
            scenario_analyses.append(analysis)

        # Rank by net benefit
        ranked = sorted(
            scenario_analyses, key=lambda x: x["summary"]["net_benefit"], reverse=True
        )

        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "scenarios": ranked,
            "best_scenario": ranked[0]["scenario_name"],
            "worst_scenario": ranked[-1]["scenario_name"],
            "benefit_range": {
                "best": ranked[0]["summary"]["net_benefit"],
                "worst": ranked[-1]["summary"]["net_benefit"],
                "difference": ranked[0]["summary"]["net_benefit"]
                - ranked[-1]["summary"]["net_benefit"],
            },
        }

        return comparison

    def sensitivity_analysis(
        self,
        confusion_matrix: Dict,
        transaction_volumes: Dict,
        varying_parameter: str,
        value_range: Tuple[float, float],
        num_points: int = 20,
    ) -> Dict:
        """
        Perform sensitivity analysis on cost parameters.

        Args:
            confusion_matrix: Base confusion matrix
            transaction_volumes: Transaction statistics
            varying_parameter: Parameter to vary
            value_range: (min, max) for parameter
            num_points: Number of points to evaluate

        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis on '{varying_parameter}'")

        original_value = self.costs.get(varying_parameter)
        test_values = np.linspace(value_range[0], value_range[1], num_points)

        results = []

        for value in test_values:
            # Temporarily set parameter
            self.costs[varying_parameter] = value

            # Calculate costs
            analysis = self.calculate_costs(confusion_matrix, transaction_volumes)

            results.append(
                {
                    "parameter_value": value,
                    "total_cost": analysis["summary"]["total_costs"],
                    "net_benefit": analysis["summary"]["net_benefit"],
                    "roi": analysis["summary"]["roi_percent"],
                }
            )

        # Restore original value
        if original_value is not None:
            self.costs[varying_parameter] = original_value

        sensitivity = {
            "parameter": varying_parameter,
            "original_value": original_value,
            "test_range": value_range,
            "results": results,
            "elasticity": self._calculate_elasticity(results),
        }

        return sensitivity

    def _calculate_elasticity(self, results: List[Dict]) -> float:
        """Calculate elasticity of net benefit to parameter changes."""
        if len(results) < 2:
            return 0.0

        # Use first and last points
        param_change = (
            results[-1]["parameter_value"] - results[0]["parameter_value"]
        ) / results[0]["parameter_value"]
        benefit_change = (
            (results[-1]["net_benefit"] - results[0]["net_benefit"])
            / results[0]["net_benefit"]
            if results[0]["net_benefit"] != 0
            else 0
        )

        elasticity = benefit_change / param_change if param_change != 0 else 0

        return elasticity

    def _generate_threshold_recommendation(self, optimal_metrics: Dict) -> str:
        """Generate recommendation based on optimal threshold."""
        precision = optimal_metrics["precision"]
        recall = optimal_metrics["recall"]
        fpr = optimal_metrics["fpr"]

        if precision > 0.8 and recall > 0.85 and fpr < 0.03:
            return "Excellent balance - deploy with confidence"
        elif precision > 0.7 and recall > 0.80:
            return "Good balance - suitable for production with monitoring"
        elif precision < 0.6:
            return "High false positive rate - consider additional features or model tuning"
        elif recall < 0.75:
            return "High false negative rate - regulatory risk, increase sensitivity"
        else:
            return "Acceptable but monitor closely for drift"

    def _log_analysis_summary(self, analysis: Dict):
        """Log summary of cost analysis."""
        summary = analysis["summary"]

        logger.info("=" * 80)
        logger.info("COST-BENEFIT ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Costs: ${summary['total_costs']:,.2f}")
        logger.info(f"Total Benefits: ${summary['total_benefits']:,.2f}")
        logger.info(f"Net Benefit: ${summary['net_benefit']:,.2f}")
        logger.info(f"ROI: {summary['roi_percent']:.2f}%")
        logger.info(f"Cost per Transaction: ${summary['cost_per_transaction']:.4f}")
        logger.info("=" * 80)

    def export_analysis(self, analysis: Dict, output_path: str):
        """
        Export analysis to JSON file.

        Args:
            analysis: Analysis results dict
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Analysis exported to {output_path}")

    def update_cost_config(self, new_costs: Dict):
        """
        Update cost configuration.

        Args:
            new_costs: Dict with updated cost parameters
        """
        self.costs.update(new_costs)
        logger.info(f"Updated {len(new_costs)} cost parameters")

    def update_risk_appetite(self, new_appetite: Dict):
        """
        Update risk appetite configuration.

        Args:
            new_appetite: Dict with updated risk appetite parameters
        """
        self.risk_appetite.update(new_appetite)
        logger.info(f"Updated risk appetite: {new_appetite}")
