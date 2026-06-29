"""
Interactive Explainability Dashboard for AML Investigations
Web-based dashboard for SAR generation reasoning and evidence trails
"""

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd

try:
    import plotly
    import plotly.graph_objs as go
except ImportError:  # pragma: no cover - optional dependency
    plotly = None
    go = None
from flask import Flask, jsonify, render_template, request
from loguru import logger


class ExplainabilityDashboard:
    """
    Web-based investigator dashboard for AML explainability.
    Shows SAR generation reasoning, attention weights, and evidence trails.
    """

    def __init__(self, port: int = 5001):
        """
        Initialize explainability dashboard.

        Args:
            port: Port to run dashboard server
        """
        self.app = Flask(__name__)
        self.port = port
        self.sar_database = {}  # In-memory SAR storage
        self.model_explainer = None

        # Register routes
        self._register_routes()

        logger.info(f"Initialized Explainability Dashboard on port {port}")

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route("/")
        def index():
            """Main dashboard page."""
            return render_template("dashboard.html")

        @self.app.route("/api/sars")
        def get_sars():
            """Get list of all SARs."""
            sars = [
                {
                    "sar_id": sar_id,
                    "entity_id": sar_data.get("entity_id"),
                    "risk_score": sar_data.get("risk_score", 0.0),
                    "timestamp": sar_data.get("timestamp"),
                    "status": sar_data.get("status", "pending"),
                }
                for sar_id, sar_data in self.sar_database.items()
            ]
            return jsonify({"sars": sars})

        @self.app.route("/api/sar/<sar_id>")
        def get_sar_details(sar_id):
            """Get detailed SAR information."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            sar = self.sar_database[sar_id]

            # Generate explainability data
            explanation = self.explain_sar(sar)

            return jsonify({"sar": sar, "explanation": explanation})

        @self.app.route("/api/sar/<sar_id>/evidence")
        def get_sar_evidence(sar_id):
            """Get evidence trail for SAR."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            sar = self.sar_database[sar_id]
            evidence = self.extract_evidence_trail(sar)

            return jsonify({"evidence": evidence})

        @self.app.route("/api/sar/<sar_id>/feature_importance")
        def get_feature_importance(sar_id):
            """Get feature importance visualization."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            sar = self.sar_database[sar_id]
            fig = self.create_feature_importance_plot(sar)

            return jsonify({"plot": fig})

        @self.app.route("/api/sar/<sar_id>/timeline")
        def get_sar_timeline(sar_id):
            """Get transaction timeline visualization."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            sar = self.sar_database[sar_id]
            fig = self.create_timeline_plot(sar)

            return jsonify({"plot": fig})

        @self.app.route("/api/sar/<sar_id>/network")
        def get_entity_network(sar_id):
            """Get entity network graph."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            sar = self.sar_database[sar_id]
            fig = self.create_network_graph(sar)

            return jsonify({"plot": fig})

        @self.app.route("/api/sar/<sar_id>/approve", methods=["POST"])
        def approve_sar(sar_id):
            """Approve SAR for filing."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            data = request.get_json() or {}
            investigator = data.get("investigator", "Unknown")
            notes = data.get("notes", "")

            self.sar_database[sar_id]["status"] = "approved"
            self.sar_database[sar_id]["approval_data"] = {
                "investigator": investigator,
                "timestamp": datetime.utcnow().isoformat(),
                "notes": notes,
            }

            logger.info(f"SAR {sar_id} approved by {investigator}")

            return jsonify({"success": True})

        @self.app.route("/api/sar/<sar_id>/reject", methods=["POST"])
        def reject_sar(sar_id):
            """Reject SAR."""
            if sar_id not in self.sar_database:
                return jsonify({"error": "SAR not found"}), 404

            data = request.get_json() or {}
            investigator = data.get("investigator", "Unknown")
            reason = data.get("reason", "")

            self.sar_database[sar_id]["status"] = "rejected"
            self.sar_database[sar_id]["rejection_data"] = {
                "investigator": investigator,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
            }

            logger.info(f"SAR {sar_id} rejected by {investigator}")

            return jsonify({"success": True})

    def add_sar(self, sar_data: Dict):
        """
        Add SAR to dashboard.

        Args:
            sar_data: Dict with SAR information
        """
        sar_id = sar_data.get("sar_id", f"SAR_{len(self.sar_database)+1}")
        self.sar_database[sar_id] = sar_data
        logger.debug(f"Added SAR {sar_id} to dashboard")

    def explain_sar(self, sar: Dict) -> Dict:
        """
        Generate comprehensive explanation for SAR.

        Args:
            sar: SAR data dict

        Returns:
            Explanation dict with reasoning
        """
        explanation = {
            "summary": self._generate_summary(sar),
            "key_factors": self._extract_key_factors(sar),
            "decision_path": self._trace_decision_path(sar),
            "attention_weights": self._calculate_attention_weights(sar),
            "comparable_cases": self._find_comparable_cases(sar),
        }

        return explanation

    def _generate_summary(self, sar: Dict) -> str:
        """Generate human-readable summary."""
        entity_id = sar.get("entity_id", "Unknown")
        risk_score = sar.get("risk_score", 0)
        crime_type = sar.get("crime_type", "Unknown")

        summary = f"Entity {entity_id} flagged with {risk_score:.1%} suspicion score. "
        summary += f"Primary concern: {crime_type}. "

        num_transactions = len(sar.get("transactions", []))
        if num_transactions > 0:
            total_amount = sum(t.get("amount", 0) for t in sar["transactions"])
            summary += f"Based on {num_transactions} transactions totaling ${total_amount:,.2f}."

        return summary

    def _extract_key_factors(self, sar: Dict) -> List[Dict]:
        """Extract key contributing factors."""
        factors = []

        # Feature importance
        if "feature_importance" in sar:
            for feature, importance in sorted(
                sar["feature_importance"].items(), key=lambda x: x[1], reverse=True
            )[:5]:
                factors.append(
                    {
                        "type": "feature",
                        "name": feature,
                        "importance": importance,
                        "value": sar.get("features", {}).get(feature, "N/A"),
                    }
                )

        # Pattern matches
        if "patterns_matched" in sar:
            for pattern in sar["patterns_matched"]:
                factors.append(
                    {
                        "type": "pattern",
                        "name": pattern["name"],
                        "confidence": pattern["confidence"],
                        "description": pattern.get("description", ""),
                    }
                )

        return factors

    def _trace_decision_path(self, sar: Dict) -> List[Dict]:
        """Trace the decision-making path."""
        path = []

        # Step 1: Initial screening
        path.append(
            {
                "step": "Initial Screening",
                "result": "Flagged",
                "details": f"Risk score: {sar.get('risk_score', 0):.3f}",
                "timestamp": sar.get("timestamp", ""),
            }
        )

        # Step 2: Feature extraction
        if "features" in sar:
            path.append(
                {
                    "step": "Feature Extraction",
                    "result": "Complete",
                    "details": f"{len(sar['features'])} features computed",
                    "timestamp": "",
                }
            )

        # Step 3: Classification
        if "crime_type" in sar:
            path.append(
                {
                    "step": "Crime Classification",
                    "result": sar["crime_type"],
                    "details": f"Confidence: {sar.get('classification_confidence', 0):.2f}",
                    "timestamp": "",
                }
            )

        # Step 4: External intelligence
        if "sanctions_match" in sar or "pep_match" in sar:
            matches = []
            if sar.get("sanctions_match"):
                matches.append("Sanctions list")
            if sar.get("pep_match"):
                matches.append("PEP list")

            path.append(
                {
                    "step": "External Intelligence",
                    "result": "Matches Found" if matches else "No Matches",
                    "details": ", ".join(matches) if matches else "Clear",
                    "timestamp": "",
                }
            )

        # Step 5: Narrative generation
        if "narrative" in sar:
            path.append(
                {
                    "step": "Narrative Generation",
                    "result": "Complete",
                    "details": f"{len(sar['narrative'])} characters",
                    "timestamp": "",
                }
            )

        # Step 6: Validation
        if "validation_score" in sar:
            path.append(
                {
                    "step": "Agent-as-Judge Validation",
                    "result": "Passed" if sar["validation_score"] > 0.7 else "Flagged",
                    "details": f"Quality score: {sar['validation_score']:.2f}",
                    "timestamp": "",
                }
            )

        return path

    def _calculate_attention_weights(self, sar: Dict) -> Dict:
        """Calculate attention weights for evidence components."""
        weights = {}

        transactions = sar.get("transactions", [])
        if transactions:
            for i, txn in enumerate(transactions):
                txn_id = txn.get("transaction_id", f"txn_{i}")
                amount = txn.get("amount", 0)
                risk = txn.get("risk_score", 0)

                attention = (amount / 1000) * risk
                weights[txn_id] = float(attention)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _find_comparable_cases(self, sar: Dict) -> List[Dict]:
        """Find similar historical cases."""
        comparable = []

        current_crime_type = sar.get("crime_type")
        current_risk = sar.get("risk_score", 0)

        for other_id, other_sar in self.sar_database.items():
            if other_id == sar.get("sar_id"):
                continue

            if other_sar.get("crime_type") == current_crime_type:
                risk_diff = abs(other_sar.get("risk_score", 0) - current_risk)

                if risk_diff < 0.1:
                    comparable.append(
                        {
                            "sar_id": other_id,
                            "entity_id": other_sar.get("entity_id"),
                            "risk_score": other_sar.get("risk_score"),
                            "similarity": 1 - risk_diff,
                        }
                    )

        comparable.sort(key=lambda x: x["similarity"], reverse=True)
        return comparable[:5]

    def extract_evidence_trail(self, sar: Dict) -> Dict:
        """
        Extract complete evidence trail with citations.

        Args:
            sar: SAR data dict

        Returns:
            Evidence trail dict
        """
        evidence = {
            "transactions": [],
            "external_data": [],
            "behavioral_patterns": [],
            "citations": [],
        }

        for txn in sar.get("transactions", []):
            evidence["transactions"].append(
                {
                    "id": txn.get("transaction_id"),
                    "amount": txn.get("amount"),
                    "timestamp": txn.get("timestamp"),
                    "sender": txn.get("sender_id"),
                    "receiver": txn.get("receiver_id"),
                    "risk_indicators": txn.get("risk_indicators", []),
                }
            )

        if sar.get("sanctions_match"):
            evidence["external_data"].append(
                {
                    "type": "Sanctions List Match",
                    "details": sar["sanctions_match"],
                    "source": "OFAC/UN/EU Sanctions Lists",
                }
            )

        if sar.get("pep_match"):
            evidence["external_data"].append(
                {
                    "type": "PEP Match",
                    "details": sar["pep_match"],
                    "source": "PEP Database",
                }
            )

        if "patterns_matched" in sar:
            for pattern in sar["patterns_matched"]:
                evidence["behavioral_patterns"].append(
                    {
                        "pattern": pattern["name"],
                        "confidence": pattern["confidence"],
                        "description": pattern.get("description", ""),
                    }
                )

        if "narrative" in sar:
            import re

            citations = re.findall(r"\[([^\]]+)\]", sar["narrative"])
            evidence["citations"] = citations

        return evidence

    def create_feature_importance_plot(self, sar: Dict) -> str:
        """
        Create interactive feature importance plot.

        Args:
            sar: SAR data dict

        Returns:
            JSON string of Plotly figure
        """
        if "feature_importance" not in sar:
            return json.dumps({})

        features = sorted(
            sar["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )[:15]

        feature_names = [f[0] for f in features]
        importances = [f[1] for f in features]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation="h",
                    marker=dict(
                        color=importances, colorscale="RdYlGn_r", showscale=True
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_timeline_plot(self, sar: Dict) -> str:
        """
        Create transaction timeline visualization.

        Args:
            sar: SAR data dict

        Returns:
            JSON string of Plotly figure
        """
        transactions = sar.get("transactions", [])

        if not transactions:
            return json.dumps({})

        df = pd.DataFrame(transactions)

        if "timestamp" in df.columns and "amount" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            risk_scores = (
                df["risk_score"].tolist()
                if "risk_score" in df.columns
                else [0.5] * len(df)
            )
            marker_sizes = [r * 20 for r in risk_scores]

            text_labels = (
                df["transaction_id"].tolist()
                if "transaction_id" in df.columns
                else df.index.tolist()
            )

            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["amount"],
                        mode="markers+lines",
                        marker=dict(
                            size=marker_sizes,
                            color=risk_scores,
                            colorscale="RdYlGn_r",
                            showscale=True,
                            colorbar=dict(title="Risk Score"),
                        ),
                        text=text_labels,
                        hovertemplate="<b>%{text}</b><br>Amount: $%{y:,.2f}<br>Date: %{x}<extra></extra>",
                    )
                ]
            )

            fig.update_layout(
                title="Transaction Timeline",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                height=400,
            )

            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return json.dumps({})

    def create_network_graph(self, sar: Dict) -> str:
        """
        Create entity network graph.

        Args:
            sar: SAR data dict

        Returns:
            JSON string of Plotly figure
        """
        transactions = sar.get("transactions", [])

        if not transactions:
            return json.dumps({})

        nodes = set()
        edges = []

        for txn in transactions:
            sender = txn.get("sender_id", "Unknown")
            receiver = txn.get("receiver_id", "Unknown")

            nodes.add(sender)
            nodes.add(receiver)

            edges.append(
                {"source": sender, "target": receiver, "amount": txn.get("amount", 0)}
            )

        node_list = list(nodes)
        node_positions = {node: (i % 5, i // 5) for i, node in enumerate(node_list)}

        edge_traces = []
        for edge in edges:
            x0, y0 = node_positions[edge["source"]]
            x1, y1 = node_positions[edge["target"]]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=edge["amount"] / 1000, color="#888"),
                hoverinfo="none",
            )
            edge_traces.append(edge_trace)

        node_trace = go.Scatter(
            x=[node_positions[node][0] for node in node_list],
            y=[node_positions[node][1] for node in node_list],
            mode="markers+text",
            marker=dict(size=20, color="lightblue"),
            text=node_list,
            textposition="top center",
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Entity Network", showlegend=False, hovermode="closest", height=500
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def run(self, debug: bool = False):
        """
        Run the dashboard server.

        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting Explainability Dashboard on port {self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=debug)
