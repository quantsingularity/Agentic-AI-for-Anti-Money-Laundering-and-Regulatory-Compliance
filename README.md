# Agentic AI for Anti-Money Laundering and Regulatory Compliance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Production Ready](https://img.shields.io/badge/production-ready-success)](README.md)

A multi-agent system that automates end-to-end Suspicious Activity Report (SAR) generation for AML and regulatory compliance. Built for production scale with adversarial robustness testing, real-time monitoring, and a Human-in-the-Loop approval workflow.

---

## Table of Contents

- [Overview](#overview)
- [Agentic SAR Pipeline](#agentic-sar-pipeline)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [Security and Compliance](#security-and-compliance)
- [Testing](#testing)
- [License](#license)

---

## Overview

| Feature                      | Description                                                                                               |
| :--------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **Agentic SAR Workflow**     | Multi-agent orchestration for evidence aggregation, narrative generation, and agent-as-judge validation   |
| **Scalability**              | Apache Kafka for distributed transaction streaming, Redis caching — supports 10M+ transactions/day        |
| **Adversarial Robustness**   | Simulates 10 evasion techniques (structuring, layering) to harden the model against sophisticated attacks |
| **Production Monitoring**    | MLflow for experiment tracking, Prometheus and Grafana for real-time health and drift detection           |
| **Cost-Benefit Analysis**    | Optimizes detection threshold to maximize net financial benefit by quantifying FP and FN costs            |
| **Explainability Dashboard** | Web-based investigator interface with SAR reasoning, feature importance, and HIL approval workflow        |
| **Real Data Validation**     | Kolmogorov-Smirnov statistical comparison between synthetic and real data with PII anonymization          |

---

## Agentic SAR Pipeline

The `Orchestrator` (`aml/agents/orchestrator.py`) coordinates eight specialized agents to process transactions and produce a final SAR.

| Step                              | Agent               | Function                                                                               |
| :-------------------------------- | :------------------ | :------------------------------------------------------------------------------------- |
| 1. Ingest and Feature Engineering | Feature Engineer    | Calculates velocity, deviation, and other complex features from raw transactions       |
| 2. Privacy Guard                  | `PrivacyGuard`      | Redacts PII using Presidio before any further processing                               |
| 3. Crime Classification           | `Classifier`        | Scores transaction suspicion probability using a trained XGBoost model                 |
| 4. External Intelligence          | Intelligence Agent  | Checks sanctions lists and PEP databases for involved entities                         |
| 5. Evidence Aggregation           | Evidence Aggregator | Collects suspicious transactions, feature importance, and intelligence hits            |
| 6. Narrative Generation           | `NarrativeAgent`    | Uses an LLM to generate a compliant, well-cited SAR narrative from aggregated evidence |
| 7. Agent-as-Judge Validation      | Judge Agent         | Independent agent reviews narrative and evidence for completeness and compliance       |
| 8. Human-in-Loop Gating           | Orchestrator        | Flags high-risk SARs for mandatory human review via the Explainability Dashboard       |

---

## Repository Structure

| Path               | Description                                              |
| :----------------- | :------------------------------------------------------- |
| `aml/agents/`      | Orchestrator, narrative agent, privacy guard, base agent |
| `aml/models/`      | XGBoost classifier training and inference                |
| `aml/analysis/`    | Cost-benefit engine and threshold optimization           |
| `aml/dashboard/`   | Flask explainability dashboard and HTML templates        |
| `aml/streaming/`   | Kafka consumer for real-time transaction ingestion       |
| `aml/caching/`     | Redis caching for entity profiles and risk scores        |
| `aml/monitoring/`  | MLflow experiment tracking and model versioning          |
| `aml/adversarial/` | Evasion technique simulation and robustness testing      |
| `aml/validation/`  | Data distribution comparison and real-data validation    |
| `aml/scripts/`     | System runner, ablation studies, benchmarking            |
| `monitoring/`      | Prometheus and Grafana configuration files               |
| `data/`            | Synthetic and sample data for testing                    |
| `aml/tests/`       | Unit and integration tests                               |

---

## Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.10+
- OpenAI API key (optional, for LLM narrative generation)

### Full Stack

```bash
git clone https://github.com/quantsingularity/Agentic-AI-for-Anti-Money-Laundering-and-Regulatory-Compliance.git
cd Agentic-AI-for-Anti-Money-Laundering-and-Regulatory-Compliance

export OPENAI_API_KEY="sk-..."
./run_full.sh
```

| Service                  | URL                     |
| :----------------------- | :---------------------- |
| Explainability Dashboard | `http://localhost:5002` |
| MLflow Tracking          | `http://localhost:5001` |
| Grafana (admin / admin)  | `http://localhost:3000` |
| Prometheus               | `http://localhost:9090` |

### Standalone Demo

```bash
pip install -r requirements.txt
# Ensure Redis is running locally
./run_quick.sh
```

---

## Performance

| Metric                  | Baseline    | Enhanced      | Change      |
| :---------------------- | :---------- | :------------ | :---------- |
| Throughput              | 1K txns/min | 10K+ txns/min | +10x        |
| Latency (P95)           | 2.5s        | 250ms         | +10x faster |
| Cache Hit Rate          | N/A         | 89%           | New         |
| Detection Rate (Recall) | 86.9%       | 87.2%         | +0.3%       |
| False Positive Rate     | 2.3%        | 1.8%          | -22%        |
| Adversarial Robustness  | Untested    | 76.3%         | New         |
| Explainability Score    | 3.2/5       | 4.7/5         | +47%        |

---

## Security and Compliance

- PII anonymization applied before any model processing
- Redis cache configured for encrypted storage
- Full audit logging via MLflow for all model decisions
- Data handling aligned with GDPR requirements

---

## Testing

| Type                   | Command                                        |
| :--------------------- | :--------------------------------------------- |
| Unit tests             | `pytest aml/tests/`                            |
| Integration tests      | `pytest aml/tests/test_integration.py`         |
| Adversarial tests      | `python aml/adversarial/adversarial_tester.py` |
| Performance benchmarks | `python aml/scripts/benchmark_system.py`       |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
