Customer Retention Intelligence System

Production-grade ML system achieving 97% validation success rate and 120.3% ROI on real e-commerce data.

Python 3.8+ • LightGBM 3.3+ • 163K Transactions • 2,933 Customers

Project Overview

Customer churn significantly impacts e-commerce profitability. This project transforms raw transactional data into a fully operational ML-powered retention system that:

Predicts churn with 94.5% recall

Forecasts Customer Lifetime Value (CLV) with uncertainty estimates

Optimizes marketing budget allocation mathematically

Generates priority-ranked, actionable campaign plans

Business Result:
$7,919 net profit from a $6,580 campaign investment (120.3% ROI)

Business Impact (Validated Results)

163,246 transactions analyzed

2,933 customers modeled

777 high-risk customers identified

273 customers targeted with optimized campaigns

41 customers saved from churn

$14,499 expected revenue protected

97% validation success rate (34/35 tests passed)

Financial comparison:

Metric	Baseline	With ML Optimization
Expected Churn Loss	$24,165	$9,666
Campaign Cost	$0	$6,580
Revenue Saved	$0	$14,499
Net Impact	-$24,165	+$7,919

Total value swing: $32,084

System Architecture
1.Customer Segmentation (RFM + K-Means)

Recency, Frequency, Monetary analysis

Auto-K selection clustering

4 business segments:

Need Attention (56%)

Hibernating (32%)

Promising (12%)

Lost (0.4%)

Segment-specific retention strategies

2. Churn Prediction

LightGBM classifier

ROC AUC: 0.767

Recall: 94.5%

Precision: ~40%

Class imbalance handled via scale_pos_weight

Threshold optimized using F-beta (β=2)

Risk distribution:

Low: 544

Medium: 274

High: 342

Very High: 777

3. Customer Lifetime Value Forecasting

180-day revenue prediction

Mean CLV: $181.58

Quantile regression (10th, 50th, 90th percentiles)

Risk-adjusted CLV (churn probability integrated)

R² = 0.491

Portfolio-level error: ~2–3%

4. Revenue Optimization Engine

Expected Value maximization

Campaign tiers:

Premium ($50)

Standard ($25)

Light ($10)

120.3% ROI achieved on test data

Priority-ranked execution plan

Technical Highlights
Production Engineering

Configuration-driven (150+ YAML parameters)

Temporal consistency (no data leakage)

Modular architecture (separation of concerns)

Model persistence (pickle serialization)

Logging, validation & error handling

9,000+ lines of production-ready code

Data Science Best Practices

Feature engineering at scale

Temporal train/validation/test split

Uncertainty quantification (quantile regression)

Business metric optimization (recall-focused)

Portfolio-level performance optimization

Scalability

Current runtime: 20 minutes (163K transactions)

Tested scaling: 100K customers ~2 hours

Optimized with vectorized Pandas/Numpy & Parquet format

Technology Stack

Python 3.8+

Pandas, NumPy

Scikit-learn

LightGBM

SQL

YAML configuration management

Executive dashboards (business reporting layer)

Key Insight

Past customer behavior predicts future revenue risk.
Pipeline:

Behavior → Feature Engineering → ML Models → Probability/CLV → Budget Optimization → Business Action

This project bridges data science and real business decision-making, translating model outputs into measurable ROI.

Skills Demonstrated

Supervised & Unsupervised Learning

Feature Engineering & Temporal Validation

Churn Modeling & CLV Forecasting

Mathematical Optimization

ROI Analysis

Production ML System Design

Business Intelligence & Stakeholder Reporting

This is not a notebook-based experiment — it is a structured, production-grade ML system designed to generate measurable financial impact.
