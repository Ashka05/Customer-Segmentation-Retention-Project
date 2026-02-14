# 🚀 Customer Retention Intelligence System

> Production-grade ML system achieving **97% validation success rate**
> and **120.3% ROI** on real e-commerce data

**Python 3.8+ \| LightGBM 3.3+ \| 163K Transactions \| 2,933 Customers**

------------------------------------------------------------------------

## 🎯 Problem

E-commerce businesses struggle to:

-   Identify customers at risk of churn\
-   Estimate Customer Lifetime Value (CLV) accurately\
-   Allocate marketing budgets efficiently\
-   Quantify ROI of retention campaigns

------------------------------------------------------------------------

## 💡 Solution

An end-to-end machine learning pipeline that:

-   Predicts churn with **94.5% recall**
-   Forecasts 180-day CLV with uncertainty intervals
-   Optimizes marketing budget using Expected Value maximization
-   Generates priority-ranked, actionable retention strategies

**Business Result:**\
💰 \$7,919 net profit from \$6,580 investment\
📈 120.3% validated ROI

------------------------------------------------------------------------

# 🏗 System Flow

    Raw Transactions (163K)
            ↓
    Data Cleaning & Validation
            ↓
    Feature Engineering (RFM + Behavioral)
            ↓
    ----------------------------------------
    | Segmentation | Churn | CLV Forecast |
    ----------------------------------------
            ↓
    Expected Value Optimization
            ↓
    Priority-Ranked Campaign Plan

------------------------------------------------------------------------

# 📊 Business Impact (Validated)

  Metric                           Result
  -------------------------------- -------------------
  Customers Modeled                2,933
  High-Risk Customers Identified   777
  Customers Targeted               273
  Customers Saved                  41
  Revenue Protected                \$14,499
  Validation Success               97% (34/35 tests)

### Financial Comparison

  Metric                Baseline    With ML
  --------------------- ----------- --------------
  Expected Churn Loss   \$24,165    \$9,666
  Campaign Cost         \$0         \$6,580
  Net Impact            -\$24,165   **+\$7,919**

**Total Value Swing: \$32,084**

------------------------------------------------------------------------

# 🧠 Core ML Components

## 1️⃣ Customer Segmentation

-   RFM Analysis (Recency, Frequency, Monetary)
-   K-Means with auto-K selection
-   4 business segments:
    -   Need Attention (56%)
    -   Hibernating (32%)
    -   Promising (12%)
    -   Lost (0.4%)

------------------------------------------------------------------------

## 2️⃣ Churn Prediction

-   LightGBM classifier
-   ROC AUC: **0.767**
-   Recall: **94.5%**
-   Precision: \~40%
-   Class imbalance handled using `scale_pos_weight`
-   Threshold optimized using F-beta (β = 2)

Risk Levels: - Low: 544\
- Medium: 274\
- High: 342\
- Very High: 777

------------------------------------------------------------------------

## 3️⃣ CLV Forecasting

-   180-day revenue prediction
-   Mean CLV: **\$181.58**
-   Quantile regression (10th, 50th, 90th percentiles)
-   R² = **0.491**
-   Portfolio-level error: \~2--3%
-   Risk-adjusted CLV integrates churn probability

------------------------------------------------------------------------

## 4️⃣ Revenue Optimization Engine

Campaign tiers:

-   Premium (\$50)
-   Standard (\$25)
-   Light (\$10)

Achieved ROI: **120.3%**

------------------------------------------------------------------------

# ⚙️ Technical Highlights

### Production Engineering

-   150+ YAML configuration parameters\
-   Zero data leakage (temporal validation)\
-   Modular architecture\
-   Pickle model persistence\
-   Logging & validation framework\
-   9,000+ lines of structured production code

### Data Science Best Practices

-   Feature engineering at scale\
-   Temporal train/validation/test splits\
-   Uncertainty quantification\
-   Business-metric optimization\
-   Portfolio-level performance focus

### Scalability

-   163K transactions processed in \~20 minutes\
-   Tested scaling to 100K customers (\~2 hours)\
-   Vectorized Pandas/Numpy with Parquet optimization

------------------------------------------------------------------------

# 📌 Key Insight

**Behavior → Features → ML Models → Risk/CLV → Optimization → Business
Action**

This system bridges predictive modeling and measurable financial impact,
transforming machine learning into operational decision-making.
