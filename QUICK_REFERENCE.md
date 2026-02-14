# Quick Reference Guide - Pillar 0

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline (fast mode)
python src/pillar_0_main.py --raw-data data/raw/online_retail.csv --single-snapshot

# 3. Check results
ls -lh data/features/customer_features.parquet
```

---

## 📁 File Structure (What's Where)

```
src/utils/utils.py          → Configuration loader, logging, utilities
src/data/validation.py      → 8-category data validation
src/data/cleaning.py        → 5-stage cleaning pipeline
src/data/feature_engineering.py → 33 features with temporal consistency
src/pillar_0_main.py        → Main orchestration script

config/config.yaml          → ALL hyperparameters (150+)

examples/pillar_0_examples.py → 6 usage examples
```

---

## 🎯 Command Line Usage

```bash
# Full pipeline (all snapshots)
python src/pillar_0_main.py --raw-data data/raw/online_retail.csv

# Single snapshot (fast for testing)
python src/pillar_0_main.py --raw-data data/raw/online_retail.csv --single-snapshot

# Strict mode (stop on validation errors)
python src/pillar_0_main.py --raw-data data/raw/online_retail.csv --strict

# Custom config
python src/pillar_0_main.py --raw-data data.csv --config my_config.yaml
```

---

## 📊 Key Configuration Parameters

```yaml
# Most important settings in config.yaml:

data_cleaning.filters.min_customer_transactions: 2
  → Minimum transactions per customer

churn.churn_window_days: 90
  → Days of inactivity = churned

churn.observation_window_days: 180
  → Days of history for features

feature_engineering.time_windows.short: 30
  → Recent activity window

paths.raw_data: "data/raw/online_retail.csv"
  → Input data location
```

---

## 🔍 How to Check Results

```python
import pandas as pd

# Load validation report
validation = pd.read_csv('outputs/validation_report.csv')
print(validation[validation['Status'] == 'FAIL'])

# Load cleaned data
df_clean = pd.read_parquet('data/processed/cleaned_data.parquet')
print(f"Rows: {len(df_clean):,}")
print(f"Customers: {df_clean['CustomerID'].nunique():,}")

# Load features
features = pd.read_parquet('data/features/customer_features.parquet')
print(f"Feature rows: {len(features):,}")
print(f"Features: {len(features.columns)}")
print(features.head())
```

---

## 🎓 Common Tasks

### Task: Get features for latest snapshot
```python
features = pd.read_parquet('data/features/customer_features.parquet')
latest = features[features['snapshot_date'] == features['snapshot_date'].max()]
```

### Task: Check data quality
```python
validation_report = pd.read_csv('outputs/validation_report.csv')
print(validation_report.to_string(index=False))
```

### Task: Get RFM summary statistics
```python
rfm_cols = ['rfm_recency_days', 'rfm_frequency_count', 'rfm_monetary_avg']
print(latest[rfm_cols].describe())
```

### Task: Find top customers by engagement
```python
top_10 = latest.nlargest(10, 'engagement_score')
print(top_10[['engagement_score', 'rfm_frequency_count', 'rfm_monetary_total']])
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run from project root: `cd retention_system` |
| "File not found" | Check path in config.yaml or use absolute path |
| "Out of memory" | Use `--single-snapshot` flag |
| "Date parsing failed" | Update `date_format` in config.yaml |
| "No snapshots generated" | Dataset too short (need 270+ days) |

---

## 📈 Performance Expectations

| Dataset Size | Time | Memory |
|--------------|------|--------|
| 100K rows | 1-2 min | 500 MB |
| 500K rows (UCI) | 4 min | 2 GB |
| 1M rows | 10 min | 4 GB |
| 10M rows | 1 hour | 16 GB |

---

## 🎯 Feature Cheat Sheet

### RFM Features (7)
- `rfm_recency_days` - Days since last purchase
- `rfm_frequency_count` - Total transactions
- `rfm_monetary_avg` - Average order value
- `rfm_monetary_total` - Total revenue
- `rfm_monetary_std` - Revenue volatility
- `days_since_first_purchase` - Customer age
- `customer_lifetime_days` - Active period

### Temporal Features (13)
- `txn_count_short/medium/long_30/60/90d` - Transaction counts in windows
- `revenue_short/medium/long_30/60/90d` - Revenue in windows
- `interpurchase_time_mean/std/median/cv` - Purchase patterns
- `purchase_frequency_per_month` - Monthly transaction rate

### Product Features (5)
- `unique_products_count` - Product diversity
- `avg_basket_size` - Items per transaction
- `std_basket_size` - Basket variability
- `return_rate` - Percentage returns
- `product_diversity_hhi` - Category diversity

### Engagement Features (4)
- `engagement_score` - (Frequency × Monetary) / Recency
- `engagement_score_normalized` - 0-1 scale
- `consistency_score` - Purchase regularity
- `activity_score` - Recent vs total activity

### Trend Features (4)
- `revenue_trend_slope` - Monthly revenue slope
- `transaction_trend_slope` - Monthly transaction slope
- `revenue_trend_direction` - +1 (growing) / -1 (declining) / 0 (flat)
- `transaction_trend_direction` - Direction indicator

---

## 🔄 Typical Workflow

1. **Initial Run** (4 min)
   ```bash
   python src/pillar_0_main.py --raw-data data.csv
   ```

2. **Inspect Results** (2 min)
   ```bash
   python examples/pillar_0_examples.py --example 3
   ```

3. **Adjust Config** (1 min)
   ```bash
   nano config/config.yaml
   # Change churn_window_days from 90 to 60
   ```

4. **Re-run** (4 min)
   ```bash
   python src/pillar_0_main.py --raw-data data.csv
   ```

5. **Compare Results** (2 min)
   ```python
   # Compare feature distributions
   ```

---

## 📞 Help & Support

**Check logs first:**
```bash
tail -100 logs/retention_system.log
```

**Common log patterns:**
- "✓" = Success
- "⚠️" = Warning (non-critical)
- "❌" = Error (needs attention)

**Debug mode:**
Edit config.yaml:
```yaml
logging.level: "DEBUG"
```

---

## 🎤 Interview Tips

When explaining this system:

1. **Start with the problem**: "Customer retention is critical for e-commerce profitability"

2. **Highlight temporal consistency**: "I implemented point-in-time correctness to prevent data leakage"

3. **Show code quality**: "All parameters are configurable, comprehensive logging, type hints throughout"

4. **Connect to business**: "These features enable churn prediction that can recover $X in revenue"

5. **Demonstrate scalability**: "This scales to 10M+ rows using Parquet and vectorized operations"

---

## ✅ Pre-Deploy Checklist

Before deploying to production:

- [ ] Run full validation suite
- [ ] Check all config parameters
- [ ] Verify temporal consistency
- [ ] Test on production-sized sample
- [ ] Set up monitoring dashboards
- [ ] Configure alerting thresholds
- [ ] Document parameter choices
- [ ] Create runbook for operations

---

## 📚 Further Reading

**In this repo:**
- PILLAR_0_README.md - Detailed documentation (800 lines)
- PROJECT_STRUCTURE.md - System overview (400 lines)
- PILLAR_0_COMPLETION_SUMMARY.md - What was built

**External:**
- UCI Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail
- RFM Analysis: Wikipedia RFM (recency, frequency, monetary_value)
- Point-in-Time Features: Google "time travel in ML feature engineering"

---

**Quick Reference Version: 1.0**
**Last Updated: 2026-02-04**
**Status: Pillar 0 Complete (6/6 tasks)**
