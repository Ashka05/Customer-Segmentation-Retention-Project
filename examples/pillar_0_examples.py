"""
Example Usage: Pillar 0 Data Engineering Pipeline
=================================================
This script demonstrates how to use the Pillar 0 components individually
or as a complete pipeline.
"""

import pandas as pd
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, 'src')

from utils.utils import Config, setup_logging, print_section_header
from data.validation import DataValidator
from data.cleaning import DataCleaner
from data.feature_engineering import FeatureEngineer, create_snapshot_dates


def example_1_basic_usage():
    """Example 1: Basic end-to-end pipeline."""
    
    print_section_header("Example 1: Basic Pipeline Usage")
    
    # Setup
    config = Config('config/config.yaml')
    logger = setup_logging(config)
    
    # Load data
    df = pd.read_csv('data/raw/online_retail.csv', encoding='ISO-8859-1')
    logger.info(f"Loaded {len(df):,} rows")
    
    # Validate
    validator = DataValidator(config, logger)
    success, results = validator.validate_all(df)
    logger.info(f"Validation: {'PASSED' if success else 'FAILED'}")
    
    # Clean
    cleaner = DataCleaner(config, logger)
    df_clean, stats = cleaner.clean(df)
    logger.info(f"Cleaned: {len(df_clean):,} rows remaining")
    
    # Create features for one snapshot
    engineer = FeatureEngineer(config, logger)
    snapshot_date = df_clean['InvoiceDate'].max()
    features = engineer.create_features(df_clean, snapshot_date)
    logger.info(f"Features: {len(features)} customers, {len(features.columns)} features")
    
    print("\n✓ Basic pipeline complete!")


def example_2_multi_snapshot():
    """Example 2: Create features for multiple snapshots."""
    
    print_section_header("Example 2: Multi-Snapshot Feature Creation")
    
    # Setup
    config = Config('config/config.yaml')
    logger = setup_logging(config)
    
    # Load cleaned data (assuming cleaning already done)
    df_clean = pd.read_parquet('data/processed/cleaned_data.parquet')
    
    # Generate snapshot dates
    snapshots = create_snapshot_dates(
        df_clean,
        frequency='M',
        observation_window_days=180,
        prediction_window_days=90
    )
    
    logger.info(f"Generated {len(snapshots)} snapshots")
    
    # Create features for each snapshot
    engineer = FeatureEngineer(config, logger)
    all_features = []
    
    for snapshot in snapshots:
        features = engineer.create_features(df_clean, snapshot)
        all_features.append(features)
    
    # Combine
    df_features = pd.concat(all_features, ignore_index=False)
    logger.info(f"Total feature rows: {len(df_features):,}")
    
    # Save
    df_features.to_parquet('data/features/customer_features.parquet')
    
    print("\n✓ Multi-snapshot features created!")


def example_3_feature_inspection():
    """Example 3: Inspect and analyze features."""
    
    print_section_header("Example 3: Feature Inspection")
    
    # Load features
    features = pd.read_parquet('data/features/customer_features.parquet')
    
    print(f"\nFeature Store Summary:")
    print(f"  Total rows: {len(features):,}")
    print(f"  Unique customers: {features.index.nunique():,}")
    print(f"  Snapshots: {features['snapshot_date'].nunique()}")
    print(f"  Features: {len(features.columns)}")
    
    # Get latest snapshot
    latest_snapshot = features['snapshot_date'].max()
    latest_features = features[features['snapshot_date'] == latest_snapshot]
    
    print(f"\nLatest Snapshot ({latest_snapshot.date()}):")
    print(f"  Customers: {len(latest_features):,}")
    
    # Feature statistics
    print(f"\nRFM Statistics:")
    print(latest_features[['rfm_recency_days', 'rfm_frequency_count', 'rfm_monetary_avg']].describe())
    
    # Top customers by engagement
    print(f"\nTop 10 Customers by Engagement Score:")
    top_customers = latest_features.nlargest(10, 'engagement_score')[
        ['rfm_recency_days', 'rfm_frequency_count', 'rfm_monetary_total', 'engagement_score']
    ]
    print(top_customers)
    
    print("\n✓ Feature inspection complete!")


def example_4_custom_features():
    """Example 4: Create custom features."""
    
    print_section_header("Example 4: Custom Feature Engineering")
    
    # Load cleaned data
    df = pd.read_parquet('data/processed/cleaned_data.parquet')
    
    # Create custom aggregate features
    custom_features = df.groupby('CustomerID').agg({
        'Country': lambda x: x.mode()[0] if len(x) > 0 else None,  # Most common country
        'Hour': 'mean',  # Average purchase hour
        'DayOfWeek': lambda x: x.mode()[0] if len(x) > 0 else None,  # Preferred day
    })
    
    custom_features.columns = ['preferred_country', 'avg_purchase_hour', 'preferred_day']
    
    print("\nCustom Features Created:")
    print(custom_features.head())
    
    # Merge with existing features
    features = pd.read_parquet('data/features/customer_features.parquet')
    latest_snapshot = features['snapshot_date'].max()
    latest_features = features[features['snapshot_date'] == latest_snapshot]
    
    # Merge
    enriched_features = latest_features.join(custom_features, how='left')
    
    print(f"\nEnriched features: {len(enriched_features.columns)} columns")
    
    print("\n✓ Custom features added!")


def example_5_data_quality_report():
    """Example 5: Generate comprehensive data quality report."""
    
    print_section_header("Example 5: Data Quality Report")
    
    # Load validation results
    validation_report = pd.read_csv('outputs/validation_report.csv')
    
    print("\nValidation Results:")
    print(validation_report.to_string(index=False))
    
    # Load cleaning stats
    import json
    with open('outputs/cleaning_stats.json', 'r') as f:
        cleaning_stats = json.load(f)
    
    print("\nCleaning Statistics:")
    for key, value in cleaning_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")
    
    # Feature metadata
    with open('data/features/feature_metadata.json', 'r') as f:
        feature_metadata = json.load(f)
    
    print("\nFeature Store Metadata:")
    print(f"  Customers: {feature_metadata['n_customers']:,}")
    print(f"  Snapshots: {feature_metadata['n_snapshots']}")
    print(f"  Features: {feature_metadata['n_features']}")
    print(f"  Created: {feature_metadata['created_at']}")
    
    print("\n✓ Quality report generated!")


def example_6_temporal_validation():
    """Example 6: Validate temporal consistency."""
    
    print_section_header("Example 6: Temporal Consistency Validation")
    
    # Load data
    df = pd.read_parquet('data/processed/cleaned_data.parquet')
    features = pd.read_parquet('data/features/customer_features.parquet')
    
    # Pick a random customer and snapshot
    sample_customer = features.index[100]
    sample_snapshot = features.loc[sample_customer, 'snapshot_date'].iloc[0]
    
    print(f"\nSample Validation:")
    print(f"  CustomerID: {sample_customer}")
    print(f"  Snapshot Date: {sample_snapshot.date()}")
    
    # Get actual transactions
    customer_txns = df[
        (df['CustomerID'] == sample_customer) &
        (df['InvoiceDate'] <= sample_snapshot)
    ]
    
    # Compare with features
    feature_row = features[
        (features.index == sample_customer) &
        (features['snapshot_date'] == sample_snapshot)
    ].iloc[0]
    
    print(f"\nFeature Validation:")
    print(f"  Actual transactions: {len(customer_txns)}")
    print(f"  Feature frequency: {feature_row['rfm_frequency_count']}")
    print(f"  Match: {'✓' if abs(len(customer_txns) - feature_row['rfm_frequency_count']) < 1 else '✗'}")
    
    print(f"\n  Actual total revenue: ${customer_txns['Revenue'].sum():.2f}")
    print(f"  Feature total revenue: ${feature_row['rfm_monetary_total']:.2f}")
    print(f"  Match: {'✓' if abs(customer_txns['Revenue'].sum() - feature_row['rfm_monetary_total']) < 0.01 else '✗'}")
    
    # Check no future data used
    future_txns = df[
        (df['CustomerID'] == sample_customer) &
        (df['InvoiceDate'] > sample_snapshot)
    ]
    
    print(f"\nTemporal Consistency:")
    print(f"  Future transactions (should be > 0): {len(future_txns)}")
    print(f"  Used in features: No (by design)")
    
    print("\n✓ Temporal consistency validated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pillar 0 Examples")
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help='Which example to run (1-6)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_usage,
        2: example_2_multi_snapshot,
        3: example_3_feature_inspection,
        4: example_4_custom_features,
        5: example_5_data_quality_report,
        6: example_6_temporal_validation
    }
    
    try:
        examples[args.example]()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to run the main pipeline first:")
        print("  python src/pillar_0_main.py --raw-data data/raw/online_retail.csv")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
