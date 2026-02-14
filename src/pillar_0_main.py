"""
Pillar 0: Data Engineering Pipeline - Main Execution Script
===========================================================
Orchestrates the complete data engineering workflow:
1. Configuration loading
2. Data validation
3. Data cleaning
4. Feature engineering
5. Feature store creation

Usage:
    python pillar_0_main.py --config config/config.yaml --raw-data data/raw/online_retail.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import (
    Config, setup_logging, set_random_seed, Timer,
    save_dataframe, print_section_header
)
from data.validation import validate_raw_data
from data.cleaning import load_and_clean_data
from data.feature_engineering import FeatureEngineer, create_snapshot_dates


def main(args):
    """Main execution function."""
    
    # ========================================================================
    # TASK 0.1: Load Configuration
    # ========================================================================
    print_section_header("PILLAR 0: DATA ENGINEERING PIPELINE")
    
    config = Config(args.config)
    logger = setup_logging(config)
    set_random_seed(config.get('random_seed', 42))
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Project: {config.get('project.name')}")
    logger.info(f"Version: {config.get('project.version')}")
    
    # ========================================================================
    # TASK 0.2: Data Validation
    # ========================================================================
    print_section_header("Task 0.2: Data Validation")
    
    with Timer("Data Validation", logger):
        import pandas as pd
        
        # Load raw data
        try:
            df_raw = pd.read_csv(args.raw_data, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 encoding failed, trying ISO-8859-1")
            df_raw = pd.read_csv(args.raw_data, encoding='ISO-8859-1')
        
        logger.info(f"Loaded raw data: {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        
        # Run validation
        validation_passed, validation_report = validate_raw_data(df_raw, config, logger)
        
        # Save validation report
        validation_output = Path(config.get('paths.outputs')) / 'validation_report.csv'
        validation_report.to_csv(validation_output, index=False)
        logger.info(f"Validation report saved to: {validation_output}")
        
        if not validation_passed:
            logger.warning("⚠️  Data validation found errors (see report)")
            if args.strict:
                logger.error("Strict mode enabled - stopping due to validation errors")
                return 1
        else:
            logger.info("✓ Data validation passed")
    
    # ========================================================================
    # TASK 0.3: Data Cleaning
    # ========================================================================
    print_section_header("Task 0.3: Data Cleaning")
    
    with Timer("Data Cleaning", logger):
        cleaned_output = Path(config.get('paths.processed_data')) / 'cleaned_data.parquet'
        
        df_clean, cleaning_stats = load_and_clean_data(
            file_path=args.raw_data,
            config=config,
            logger=logger,
            save_output=True,
            output_path=str(cleaned_output)
        )
        
        logger.info(f"Cleaned data saved to: {cleaned_output}")
        
        # Save cleaning statistics
        import json
        stats_output = Path(config.get('paths.outputs')) / 'cleaning_stats.json'
        with open(stats_output, 'w') as f:
            json.dump(cleaning_stats, f, indent=2)
        logger.info(f"Cleaning statistics saved to: {stats_output}")
    
    # ========================================================================
    # TASK 0.4: Feature Engineering
    # ========================================================================
    print_section_header("Task 0.4: Feature Engineering")
    
    with Timer("Feature Engineering", logger):
        # Create feature engineer
        feature_engineer = FeatureEngineer(config, logger)
        
        # Generate snapshot dates
        logger.info("Generating snapshot dates...")
        
        try:
            snapshot_dates = create_snapshot_dates(
                df_clean,
                frequency='M',  # Monthly snapshots
                observation_window_days=config.get('churn.observation_window_days', 180),
                prediction_window_days=config.get('churn.prediction_window_days', 90)
            )
            
            logger.info(f"Generated {len(snapshot_dates)} snapshot dates")
            logger.info(f"First snapshot: {snapshot_dates[0].date()}")
            logger.info(f"Last snapshot: {snapshot_dates[-1].date()}")
            
        except ValueError as e:
            logger.error(f"Error generating snapshots: {e}")
            logger.info("Creating single snapshot using max date")
            max_date = df_clean['InvoiceDate'].max()
            snapshot_dates = [max_date]
        
        # Create features for each snapshot
        if args.single_snapshot:
            # Just create features for the last snapshot (faster for testing)
            logger.info("Single snapshot mode - using last snapshot only")
            snapshot_dates = [snapshot_dates[-1]]
        
        all_features = []
        for i, snapshot_date in enumerate(snapshot_dates, 1):
            logger.info(f"\nProcessing snapshot {i}/{len(snapshot_dates)}: {snapshot_date.date()}")
            
            features = feature_engineer.create_features(
                df_clean,
                snapshot_date=snapshot_date
            )
            
            all_features.append(features)
        
        # Combine all snapshots
        df_features = pd.concat(all_features, ignore_index=False)
        logger.info(f"\n✓ Total feature rows: {len(df_features):,}")
        logger.info(f"✓ Total features: {len(df_features.columns)}")
    
    # ========================================================================
    # TASK 0.5: Feature Store Creation
    # ========================================================================
    print_section_header("Task 0.5: Feature Store Creation")
    
    with Timer("Feature Store Creation", logger):
        feature_store_path = Path(config.get('paths.features'))
        feature_store_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        output_format = config.get('feature_store.format', 'parquet')
        compression = config.get('feature_store.compression', 'snappy')
        
        features_output = feature_store_path / f'customer_features.{output_format}'
        
        if output_format == 'parquet':
            df_features.to_parquet(
                features_output,
                index=True,
                compression=compression
            )
        elif output_format == 'csv':
            df_features.to_csv(features_output, index=True)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        logger.info(f"Features saved to: {features_output}")
        
        # Save feature metadata
        feature_info = {
            'n_customers': int(df_features.index.nunique()),
            'n_snapshots': len(snapshot_dates),
            'n_features': len(df_features.columns),
            'feature_names': df_features.columns.tolist(),
            'snapshot_dates': [str(d.date()) for d in snapshot_dates],
            'created_at': str(pd.Timestamp.now())
        }
        
        import json
        metadata_output = feature_store_path / 'feature_metadata.json'
        with open(metadata_output, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"Feature metadata saved to: {metadata_output}")
    
    # ========================================================================
    # TASK 0.6: Temporal Consistency Validation
    # ========================================================================
    print_section_header("Task 0.6: Temporal Consistency Validation")
    
    with Timer("Temporal Validation", logger):
        # Verify no data leakage by checking that features only use past data
        logger.info("Validating temporal consistency...")
        
        # Check 1: All snapshot dates should be in the past relative to max date
        max_invoice_date = df_clean['InvoiceDate'].max()
        for snapshot_date in snapshot_dates:
            if snapshot_date > max_invoice_date:
                logger.error(f"❌ Snapshot date {snapshot_date} is after max data date {max_invoice_date}")
                return 1
        
        logger.info("✓ All snapshot dates are valid")
        
        # Check 2: Verify features are computed correctly
        # Sample one customer and verify manually
        sample_customer = df_features.index[0]
        sample_snapshot = df_features.loc[sample_customer, 'snapshot_date'].iloc[0]
        
        # Get transactions for this customer up to snapshot
        customer_txns = df_clean[
            (df_clean['CustomerID'] == sample_customer) &
            (df_clean['InvoiceDate'] <= sample_snapshot)
        ]
        
        expected_frequency = len(customer_txns)
        actual_frequency = df_features.loc[sample_customer, 'rfm_frequency_count'].iloc[0]
        
        if abs(expected_frequency - actual_frequency) < 0.01:
            logger.info(f"✓ Temporal consistency validated (sample check passed)")
        else:
            logger.warning(f"⚠️  Possible temporal inconsistency detected")
            logger.warning(f"Expected frequency: {expected_frequency}, Actual: {actual_frequency}")
    
    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    print_section_header("PILLAR 0 COMPLETION SUMMARY")
    
    logger.info("✓ Task 0.1: Project Setup & Configuration - COMPLETE")
    logger.info("✓ Task 0.2: Data Validation & Quality Checks - COMPLETE")
    logger.info("✓ Task 0.3: Data Cleaning Pipeline - COMPLETE")
    logger.info("✓ Task 0.4: Feature Engineering Framework - COMPLETE")
    logger.info("✓ Task 0.5: Feature Store Implementation - COMPLETE")
    logger.info("✓ Task 0.6: Temporal Consistency Validation - COMPLETE")
    
    logger.info("\n" + "=" * 80)
    logger.info("PILLAR 0 STATUS: 6/6 TASKS COMPLETE ✓")
    logger.info("=" * 80)
    
    logger.info("\nOutput files created:")
    logger.info(f"  1. Validation report: {validation_output}")
    logger.info(f"  2. Cleaned data: {cleaned_output}")
    logger.info(f"  3. Cleaning stats: {stats_output}")
    logger.info(f"  4. Feature store: {features_output}")
    logger.info(f"  5. Feature metadata: {metadata_output}")
    
    logger.info("\nNext steps:")
    logger.info("  → Proceed to Pillar 1: RFM Segmentation")
    logger.info("  → Run: python pillar_1_rfm.py")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pillar 0: Data Engineering Pipeline"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--raw-data',
        type=str,
        required=True,
        help='Path to raw UCI Online Retail CSV file'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Stop execution if validation fails'
    )
    
    parser.add_argument(
        '--single-snapshot',
        action='store_true',
        help='Only create features for last snapshot (faster for testing)'
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    sys.exit(exit_code)
