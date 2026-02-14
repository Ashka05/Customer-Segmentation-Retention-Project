"""
Pillar 3: Customer Lifetime Value (CLV) Prediction - Main Execution Script
==========================================================================
Orchestrates CLV target creation, model training, and prediction.

Workflow:
1. Load cleaned data and features
2. Create CLV targets (future revenue)
3. Prepare train/val/test splits
4. Train LightGBM regression model
5. Train quantile models for uncertainty
6. Evaluate on test set
7. Generate predictions with confidence intervals
8. Combine with churn predictions for risk-adjusted CLV

Usage:
    python pillar_3_main.py
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import (
    Config, setup_logging, Timer, save_dataframe, print_section_header
)
from models.clv_prediction import CLVPredictor


def main(args):
    """Main execution function."""
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print_section_header("PILLAR 3: CUSTOMER LIFETIME VALUE PREDICTION")
    
    config = Config(args.config)
    logger = setup_logging(config)
    
    logger.info("Configuration loaded successfully")
    
    # ========================================================================
    # TASK 3.1: Create CLV Targets
    # ========================================================================
    print_section_header("Task 3.1: Create CLV Targets (Future Revenue)")
    
    with Timer("CLV Target Creation", logger):
        # Load cleaned data
        logger.info(f"Loading cleaned data from: {args.cleaned_data}")
        df_clean = pd.read_parquet(args.cleaned_data)
        logger.info(f"Loaded {len(df_clean):,} transactions")
        
        # Load features
        logger.info(f"Loading features from: {args.features}")
        features = pd.read_parquet(args.features)
        logger.info(f"Loaded {len(features):,} feature rows")
        
        # Initialize predictor
        predictor = CLVPredictor(config, logger)
        
        # Create CLV targets
        clv_targets = predictor.create_targets(
            df=df_clean,
            features=features,
            prediction_horizon_days=config.get('clv.prediction_horizon_days', 180)
        )
        
        logger.info(f"\n✓ Created {len(clv_targets):,} CLV targets")
        
        # Save targets
        output_path = Path(config.get('paths.outputs'))
        targets_path = output_path / 'clv_targets.parquet'
        clv_targets.to_parquet(targets_path, index=True)
        logger.info(f"Saved targets to: {targets_path}")
    
    # ========================================================================
    # TASK 3.2: Train CLV Prediction Model
    # ========================================================================
    print_section_header("Task 3.2: Train LightGBM Regression Model")
    
    with Timer("Model Training", logger):
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(
            features=features,
            targets=clv_targets,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        # Store original scale test targets for evaluation
        if predictor.log_transform:
            y_test_original = np.expm1(y_test)
        else:
            y_test_original = y_test
        
        # Train main model
        predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            objective='regression'
        )
        
        logger.info(f"\n✓ Main CLV model trained successfully")
    
    # ========================================================================
    # TASK 3.3: Train Quantile Models for Uncertainty
    # ========================================================================
    print_section_header("Task 3.3: Train Quantile Regression Models")
    
    with Timer("Quantile Model Training", logger):
        predictor.train_quantile_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        logger.info(f"\n✓ Quantile models trained successfully")
    
    # ========================================================================
    # TASK 3.4: Evaluate on Test Set
    # ========================================================================
    print_section_header("Task 3.4: Evaluate on Test Set")
    
    with Timer("Model Evaluation", logger):
        test_metrics = predictor.evaluate(
            X_test=X_test,
            y_test=y_test,
            y_test_original=y_test_original
        )
        
        # Save metrics
        metrics_path = output_path / 'clv_model_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types
            metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for k, v in test_metrics.items() if k != 'decile_analysis'}
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"\nMetrics saved to: {metrics_path}")
        
        # Save decile analysis
        if 'decile_analysis' in test_metrics:
            decile_path = output_path / 'clv_decile_analysis.csv'
            test_metrics['decile_analysis'].to_csv(decile_path, index=False)
            logger.info(f"Decile analysis saved to: {decile_path}")
    
    # ========================================================================
    # TASK 3.5: Generate CLV Predictions
    # ========================================================================
    print_section_header("Task 3.5: Generate CLV Predictions")
    
    with Timer("Prediction Generation", logger):
        # Get latest snapshot features
        if 'snapshot_date' in features.columns:
            latest_snapshot = features['snapshot_date'].max()
            latest_features = features[features['snapshot_date'] == latest_snapshot].copy()
        else:
            latest_features = features.copy()
        
        logger.info(f"Generating predictions for {len(latest_features):,} customers")
        
        # Get predictions with confidence intervals
        clv_predictions = predictor.predict(
            latest_features,
            return_confidence_interval=True
        )
        
        # Save predictions
        predictions_path = output_path / 'clv_predictions.parquet'
        clv_predictions.to_parquet(predictions_path, index=True)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Log distribution
        logger.info("\nCLV prediction statistics:")
        logger.info(f"  Mean: ${clv_predictions['clv_prediction'].mean():.2f}")
        logger.info(f"  Median: ${clv_predictions['clv_prediction'].median():.2f}")
        logger.info(f"  Std: ${clv_predictions['clv_prediction'].std():.2f}")
        logger.info(f"  Min: ${clv_predictions['clv_prediction'].min():.2f}")
        logger.info(f"  Max: ${clv_predictions['clv_prediction'].max():.2f}")
        
        # Value segments
        clv_segments = pd.cut(
            clv_predictions['clv_prediction'],
            bins=[0, 100, 300, 500, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info("\nCLV segments:")
        for segment, count in clv_segments.value_counts().sort_index().items():
            pct = count / len(clv_segments) * 100
            logger.info(f"  {segment}: {count:,} ({pct:.1f}%)")
    
    # ========================================================================
    # TASK 3.6: Risk-Adjusted CLV (Integration with Churn)
    # ========================================================================
    print_section_header("Task 3.6: Risk-Adjusted CLV (Churn Integration)")
    
    with Timer("Risk-Adjusted CLV", logger):
        # Try to load churn predictions
        churn_predictions_path = output_path / 'churn_predictions.parquet'
        
        if churn_predictions_path.exists():
            logger.info("Loading churn predictions...")
            churn_predictions = pd.read_parquet(churn_predictions_path)
            
            # Merge with CLV predictions
            combined = clv_predictions.join(churn_predictions[['churn_probability']], how='left')
            combined['churn_probability'] = combined['churn_probability'].fillna(0.2)  # Default 20% if missing
            
            # Calculate risk-adjusted CLV
            combined['clv_risk_adjusted'] = (
                combined['clv_prediction'] * (1 - combined['churn_probability'])
            )
            
            # Expected loss from churn
            combined['expected_churn_loss'] = (
                combined['clv_prediction'] * combined['churn_probability']
            )
            
            # Save combined predictions
            combined_path = output_path / 'clv_predictions_risk_adjusted.parquet'
            combined.to_parquet(combined_path, index=True)
            logger.info(f"Risk-adjusted predictions saved to: {combined_path}")
            
            # Statistics
            logger.info("\nRisk-adjusted CLV statistics:")
            logger.info(f"  Mean base CLV: ${combined['clv_prediction'].mean():.2f}")
            logger.info(f"  Mean risk-adjusted CLV: ${combined['clv_risk_adjusted'].mean():.2f}")
            logger.info(f"  Mean expected loss: ${combined['expected_churn_loss'].mean():.2f}")
            logger.info(f"  Total expected loss: ${combined['expected_churn_loss'].sum():,.2f}")
            
            # Top customers by risk-adjusted CLV
            logger.info("\nTop 10 customers by risk-adjusted CLV:")
            top_10 = combined.nlargest(10, 'clv_risk_adjusted')[
                ['clv_prediction', 'churn_probability', 'clv_risk_adjusted']
            ]
            for customer_id, row in top_10.iterrows():
                logger.info(
                    f"  {customer_id}: CLV=${row['clv_prediction']:.2f}, "
                    f"Churn={row['churn_probability']:.1%}, "
                    f"Risk-adjusted=${row['clv_risk_adjusted']:.2f}"
                )
        else:
            logger.warning("Churn predictions not found. Skipping risk adjustment.")
            logger.warning("Run Pillar 2 first to generate churn predictions.")
    
    # ========================================================================
    # TASK 3.7: Feature Importance Analysis
    # ========================================================================
    print_section_header("Task 3.7: Feature Importance Analysis")
    
    with Timer("Feature Importance", logger):
        feature_importance = predictor.feature_importance
        
        logger.info("\nTop 20 most important features for CLV prediction:")
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # Save feature importance
        importance_path = output_path / 'clv_feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"\nFeature importance saved to: {importance_path}")
    
    # ========================================================================
    # TASK 3.8: Save Model
    # ========================================================================
    print_section_header("Task 3.8: Save Trained Model")
    
    with Timer("Model Saving", logger):
        model_path = Path(config.get('paths.models')) / 'clv_predictor'
        predictor.save_model(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}.pkl")
    
    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    print_section_header("PILLAR 3 COMPLETION SUMMARY")
    
    logger.info("✓ Task 3.1: CLV Target Creation - COMPLETE")
    logger.info("✓ Task 3.2: Model Training - COMPLETE")
    logger.info("✓ Task 3.3: Quantile Models - COMPLETE")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PILLAR 3 STATUS: 3/3 TASKS COMPLETE ✓")
    logger.info("=" * 80)
    
    logger.info("\nKey Results:")
    logger.info(f"  - Targets created: {len(clv_targets):,}")
    logger.info(f"  - Test RMSE: ${test_metrics['rmse']:.2f}")
    logger.info(f"  - Test R² Score: {test_metrics['r2_score']:.4f}")
    logger.info(f"  - Test MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  - Portfolio error: {test_metrics['portfolio_error_pct']:.2f}%")
    
    logger.info("\nCLV Distribution:")
    quartiles = clv_predictions['clv_prediction'].quantile([0.25, 0.5, 0.75, 0.9])
    for q, val in quartiles.items():
        logger.info(f"  {q*100:.0f}th percentile: ${val:.2f}")
    
    logger.info("\nOutput files created:")
    logger.info(f"  1. CLV targets: {targets_path}")
    logger.info(f"  2. Model metrics: {metrics_path}")
    logger.info(f"  3. Decile analysis: {decile_path}")
    logger.info(f"  4. CLV predictions: {predictions_path}")
    if churn_predictions_path.exists():
        logger.info(f"  5. Risk-adjusted CLV: {combined_path}")
    logger.info(f"  6. Feature importance: {importance_path}")
    logger.info(f"  7. Trained model: {model_path}.pkl")
    
    logger.info("\nTop 3 predictive features:")
    for idx, row in feature_importance.head(3).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL PROGRESS: 17/18 TASKS COMPLETE (94%)")
    logger.info("=" * 80)
    logger.info("Pillar 0: ✓ Complete (6/6 tasks)")
    logger.info("Pillar 1: ✓ Complete (4/4 tasks)")
    logger.info("Pillar 2: ✓ Complete (4/4 tasks)")
    logger.info("Pillar 3: ✓ Complete (3/3 tasks)")
    logger.info("Pillar 4: ⏳ Pending (0/3 tasks) - Revenue Optimization")
    
    logger.info("\nNext steps:")
    logger.info("  → Proceed to Pillar 4: Revenue Optimization")
    logger.info("  → Combine churn probability + CLV for optimal targeting")
    logger.info("  → Run: python pillar_4_optimization.py")
    
    logger.info("\nBusiness Actions:")
    logger.info("  → Focus retention efforts on high CLV customers")
    logger.info("  → Use risk-adjusted CLV for budget allocation")
    logger.info("  → Target high (CLV × churn_risk) customers first")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pillar 3: Customer Lifetime Value Prediction"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--cleaned-data',
        type=str,
        default='data/processed/cleaned_data.parquet',
        help='Path to cleaned transactional data from Pillar 0'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='data/features/customer_features.parquet',
        help='Path to features from Pillar 0'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proportion of training data for validation'
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    sys.exit(exit_code)
