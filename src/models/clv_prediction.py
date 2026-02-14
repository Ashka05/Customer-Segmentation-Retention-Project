"""
Customer Lifetime Value (CLV) Prediction Model
==============================================
LightGBM-based regression for predicting customer lifetime value.

Approaches:
1. Point prediction: Mean/median revenue prediction
2. Quantile regression: Uncertainty estimation (10th, 50th, 90th percentiles)
3. Risk-adjusted CLV: Integration with churn predictions

Key Features:
- Handles right-skewed revenue distribution
- Log transformation for better predictions
- Quantile regression for confidence intervals
- Integration with churn for risk-adjusted CLV
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class CLVPredictor:
    """
    Customer Lifetime Value prediction using LightGBM regression.
    
    Workflow:
    1. Create CLV targets (future revenue)
    2. Train point estimate model (mean prediction)
    3. Train quantile models (uncertainty)
    4. Risk-adjusted CLV (combine with churn)
    5. Evaluate and interpret
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize CLV predictor.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = None
        self.model_lower = None  # 10th percentile
        self.model_upper = None  # 90th percentile
        self.feature_names = None
        self.feature_importance = None
        self.log_transform = config.get('clv.log_transform', True)
        
    def create_targets(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        prediction_horizon_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create CLV targets (future revenue) for each customer at each snapshot.
        
        Args:
            df: Cleaned transactional data
            features: Feature DataFrame with snapshot_date
            prediction_horizon_days: Days of future revenue to predict
            
        Returns:
            DataFrame with CLV targets
        """
        self.logger.info("=" * 80)
        self.logger.info("CREATING CLV TARGETS")
        self.logger.info("=" * 80)
        
        if prediction_horizon_days is None:
            prediction_horizon_days = self.config.get('clv.prediction_horizon_days', 180)
        
        self.logger.info(f"Prediction horizon: {prediction_horizon_days} days")
        
        # Get unique snapshots
        if 'snapshot_date' not in features.columns:
            raise ValueError("Features must have snapshot_date column")
        
        snapshots = sorted(features['snapshot_date'].unique())
        self.logger.info(f"Processing {len(snapshots)} snapshots")
        
        all_targets = []
        
        for snapshot_date in snapshots:
            # Define prediction window
            prediction_start = snapshot_date
            prediction_end = prediction_start + pd.Timedelta(days=prediction_horizon_days)
            
            # Get revenue in prediction window
            df_prediction = df[
                (df['InvoiceDate'] > prediction_start) &
                (df['InvoiceDate'] <= prediction_end)
            ].copy()
            
            # Calculate revenue per customer
            customer_revenue = df_prediction.groupby('CustomerID')['Revenue'].sum()
            
            # Get customers from this snapshot
            snapshot_customers = features[features['snapshot_date'] == snapshot_date].index
            
            # Create targets (0 for customers with no future revenue)
            targets = pd.DataFrame({
                'CustomerID': snapshot_customers,
                'snapshot_date': snapshot_date,
                'clv_target': [customer_revenue.get(cid, 0.0) for cid in snapshot_customers]
            })
            
            all_targets.append(targets)
        
        # Combine all targets
        targets_df = pd.concat(all_targets, ignore_index=True)
        targets_df.set_index('CustomerID', inplace=True)
        
        # Statistics
        self.logger.info(f"\nCLV Target Statistics:")
        self.logger.info(f"  Total samples: {len(targets_df):,}")
        self.logger.info(f"  Mean CLV: ${targets_df['clv_target'].mean():.2f}")
        self.logger.info(f"  Median CLV: ${targets_df['clv_target'].median():.2f}")
        self.logger.info(f"  Std CLV: ${targets_df['clv_target'].std():.2f}")
        self.logger.info(f"  Max CLV: ${targets_df['clv_target'].max():.2f}")
        self.logger.info(f"  Customers with $0 CLV: {(targets_df['clv_target'] == 0).sum():,} ({(targets_df['clv_target'] == 0).sum()/len(targets_df)*100:.1f}%)")
        
        # Distribution
        quartiles = targets_df['clv_target'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        self.logger.info(f"\nCLV Distribution (quartiles):")
        for q, val in quartiles.items():
            self.logger.info(f"  {q*100:.0f}th percentile: ${val:.2f}")
        
        return targets_df
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training with temporal split.
        
        Args:
            features: Feature DataFrame
            targets: CLV targets
            test_size: Proportion for test set
            val_size: Proportion of train for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PREPARING DATA FOR CLV PREDICTION")
        self.logger.info("=" * 80)
        
        # Merge features with targets
        data = features.join(targets[['clv_target', 'snapshot_date']], how='inner', rsuffix='_target')
        
        # Use snapshot_date from targets if both exist
        if 'snapshot_date_target' in data.columns:
            data['snapshot_date'] = data['snapshot_date_target']
            data = data.drop('snapshot_date_target', axis=1)
        
        self.logger.info(f"Total samples: {len(data):,}")
        
        # Sort by snapshot_date for temporal split
        data = data.sort_values('snapshot_date')
        
        # Remove metadata columns
        feature_cols = [col for col in data.columns if col not in [
            'clv_target', 'snapshot_date', 'observation_start_date', 
            'observation_window_days', 'segment', 'segment_name',
            'churn_label', 'last_purchase_date', 'days_since_last_purchase',
            'transaction_count', 'tenure_days'
        ]]
        
        X = data[feature_cols].copy()
        y = data['clv_target'].copy()
        
        self.feature_names = feature_cols
        self.logger.info(f"Selected {len(feature_cols)} features for training")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Log transformation if enabled
        if self.log_transform:
            y_transformed = np.log1p(y)  # log(1 + y) to handle zeros
            self.logger.info("Applied log transformation to target: log(1 + revenue)")
        else:
            y_transformed = y
        
        # Temporal split (no shuffling!)
        train_val_size = int(len(X) * (1 - test_size))
        X_train_val = X.iloc[:train_val_size]
        X_test = X.iloc[train_val_size:]
        y_train_val = y_transformed.iloc[:train_val_size]
        y_test = y_transformed.iloc[train_val_size:]
        
        # Split train into train + validation
        val_absolute_size = int(len(X_train_val) * val_size)
        X_train = X_train_val.iloc[:-val_absolute_size]
        X_val = X_train_val.iloc[-val_absolute_size:]
        y_train = y_train_val.iloc[:-val_absolute_size]
        y_val = y_train_val.iloc[-val_absolute_size:]
        
        self.logger.info(f"\nTrain set: {len(X_train):,} samples")
        self.logger.info(f"Validation set: {len(X_val):,} samples")
        self.logger.info(f"Test set: {len(X_test):,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        objective: str = 'regression'
    ):
        """
        Train LightGBM CLV prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets (log-transformed if enabled)
            X_val: Validation features
            y_val: Validation targets
            objective: 'regression' for mean, 'quantile' for quantile regression
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING CLV PREDICTION MODEL")
        self.logger.info("=" * 80)
        
        # LightGBM parameters
        params = {
            'objective': objective,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': self.config.get('random_seed', 42)
        }
        
        if objective == 'quantile':
            params['alpha'] = 0.5  # Median
        
        self.logger.info("\nModel hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.logger.info("\nTraining LightGBM model...")
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"\n✓ Model trained successfully")
        self.logger.info(f"Best iteration: {self.model.best_iteration}")
        self.logger.info(f"Best score (RMSE): {self.model.best_score['valid']['rmse']:.4f}")
        
        # Log top features
        self.logger.info("\nTop 10 most important features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def train_quantile_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """
        Train quantile regression models for uncertainty estimation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING QUANTILE REGRESSION MODELS")
        self.logger.info("=" * 80)
        
        # Train 10th percentile model
        self.logger.info("\nTraining 10th percentile model (lower bound)...")
        params_lower = {
            'objective': 'quantile',
            'alpha': 0.1,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'random_state': self.config.get('random_seed', 42)
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model_lower = lgb.train(
            params_lower,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        # Train 90th percentile model
        self.logger.info("\nTraining 90th percentile model (upper bound)...")
        params_upper = params_lower.copy()
        params_upper['alpha'] = 0.9
        
        self.model_upper = lgb.train(
            params_upper,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        self.logger.info("\n✓ Quantile models trained successfully")
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_original: Optional[pd.Series] = None
    ) -> Dict:
        """
        Evaluate CLV prediction model.
        
        Args:
            X_test: Test features
            y_test: Test targets (log-transformed)
            y_test_original: Original scale targets (before log transform)
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATING CLV PREDICTION MODEL")
        self.logger.info("=" * 80)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform if log was applied
        if self.log_transform:
            y_pred_original = np.expm1(y_pred)  # exp(y) - 1
            if y_test_original is not None:
                y_test_eval = y_test_original
            else:
                y_test_eval = np.expm1(y_test)
        else:
            y_pred_original = y_pred
            y_test_eval = y_test_original if y_test_original is not None else y_test
        
        # Ensure we have valid arrays (convert Series to arrays if needed)
        if hasattr(y_test_eval, 'values'):
            y_test_eval = y_test_eval.values
        if hasattr(y_pred_original, 'values'):
            y_pred_original = y_pred_original.values
        
        # Clip negative predictions and handle any NaN
        y_pred_original = np.maximum(y_pred_original, 0)
        
        # Remove any NaN values from evaluation (shouldn't happen, but safe)
        valid_mask = ~(np.isnan(y_test_eval) | np.isnan(y_pred_original))
        if not valid_mask.all():
            self.logger.warning(f"Removing {(~valid_mask).sum()} samples with NaN values")
            y_test_eval = y_test_eval[valid_mask]
            y_pred_original = y_pred_original[valid_mask]
        
        # Compute metrics on original scale
        mse = mean_squared_error(y_test_eval, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_eval, y_pred_original)
        r2 = r2_score(y_test_eval, y_pred_original)
        
        # MAPE (avoid division by zero)
        mask = y_test_eval > 0
        mape = np.mean(np.abs((y_test_eval[mask] - y_pred_original[mask]) / y_test_eval[mask])) * 100
        
        # Portfolio-level error
        total_actual = y_test_eval.sum()
        total_predicted = y_pred_original.sum()
        portfolio_error = abs(total_actual - total_predicted) / total_actual * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'portfolio_error_pct': float(portfolio_error),
            'total_actual_clv': float(total_actual),
            'total_predicted_clv': float(total_predicted)
        }
        
        # Log results
        self.logger.info(f"Regression metrics (original scale):")
        self.logger.info(f"  RMSE: ${rmse:.2f}")
        self.logger.info(f"  MAE: ${mae:.2f}")
        self.logger.info(f"  R² Score: {r2:.4f}")
        self.logger.info(f"  MAPE: {mape:.2f}%")
        
        self.logger.info(f"\nPortfolio-level metrics:")
        self.logger.info(f"  Total actual CLV: ${total_actual:,.2f}")
        self.logger.info(f"  Total predicted CLV: ${total_predicted:,.2f}")
        self.logger.info(f"  Portfolio error: {portfolio_error:.2f}%")
        
        # Decile analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DECILE ANALYSIS")
        self.logger.info("=" * 80)
        
        decile_analysis = self._decile_analysis(y_test_eval, y_pred_original)
        metrics['decile_analysis'] = decile_analysis
        
        return metrics
    
    def _decile_analysis(self, y_true, y_pred) -> pd.DataFrame:
        """
        Perform decile analysis (sort by prediction, compute actual CLV per decile).
        
        Args:
            y_true: Actual CLV values
            y_pred: Predicted CLV values
            
        Returns:
            DataFrame with decile analysis
        """
        # Create DataFrame
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        
        # Sort by predicted and create deciles
        df = df.sort_values('predicted', ascending=False).reset_index(drop=True)
        df['decile'] = pd.qcut(df.index, q=10, labels=False, duplicates='drop') + 1
        
        # Aggregate by decile
        decile_stats = df.groupby('decile').agg({
            'actual': ['sum', 'mean', 'count'],
            'predicted': ['sum', 'mean']
        }).reset_index()
        
        decile_stats.columns = ['decile', 'actual_sum', 'actual_mean', 'count', 'predicted_sum', 'predicted_mean']
        
        # Add cumulative percentage
        total_actual = decile_stats['actual_sum'].sum()
        decile_stats['actual_pct'] = decile_stats['actual_sum'] / total_actual * 100
        decile_stats['cumulative_pct'] = decile_stats['actual_pct'].cumsum()
        
        self.logger.info("\nDecile analysis (top decile = highest predicted CLV):")
        for _, row in decile_stats.iterrows():
            self.logger.info(
                f"  Decile {int(row['decile'])}: "
                f"Actual=${row['actual_sum']:,.0f} ({row['actual_pct']:.1f}%), "
                f"Predicted=${row['predicted_sum']:,.0f}, "
                f"Count={int(row['count'])}"
            )
        
        return decile_stats
    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence_interval: bool = False
    ) -> pd.DataFrame:
        """
        Predict CLV for customers.
        
        Args:
            X: Features
            return_confidence_interval: If True and quantile models trained, return bounds
            
        Returns:
            DataFrame with predictions and optionally confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match training
        X_aligned = X[self.feature_names].copy()
        X_aligned = X_aligned.fillna(X_aligned.median())
        
        # Predict
        y_pred = self.model.predict(X_aligned)
        
        # Inverse transform
        if self.log_transform:
            y_pred = np.expm1(y_pred)
        
        # Clip negative
        y_pred = np.maximum(y_pred, 0)
        
        result = pd.DataFrame({
            'CustomerID': X.index,
            'clv_prediction': y_pred
        }).set_index('CustomerID')
        
        # Add confidence intervals if quantile models available
        if return_confidence_interval and self.model_lower is not None and self.model_upper is not None:
            y_lower = self.model_lower.predict(X_aligned)
            y_upper = self.model_upper.predict(X_aligned)
            
            if self.log_transform:
                y_lower = np.expm1(y_lower)
                y_upper = np.expm1(y_upper)
            
            result['clv_lower_10pct'] = np.maximum(y_lower, 0)
            result['clv_upper_90pct'] = np.maximum(y_upper, 0)
            result['clv_uncertainty'] = result['clv_upper_90pct'] - result['clv_lower_10pct']
        
        return result
    
    def predict_with_churn_adjustment(
        self,
        X: pd.DataFrame,
        churn_probabilities: pd.Series
    ) -> pd.DataFrame:
        """
        Predict risk-adjusted CLV (accounting for churn probability).
        
        Args:
            X: Features
            churn_probabilities: Churn probabilities for each customer
            
        Returns:
            DataFrame with adjusted CLV predictions
        """
        # Get base CLV predictions
        clv_predictions = self.predict(X, return_confidence_interval=True)
        
        # Adjust for churn risk
        # Expected CLV = CLV × (1 - churn_probability)
        clv_predictions['churn_probability'] = churn_probabilities
        clv_predictions['clv_risk_adjusted'] = (
            clv_predictions['clv_prediction'] * (1 - clv_predictions['churn_probability'])
        )
        
        return clv_predictions
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'model_lower': self.model_lower,
            'model_upper': self.model_upper,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'log_transform': self.log_transform
        }
        
        output_path = Path(path).with_suffix('.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to: {output_path}")
    
    @classmethod
    def load_model(cls, path: str, config, logger=None):
        """Load trained model from disk."""
        model_path = Path(path).with_suffix('.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(config, logger)
        instance.model = model_data['model']
        instance.model_lower = model_data.get('model_lower')
        instance.model_upper = model_data.get('model_upper')
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']
        instance.log_transform = model_data.get('log_transform', True)
        
        if logger:
            logger.info(f"Model loaded from: {model_path}")
        
        return instance


if __name__ == "__main__":
    print("CLV Prediction Model")
    print("=" * 80)
    print("LightGBM-based regression for customer lifetime value prediction")
    print("Includes quantile regression for uncertainty estimation")
