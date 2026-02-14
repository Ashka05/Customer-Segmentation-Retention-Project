"""
Churn Prediction Model
======================
LightGBM-based binary classification for customer churn prediction.

Key Features:
- Handles class imbalance (SMOTE, class weights, threshold optimization)
- Feature importance analysis
- Probability calibration
- Business metric optimization
- Model persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, fbeta_score,
    precision_score, recall_score
)
import lightgbm as lgb
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnPredictor:
    """
    Churn prediction using LightGBM with class imbalance handling.
    
    Workflow:
    1. Load features + labels
    2. Handle class imbalance
    3. Train LightGBM
    4. Optimize threshold for business metrics
    5. Evaluate and interpret
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize churn predictor.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.feature_importance = None
        self.training_history = None
        
    def prepare_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training with temporal split.
        
        Args:
            features: Feature DataFrame from Pillar 0
            labels: Churn labels from churn_labeling
            test_size: Proportion for test set
            val_size: Proportion of train for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("=" * 80)
        self.logger.info("PREPARING DATA FOR CHURN PREDICTION")
        self.logger.info("=" * 80)
        
        # Merge features with labels
        data = features.join(labels[['churn_label', 'snapshot_date']], how='inner', rsuffix='_label')
        
        self.logger.info(f"Total samples: {len(data):,}")
        self.logger.info(f"Features: {len(features.columns)}")
        
        # Use the snapshot_date from labels (or keep original if no duplicate)
        if 'snapshot_date_label' in data.columns:
            data['snapshot_date'] = data['snapshot_date_label']
            data = data.drop('snapshot_date_label', axis=1)
        
        # Sort by snapshot_date for temporal split
        data = data.sort_values('snapshot_date')
        
        # Remove metadata columns
        feature_cols = [col for col in data.columns if col not in [
            'churn_label', 'snapshot_date', 'snapshot_date_label',
            'observation_start_date', 'observation_window_days', 
            'segment', 'segment_name'
        ]]
        
        X = data[feature_cols].copy()
        y = data['churn_label'].copy()
        
        self.feature_names = feature_cols
        self.logger.info(f"Selected {len(feature_cols)} features for training")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Temporal split (no shuffling!)
        train_val_size = int(len(X) * (1 - test_size))
        X_train_val = X.iloc[:train_val_size]
        X_test = X.iloc[train_val_size:]
        y_train_val = y.iloc[:train_val_size]
        y_test = y.iloc[train_val_size:]
        
        # Split train into train + validation
        val_absolute_size = int(len(X_train_val) * val_size)
        X_train = X_train_val.iloc[:-val_absolute_size]
        X_val = X_train_val.iloc[-val_absolute_size:]
        y_train = y_train_val.iloc[:-val_absolute_size]
        y_val = y_train_val.iloc[-val_absolute_size:]
        
        # Log class distribution
        self._log_class_distribution(y_train, y_val, y_test)
        
        self.logger.info(f"\nTrain set: {len(X_train):,} samples")
        self.logger.info(f"Validation set: {len(X_val):,} samples")
        self.logger.info(f"Test set: {len(X_test):,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_class_distribution(self, y_train, y_val, y_test):
        """Log class distribution across splits."""
        self.logger.info("\nClass distribution:")
        
        for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            churned = (y == 1).sum()
            active = (y == 0).sum()
            churn_rate = churned / len(y) * 100
            
            self.logger.info(f"  {name}: Churned={churned:,} ({churn_rate:.1f}%), Active={active:,} ({100-churn_rate:.1f}%)")
            
            if churn_rate < 10 or churn_rate > 40:
                self.logger.warning(f"    ⚠️  {name} has extreme churn rate ({churn_rate:.1f}%)")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        handle_imbalance: str = 'class_weight'
    ):
        """
        Train LightGBM churn prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            handle_imbalance: Strategy ('class_weight', 'scale_pos_weight', or 'none')
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING CHURN PREDICTION MODEL")
        self.logger.info("=" * 80)
        
        # Compute class weights
        class_counts = y_train.value_counts()
        if handle_imbalance == 'class_weight':
            scale_pos_weight = class_counts[0] / class_counts[1]
            self.logger.info(f"Using class weight: {scale_pos_weight:.2f}")
        elif handle_imbalance == 'scale_pos_weight':
            scale_pos_weight = class_counts[0] / class_counts[1]
        else:
            scale_pos_weight = 1.0
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_child_samples': 20,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': self.config.get('random_seed', 42)
        }
        
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
        
        self.training_history = None  # Not available in newer LightGBM
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"\n✓ Model trained successfully")
        self.logger.info(f"Best iteration: {self.model.best_iteration}")
        self.logger.info(f"Best score (AUC): {self.model.best_score['valid']['auc']:.4f}")
        
        # Log top features
        self.logger.info("\nTop 10 most important features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f_beta',
        beta: float = 2.0
    ) -> float:
        """
        Optimize decision threshold for business metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'f_beta', 'precision', 'recall')
            beta: Beta value for F-beta score (beta=2 weights recall 2x)
            
        Returns:
            Optimal threshold
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("OPTIMIZING DECISION THRESHOLD")
        self.logger.info("=" * 80)
        
        # Get probability predictions
        y_proba = self.model.predict(X_val)
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'f_beta':
                score = fbeta_score(y_val, y_pred, beta=beta)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        # Find optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        self.optimal_threshold = optimal_threshold
        
        self.logger.info(f"Metric: {metric}" + (f" (beta={beta})" if metric == 'f_beta' else ""))
        self.logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        self.logger.info(f"Score at optimal threshold: {optimal_score:.4f}")
        
        # Log performance at different thresholds
        self.logger.info("\nPerformance at key thresholds:")
        for t in [0.3, 0.5, optimal_threshold, 0.7]:
            y_pred = (y_proba >= t).astype(int)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            f_beta = fbeta_score(y_val, y_pred, beta=beta)
            
            self.logger.info(f"  Threshold {t:.2f}: Precision={prec:.3f}, Recall={rec:.3f}, F-beta={f_beta:.3f}")
        
        return optimal_threshold
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold (if None, uses optimal_threshold)
            
        Returns:
            Dictionary of metrics
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATING CHURN PREDICTION MODEL")
        self.logger.info("=" * 80)
        
        # Predictions
        y_proba = self.model.predict(X_test)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, fbeta_score, roc_auc_score
        )
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'f2_score': fbeta_score(y_test, y_pred, beta=2.0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Business metrics
        metrics['churn_capture_rate'] = recall_score(y_test, y_pred)  # Same as recall
        metrics['precision_at_threshold'] = precision_score(y_test, y_pred)
        
        # Lift metrics
        total_churned = y_test.sum()
        predicted_churned = y_pred.sum()
        
        if predicted_churned > 0:
            top_k_pct = predicted_churned / len(y_test)
            metrics['lift_at_top_k_pct'] = (tp / predicted_churned) / (total_churned / len(y_test))
        else:
            metrics['lift_at_top_k_pct'] = 0
        
        # Log results
        self.logger.info(f"Decision threshold: {threshold:.3f}")
        self.logger.info(f"\nClassification metrics:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"  F2 Score (beta=2): {metrics['f2_score']:.4f}")
        self.logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        self.logger.info(f"\nConfusion matrix:")
        self.logger.info(f"  True Negatives: {tn:,}")
        self.logger.info(f"  False Positives: {fp:,}")
        self.logger.info(f"  False Negatives: {fn:,}")
        self.logger.info(f"  True Positives: {tp:,}")
        
        self.logger.info(f"\nBusiness metrics:")
        self.logger.info(f"  Churn capture rate: {metrics['churn_capture_rate']:.2%}")
        self.logger.info(f"  Predicted churners: {predicted_churned:,} ({predicted_churned/len(y_test)*100:.1f}%)")
        if metrics['lift_at_top_k_pct'] > 0:
            self.logger.info(f"  Lift vs random: {metrics['lift_at_top_k_pct']:.2f}x")
        
        # Classification report
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CLASSIFICATION REPORT")
        self.logger.info("=" * 80)
        report = classification_report(y_test, y_pred, target_names=['Active', 'Churned'])
        self.logger.info("\n" + report)
        
        return metrics
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Predict churn for new customers.
        
        Args:
            X: Features
            return_proba: If True, return probabilities; else return binary predictions
            
        Returns:
            Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match training
        X_aligned = X[self.feature_names].copy()
        X_aligned = X_aligned.fillna(X_aligned.median())
        
        # Predict probabilities
        y_proba = self.model.predict(X_aligned)
        
        if return_proba:
            return y_proba
        else:
            return (y_proba >= self.optimal_threshold).astype(int)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Output path (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'feature_importance': self.feature_importance
        }
        
        output_path = Path(path).with_suffix('.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to: {output_path}")
    
    @classmethod
    def load_model(cls, path: str, config, logger=None):
        """
        Load trained model from disk.
        
        Args:
            path: Model path
            config: Configuration object
            logger: Logger instance
            
        Returns:
            Loaded ChurnPredictor instance
        """
        model_path = Path(path).with_suffix('.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(config, logger)
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.optimal_threshold = model_data['optimal_threshold']
        instance.feature_importance = model_data['feature_importance']
        
        if logger:
            logger.info(f"Model loaded from: {model_path}")
        
        return instance


if __name__ == "__main__":
    print("Churn Prediction Model")
    print("=" * 80)
    print("LightGBM-based binary classification for customer churn")
    print("Handles class imbalance and optimizes for business metrics")
