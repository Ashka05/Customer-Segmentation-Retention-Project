"""
Feature Engineering Module
===========================
Production-grade feature engineering with point-in-time correctness.

Key principles:
1. Temporal consistency: Only use data available at prediction time
2. Snapshot-based: Features computed for specific snapshot dates
3. Scalable: Vectorized operations for millions of records
4. Modular: Separate feature groups for flexibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Feature engineering pipeline with temporal consistency.
    
    Feature groups:
    1. RFM (Recency, Frequency, Monetary)
    2. Temporal behavior (trends, patterns)
    3. Product interaction
    4. Engagement metrics
    5. Seasonality
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.feature_metadata = {}
        
    def create_features(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        observation_window_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Create features for a specific snapshot date.
        
        CRITICAL: Only uses data up to snapshot_date (point-in-time correctness)
        
        Args:
            df: Cleaned transactional data
            snapshot_date: Date to compute features for
            observation_window_days: Days of history to use (None = all available)
            
        Returns:
            DataFrame with features per customer
        """
        self.logger.info("=" * 80)
        self.logger.info(f"FEATURE ENGINEERING FOR SNAPSHOT: {snapshot_date.date()}")
        self.logger.info("=" * 80)
        
        if observation_window_days is None:
            observation_window_days = self.config.get('churn.observation_window_days', 180)
        
        # Filter data to snapshot date (TEMPORAL CONSISTENCY)
        observation_start = snapshot_date - pd.Timedelta(days=observation_window_days)
        df_snapshot = df[
            (df['InvoiceDate'] <= snapshot_date) &
            (df['InvoiceDate'] >= observation_start)
        ].copy()
        
        self.logger.info(f"Observation window: {observation_start.date()} to {snapshot_date.date()}")
        self.logger.info(f"Transactions in window: {len(df_snapshot):,}")
        self.logger.info(f"Unique customers: {df_snapshot['CustomerID'].nunique():,}")
        
        # Initialize feature dictionary
        features_dict = {}
        
        # 1. RFM Features
        self.logger.info("\n--- Computing RFM Features ---")
        rfm_features = self._compute_rfm_features(df_snapshot, snapshot_date)
        features_dict.update(rfm_features)
        
        # 2. Temporal Behavior Features
        self.logger.info("--- Computing Temporal Behavior Features ---")
        temporal_features = self._compute_temporal_features(df_snapshot, snapshot_date)
        features_dict.update(temporal_features)
        
        # 3. Product Interaction Features
        self.logger.info("--- Computing Product Interaction Features ---")
        product_features = self._compute_product_features(df_snapshot)
        features_dict.update(product_features)
        
        # 4. Engagement Features
        self.logger.info("--- Computing Engagement Features ---")
        engagement_features = self._compute_engagement_features(
            rfm_features,
            temporal_features,
            snapshot_date
        )
        features_dict.update(engagement_features)
        
        # 5. Trend Features
        self.logger.info("--- Computing Trend Features ---")
        trend_features = self._compute_trend_features(df_snapshot, snapshot_date)
        features_dict.update(trend_features)
        
        # Combine all features
        df_features = pd.concat(features_dict.values(), axis=1)
        
        # Add metadata
        df_features['snapshot_date'] = snapshot_date
        df_features['observation_start_date'] = observation_start
        df_features['observation_window_days'] = observation_window_days
        
        self.logger.info(f"\n✓ Created {len(df_features.columns)} features for {len(df_features):,} customers")
        
        return df_features
    
    def _compute_rfm_features(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute RFM (Recency, Frequency, Monetary) features.
        
        Args:
            df: Transactional data
            snapshot_date: Reference date for recency calculation
            
        Returns:
            Dictionary of feature DataFrames
        """
        # Group by customer
        customer_data = df.groupby('CustomerID').agg({
            'InvoiceDate': ['min', 'max', 'count'],
            'Revenue': ['sum', 'mean', 'std'],
            'InvoiceNo': 'nunique',
            'Quantity': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        customer_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in customer_data.columns]
        customer_data.columns = ['CustomerID', 
                                'first_purchase_date', 'last_purchase_date', 'transaction_count',
                                'total_revenue', 'avg_revenue_per_transaction', 'std_revenue',
                                'unique_invoices', 'total_quantity', 'avg_quantity']
        
        # Recency (days since last purchase)
        customer_data['rfm_recency_days'] = (
            snapshot_date - customer_data['last_purchase_date']
        ).dt.days
        
        # Frequency (number of transactions)
        customer_data['rfm_frequency_count'] = customer_data['transaction_count']
        
        # Monetary (average revenue)
        customer_data['rfm_monetary_avg'] = customer_data['avg_revenue_per_transaction']
        customer_data['rfm_monetary_total'] = customer_data['total_revenue']
        customer_data['rfm_monetary_std'] = customer_data['std_revenue'].fillna(0)
        
        # Additional RFM metrics
        customer_data['days_since_first_purchase'] = (
            snapshot_date - customer_data['first_purchase_date']
        ).dt.days
        
        customer_data['customer_lifetime_days'] = (
            customer_data['last_purchase_date'] - customer_data['first_purchase_date']
        ).dt.days
        
        # Set index
        rfm_df = customer_data.set_index('CustomerID')[[
            'rfm_recency_days',
            'rfm_frequency_count',
            'rfm_monetary_avg',
            'rfm_monetary_total',
            'rfm_monetary_std',
            'days_since_first_purchase',
            'customer_lifetime_days'
        ]]
        
        self.logger.info(f"  ✓ RFM features: {len(rfm_df.columns)} features")
        
        return {'rfm': rfm_df}
    
    def _compute_temporal_features(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute temporal behavior features.
        
        Features:
        - Purchase frequency per time period
        - Interpurchase time statistics
        - Activity in different time windows
        """
        time_windows = self.config.get('feature_engineering.time_windows', {
            'short': 30,
            'medium': 60,
            'long': 90
        })
        
        # Get all customers from the dataframe
        all_customers = df['CustomerID'].unique()
        
        # Initialize DataFrame with all customers
        temporal_df = pd.DataFrame(index=pd.Index(all_customers, name='CustomerID'))
        
        # For each time window, compute metrics
        for window_name, window_days in time_windows.items():
            window_start = snapshot_date - pd.Timedelta(days=window_days)
            df_window = df[df['InvoiceDate'] >= window_start]
            
            # Transactions in window
            txn_in_window = df_window.groupby('CustomerID').size()
            temporal_df[f'txn_count_{window_name}_{window_days}d'] = txn_in_window
            
            # Revenue in window
            revenue_in_window = df_window.groupby('CustomerID')['Revenue'].sum()
            temporal_df[f'revenue_{window_name}_{window_days}d'] = revenue_in_window
        
        # Interpurchase time statistics
        # Group and calculate, then convert to dict for mapping
        interpurchase_times = df.groupby('CustomerID')['InvoiceDate'].apply(
            lambda x: x.sort_values().diff().dt.days.dropna() if len(x) > 1 else pd.Series([], dtype='float64')
        )
        
        # Calculate statistics
        ipt_mean = interpurchase_times.apply(
            lambda x: x.mean() if isinstance(x, pd.Series) and len(x) > 0 else np.nan
        )
        ipt_std = interpurchase_times.apply(
            lambda x: x.std() if isinstance(x, pd.Series) and len(x) > 1 else np.nan
        )
        ipt_median = interpurchase_times.apply(
            lambda x: x.median() if isinstance(x, pd.Series) and len(x) > 0 else np.nan
        )
        
        # Convert to dict and map to temporal_df index
        temporal_df['interpurchase_time_mean'] = temporal_df.index.map(ipt_mean.to_dict()).fillna(np.nan)
        temporal_df['interpurchase_time_std'] = temporal_df.index.map(ipt_std.to_dict()).fillna(np.nan)
        temporal_df['interpurchase_time_median'] = temporal_df.index.map(ipt_median.to_dict()).fillna(np.nan)
        
        temporal_df['interpurchase_time_cv'] = (
            temporal_df['interpurchase_time_std'] / 
            temporal_df['interpurchase_time_mean']
        ).fillna(0)
        
        # Purchase frequency (transactions per month)
        customer_lifetime = df.groupby('CustomerID').apply(
            lambda x: (x['InvoiceDate'].max() - x['InvoiceDate'].min()).days / 30
        ).replace(0, 1)  # Avoid division by zero
        
        transaction_count = df.groupby('CustomerID').size()
        
        # Calculate and map using dictionary
        purchase_freq = (transaction_count / customer_lifetime).fillna(0)
        temporal_df['purchase_frequency_per_month'] = temporal_df.index.map(purchase_freq.to_dict()).fillna(0)
        
        # Fill any remaining NaN values
        temporal_df = temporal_df.fillna(0)
        
        self.logger.info(f"  ✓ Temporal features: {len(temporal_df.columns)} features")
        
        return {'temporal': temporal_df}
    
    def _compute_product_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute product interaction features.
        
        Features:
        - Unique products purchased
        - Average basket size
        - Product diversity
        - Return rate
        """
        product_features = {}
        
        # Unique products
        product_features['unique_products_count'] = df.groupby('CustomerID')['StockCode'].nunique()
        
        # Average basket size (items per transaction)
        basket_size = df.groupby(['CustomerID', 'InvoiceNo'])['Quantity'].sum().abs()
        product_features['avg_basket_size'] = basket_size.groupby('CustomerID').mean()
        product_features['std_basket_size'] = basket_size.groupby('CustomerID').std().fillna(0)
        
        # Return rate (percentage of transactions that are returns)
        if 'is_return' in df.columns:
            total_txn = df.groupby('CustomerID').size()
            return_txn = df[df['is_return']].groupby('CustomerID').size()
            product_features['return_rate'] = (return_txn / total_txn).fillna(0)
        
        # Product diversity (inverse of concentration)
        # Higher value = more diverse purchases
        product_counts = df.groupby(['CustomerID', 'StockCode']).size()
        total_purchases = product_counts.groupby('CustomerID').sum()
        product_shares = product_counts / total_purchases
        product_features['product_diversity_hhi'] = 1 - (
            (product_shares ** 2).groupby('CustomerID').sum()
        )
        
        # Create DataFrame
        product_df = pd.DataFrame(product_features)
        product_df = product_df.fillna(0)
        
        self.logger.info(f"  ✓ Product features: {len(product_df.columns)} features")
        
        return {'product': product_df}
    
    def _compute_engagement_features(
        self,
        rfm_features: Dict[str, pd.DataFrame],
        temporal_features: Dict[str, pd.DataFrame],
        snapshot_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute engagement scores and derived metrics.
        
        Engagement = f(Frequency, Monetary, Recency)
        """
        rfm_df = rfm_features['rfm']
        temporal_df = temporal_features['temporal']
        
        # Start with RFM index to ensure consistency
        engagement_df = pd.DataFrame(index=rfm_df.index)
        
        # Classic engagement score: (Frequency × Monetary) / Recency
        # Add 1 to recency to avoid division by zero
        engagement_df['engagement_score'] = (
            (rfm_df['rfm_frequency_count'] * rfm_df['rfm_monetary_avg']) /
            (rfm_df['rfm_recency_days'] + 1)
        )
        
        # Normalized engagement score (0-1 scale)
        engagement_df['engagement_score_normalized'] = (
            engagement_df['engagement_score'] / 
            engagement_df['engagement_score'].max()
        ).fillna(0)
        
        # Consistency score: inverse of interpurchase time CV
        # Higher = more consistent purchasing
        if 'interpurchase_time_cv' in temporal_df.columns:
            engagement_df['consistency_score'] = 1 / (
                temporal_df['interpurchase_time_cv'] + 1
            )
        
        # Activity score: recent activity vs total activity
        if 'txn_count_short_30d' in temporal_df.columns:
            engagement_df['activity_score'] = (
                temporal_df['txn_count_short_30d'] / 
                (rfm_df['rfm_frequency_count'] + 1)
            ).fillna(0)
        
        # Fill any remaining NaN values
        engagement_df = engagement_df.fillna(0)
        
        self.logger.info(f"  ✓ Engagement features: {len(engagement_df.columns)} features")
        
        return {'engagement': engagement_df}
    
    def _compute_trend_features(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute trend features (are metrics increasing/decreasing?).
        
        Uses linear regression on monthly aggregates.
        """
        trend_features = {}
        
        # Monthly revenue trend
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['InvoiceDate'].dt.to_period('M')
        
        monthly_revenue = df_monthly.groupby(['CustomerID', 'month'])['Revenue'].sum().reset_index()
        
        # Compute trend slope for each customer
        from scipy import stats
        
        def compute_trend(group):
            if len(group) < 2:
                return 0
            x = np.arange(len(group))
            y = group['Revenue'].values
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        
        trend_features['revenue_trend_slope'] = monthly_revenue.groupby('CustomerID').apply(
            compute_trend
        )
        
        # Monthly transaction count trend
        monthly_txn = df_monthly.groupby(['CustomerID', 'month']).size().reset_index(name='count')
        
        def compute_txn_trend(group):
            if len(group) < 2:
                return 0
            x = np.arange(len(group))
            y = group['count'].values
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        
        trend_features['transaction_trend_slope'] = monthly_txn.groupby('CustomerID').apply(
            compute_txn_trend
        )
        
        # Trend direction (positive/negative/flat)
        trend_df = pd.DataFrame(trend_features)
        trend_df['revenue_trend_direction'] = np.sign(trend_df['revenue_trend_slope'])
        trend_df['transaction_trend_direction'] = np.sign(trend_df['transaction_trend_slope'])
        
        trend_df = trend_df.fillna(0)
        
        self.logger.info(f"  ✓ Trend features: {len(trend_df.columns)} features")
        
        return {'trend': trend_df}
    
    def create_multiple_snapshots(
        self,
        df: pd.DataFrame,
        snapshot_dates: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Create features for multiple snapshot dates.
        
        Args:
            df: Cleaned transactional data
            snapshot_dates: List of snapshot dates
            
        Returns:
            Combined DataFrame with all snapshots
        """
        all_features = []
        
        for snapshot_date in snapshot_dates:
            self.logger.info(f"\nProcessing snapshot: {snapshot_date.date()}")
            features = self.create_features(df, snapshot_date)
            all_features.append(features)
        
        df_combined = pd.concat(all_features, ignore_index=False)
        
        self.logger.info(f"\n✓ Created features for {len(snapshot_dates)} snapshots")
        self.logger.info(f"Total feature rows: {len(df_combined):,}")
        
        return df_combined


def create_snapshot_dates(
    df: pd.DataFrame,
    frequency: str = 'ME',
    observation_window_days: int = 180,
    prediction_window_days: int = 90
) -> List[pd.Timestamp]:
    """
    Create snapshot dates for temporal validation.
    
    Args:
        df: DataFrame with InvoiceDate column
        frequency: Snapshot frequency ('ME' = month end, 'W' = weekly)
        observation_window_days: Days of history needed
        prediction_window_days: Days ahead to predict
        
    Returns:
        List of valid snapshot dates
    """
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    
    # First snapshot needs observation window of history
    first_snapshot = min_date + pd.Timedelta(days=observation_window_days)
    
    # Last snapshot needs prediction window in future
    last_snapshot = max_date - pd.Timedelta(days=prediction_window_days)
    
    if first_snapshot >= last_snapshot:
        raise ValueError(
            f"Insufficient data range. Need at least {observation_window_days + prediction_window_days} days. "
            f"Available: {(max_date - min_date).days} days"
        )
    
    # Generate snapshots
    snapshots = pd.date_range(
        start=first_snapshot,
        end=last_snapshot,
        freq=frequency.replace('M', 'ME')  # Handle both 'M' and 'ME'
    )
    
    return snapshots.tolist()


if __name__ == "__main__":
    print("Feature Engineering Module - Test Mode")
    print("=" * 80)
    print("This module provides point-in-time correct feature engineering")
    print("Key principles:")
    print("1. Temporal consistency (no data leakage)")
    print("2. Snapshot-based features")
    print("3. Scalable vectorized operations")
