"""
Churn Label Creation Module
============================
Creates churn labels from transactional data based on inactivity windows.

Key Concept: Churn = Customer inactive for X days (configurable, typically 90 days)

Critical: Must maintain temporal consistency (no data leakage)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from datetime import timedelta


class ChurnLabeler:
    """
    Create churn labels from transactional data.
    
    Methodology:
    - Churn = No purchase within CHURN_WINDOW days after observation end
    - Must have minimum tenure and transactions to be eligible
    - Validates labels by checking if churned customers stay churned
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize churn labeler.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Get parameters from config
        self.churn_window = config.get('churn.churn_window_days', 90)
        self.observation_window = config.get('churn.observation_window_days', 180)
        self.prediction_window = config.get('churn.prediction_window_days', 90)
        self.gap_days = config.get('churn.gap_days', 0)
        self.min_tenure = config.get('churn.min_tenure_days', 30)
        self.min_transactions = config.get('churn.min_transactions', 2)
        
    def create_labels(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        return_stats: bool = True
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Create churn labels for a specific snapshot date.
        
        Args:
            df: Cleaned transactional data (from Pillar 0)
            snapshot_date: Date to create labels for
            return_stats: Whether to return label statistics
            
        Returns:
            Tuple of (labels DataFrame, statistics dict)
        """
        self.logger.info("=" * 80)
        self.logger.info(f"CREATING CHURN LABELS FOR SNAPSHOT: {snapshot_date.date()}")
        self.logger.info("=" * 80)
        
        # Define time windows
        observation_start = snapshot_date - pd.Timedelta(days=self.observation_window)
        observation_end = snapshot_date
        
        # Optional gap (to avoid labeling very recent customers as churned)
        prediction_start = observation_end + pd.Timedelta(days=self.gap_days)
        prediction_end = prediction_start + pd.Timedelta(days=self.prediction_window)
        
        self.logger.info(f"Observation window: {observation_start.date()} to {observation_end.date()}")
        self.logger.info(f"Prediction window: {prediction_start.date()} to {prediction_end.date()}")
        if self.gap_days > 0:
            self.logger.info(f"Gap period: {self.gap_days} days")
        
        # Filter data for observation window
        df_observation = df[
            (df['InvoiceDate'] >= observation_start) &
            (df['InvoiceDate'] <= observation_end)
        ].copy()
        
        # Filter data for prediction window
        df_prediction = df[
            (df['InvoiceDate'] > prediction_start) &
            (df['InvoiceDate'] <= prediction_end)
        ].copy()
        
        self.logger.info(f"Transactions in observation window: {len(df_observation):,}")
        self.logger.info(f"Transactions in prediction window: {len(df_prediction):,}")
        
        # Get customer-level statistics for observation window
        customer_stats = self._compute_customer_stats(df_observation, observation_start, observation_end)
        
        self.logger.info(f"Customers in observation window: {len(customer_stats):,}")
        
        # Filter customers by eligibility criteria
        eligible_customers = self._filter_eligible_customers(customer_stats)
        
        self.logger.info(f"Eligible customers (after filters): {len(eligible_customers):,}")
        
        # Determine which customers made purchases in prediction window
        active_customers = set(df_prediction['CustomerID'].unique())
        
        # Create labels
        labels = []
        for customer_id in eligible_customers.index:
            is_churned = 1 if customer_id not in active_customers else 0
            
            labels.append({
                'CustomerID': customer_id,
                'snapshot_date': snapshot_date,
                'churn_label': is_churned,
                'last_purchase_date': eligible_customers.loc[customer_id, 'last_purchase_date'],
                'days_since_last_purchase': eligible_customers.loc[customer_id, 'days_since_last_purchase'],
                'transaction_count': eligible_customers.loc[customer_id, 'transaction_count'],
                'tenure_days': eligible_customers.loc[customer_id, 'tenure_days']
            })
        
        labels_df = pd.DataFrame(labels)
        labels_df.set_index('CustomerID', inplace=True)
        
        # Compute statistics
        stats = None
        if return_stats:
            stats = self._compute_label_stats(labels_df, eligible_customers, active_customers)
        
        self.logger.info("\n✓ Churn labels created successfully")
        
        return labels_df, stats
    
    def _compute_customer_stats(
        self,
        df: pd.DataFrame,
        observation_start: pd.Timestamp,
        observation_end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Compute customer-level statistics for observation window.
        
        Args:
            df: Transactional data in observation window
            observation_start: Start of observation window
            observation_end: End of observation window
            
        Returns:
            DataFrame with customer statistics
        """
        customer_stats = df.groupby('CustomerID').agg({
            'InvoiceDate': ['min', 'max', 'count'],
            'Revenue': 'sum'
        })
        
        # Flatten column names
        customer_stats.columns = ['first_purchase_date', 'last_purchase_date', 'transaction_count', 'total_revenue']
        
        # Compute derived metrics
        customer_stats['days_since_last_purchase'] = (
            observation_end - customer_stats['last_purchase_date']
        ).dt.days
        
        customer_stats['tenure_days'] = (
            customer_stats['last_purchase_date'] - customer_stats['first_purchase_date']
        ).dt.days
        
        # Days active in observation window
        customer_stats['days_in_observation'] = (
            observation_end - customer_stats['first_purchase_date'].clip(lower=observation_start)
        ).dt.days
        
        return customer_stats
    
    def _filter_eligible_customers(self, customer_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Filter customers based on eligibility criteria.
        
        Args:
            customer_stats: Customer statistics DataFrame
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(customer_stats)
        
        # Filter 1: Minimum transactions
        eligible = customer_stats[customer_stats['transaction_count'] >= self.min_transactions].copy()
        removed_min_txn = initial_count - len(eligible)
        
        # Filter 2: Minimum tenure (had first purchase long enough ago)
        eligible = eligible[eligible['days_in_observation'] >= self.min_tenure].copy()
        removed_min_tenure = len(customer_stats) - removed_min_txn - len(eligible)
        
        self.logger.info(f"\nEligibility filtering:")
        self.logger.info(f"  Initial customers: {initial_count:,}")
        self.logger.info(f"  Removed (< {self.min_transactions} transactions): {removed_min_txn:,}")
        self.logger.info(f"  Removed (< {self.min_tenure} days tenure): {removed_min_tenure:,}")
        self.logger.info(f"  Eligible customers: {len(eligible):,}")
        
        return eligible
    
    def _compute_label_stats(
        self,
        labels_df: pd.DataFrame,
        eligible_customers: pd.DataFrame,
        active_customers: set
    ) -> Dict:
        """
        Compute statistics about the created labels.
        
        Args:
            labels_df: Created labels
            eligible_customers: Customer statistics
            active_customers: Set of customers active in prediction window
            
        Returns:
            Dictionary of statistics
        """
        total_customers = len(labels_df)
        churned_count = (labels_df['churn_label'] == 1).sum()
        active_count = (labels_df['churn_label'] == 0).sum()
        
        churn_rate = (churned_count / total_customers * 100) if total_customers > 0 else 0
        
        # Statistics by churn status
        churned_stats = eligible_customers.loc[
            labels_df[labels_df['churn_label'] == 1].index
        ]
        active_stats = eligible_customers.loc[
            labels_df[labels_df['churn_label'] == 0].index
        ]
        
        stats = {
            'total_customers': int(total_customers),
            'churned_count': int(churned_count),
            'active_count': int(active_count),
            'churn_rate_pct': float(churn_rate),
            
            # Churned customer characteristics
            'churned_avg_recency': float(churned_stats['days_since_last_purchase'].mean()) if len(churned_stats) > 0 else None,
            'churned_avg_frequency': float(churned_stats['transaction_count'].mean()) if len(churned_stats) > 0 else None,
            'churned_avg_revenue': float(churned_stats['total_revenue'].mean()) if len(churned_stats) > 0 else None,
            
            # Active customer characteristics
            'active_avg_recency': float(active_stats['days_since_last_purchase'].mean()) if len(active_stats) > 0 else None,
            'active_avg_frequency': float(active_stats['transaction_count'].mean()) if len(active_stats) > 0 else None,
            'active_avg_revenue': float(active_stats['total_revenue'].mean()) if len(active_stats) > 0 else None,
            
            # Configuration used
            'churn_window_days': self.churn_window,
            'observation_window_days': self.observation_window,
            'prediction_window_days': self.prediction_window,
            'min_tenure_days': self.min_tenure,
            'min_transactions': self.min_transactions
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CHURN LABEL STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total customers: {stats['total_customers']:,}")
        self.logger.info(f"Churned: {stats['churned_count']:,} ({stats['churn_rate_pct']:.2f}%)")
        self.logger.info(f"Active: {stats['active_count']:,} ({100 - stats['churn_rate_pct']:.2f}%)")
        
        self.logger.info("\nChurned customer profile:")
        self.logger.info(f"  Avg recency: {stats['churned_avg_recency']:.1f} days")
        self.logger.info(f"  Avg frequency: {stats['churned_avg_frequency']:.1f} transactions")
        self.logger.info(f"  Avg revenue: ${stats['churned_avg_revenue']:.2f}")
        
        self.logger.info("\nActive customer profile:")
        self.logger.info(f"  Avg recency: {stats['active_avg_recency']:.1f} days")
        self.logger.info(f"  Avg frequency: {stats['active_avg_frequency']:.1f} transactions")
        self.logger.info(f"  Avg revenue: ${stats['active_avg_revenue']:.2f}")
        
        self.logger.info("=" * 80 + "\n")
        
        return stats
    
    def validate_labels(
        self,
        df: pd.DataFrame,
        labels_df: pd.DataFrame,
        validation_window_days: int = 90
    ) -> Dict:
        """
        Validate churn labels by checking if churned customers stay churned.
        
        Args:
            df: Full transactional data
            labels_df: Created churn labels
            validation_window_days: Days after prediction window to check
            
        Returns:
            Validation statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("VALIDATING CHURN LABELS")
        self.logger.info("=" * 80)
        
        snapshot_date = labels_df['snapshot_date'].iloc[0]
        prediction_start = snapshot_date + pd.Timedelta(days=self.gap_days)
        prediction_end = prediction_start + pd.Timedelta(days=self.prediction_window)
        validation_end = prediction_end + pd.Timedelta(days=validation_window_days)
        
        self.logger.info(f"Validation window: {prediction_end.date()} to {validation_end.date()}")
        
        # Get transactions in validation window
        df_validation = df[
            (df['InvoiceDate'] > prediction_end) &
            (df['InvoiceDate'] <= validation_end)
        ].copy()
        
        returned_customers = set(df_validation['CustomerID'].unique())
        
        # Check churned customers
        churned_customers = labels_df[labels_df['churn_label'] == 1].index
        
        # How many churned customers returned?
        false_churns = set(churned_customers) & returned_customers
        true_churns = set(churned_customers) - returned_customers
        
        # Precision of churn label
        precision = len(true_churns) / len(churned_customers) if len(churned_customers) > 0 else 0
        
        validation_stats = {
            'churned_customers': len(churned_customers),
            'stayed_churned': len(true_churns),
            'returned': len(false_churns),
            'churn_label_precision': float(precision * 100),
            'validation_window_days': validation_window_days
        }
        
        self.logger.info(f"\nChurned customers: {validation_stats['churned_customers']:,}")
        self.logger.info(f"Stayed churned: {validation_stats['stayed_churned']:,}")
        self.logger.info(f"Returned: {validation_stats['returned']:,}")
        self.logger.info(f"Churn label precision: {validation_stats['churn_label_precision']:.2f}%")
        
        if precision < 0.8:
            self.logger.warning(f"⚠️  Low churn label precision ({precision*100:.1f}%). Consider:")
            self.logger.warning(f"   - Increasing churn_window_days (currently {self.churn_window})")
            self.logger.warning(f"   - Adjusting prediction_window_days")
        else:
            self.logger.info(f"✓ Good churn label precision ({precision*100:.1f}%)")
        
        self.logger.info("=" * 80 + "\n")
        
        return validation_stats
    
    def create_labels_for_multiple_snapshots(
        self,
        df: pd.DataFrame,
        snapshot_dates: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create labels for multiple snapshot dates.
        
        Args:
            df: Cleaned transactional data
            snapshot_dates: List of snapshot dates
            
        Returns:
            Tuple of (combined labels DataFrame, statistics DataFrame)
        """
        all_labels = []
        all_stats = []
        
        for i, snapshot_date in enumerate(snapshot_dates, 1):
            self.logger.info(f"\nProcessing snapshot {i}/{len(snapshot_dates)}: {snapshot_date.date()}")
            
            labels_df, stats = self.create_labels(df, snapshot_date, return_stats=True)
            
            all_labels.append(labels_df)
            all_stats.append(stats)
        
        # Combine all labels
        combined_labels = pd.concat(all_labels, ignore_index=False)
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(all_stats)
        stats_df['snapshot_date'] = snapshot_dates
        
        self.logger.info(f"\n✓ Created labels for {len(snapshot_dates)} snapshots")
        self.logger.info(f"Total label rows: {len(combined_labels):,}")
        
        return combined_labels, stats_df


if __name__ == "__main__":
    print("Churn Label Creation Module")
    print("=" * 80)
    print("Creates ground truth churn labels from transactional data")
    print("Based on inactivity windows (typically 90 days)")
