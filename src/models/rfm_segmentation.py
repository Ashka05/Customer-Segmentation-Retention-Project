"""
RFM Customer Segmentation Module
=================================
Unsupervised learning for customer segmentation using RFM (Recency, Frequency, Monetary) features.

Approach:
1. K-Means clustering on standardized RFM features
2. Optimal K selection using elbow method + silhouette score
3. Segment profiling and characterization
4. Business strategy mapping

Key Principle: Interpretable segments that drive business action
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
from pathlib import Path


class RFMSegmenter:
    """
    RFM-based customer segmentation using K-Means clustering.
    
    Features:
    - Automatic optimal K selection
    - Segment profiling with business metrics
    - Strategy recommendations per segment
    - Model persistence for production use
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize RFM segmenter.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.scaler = None
        self.kmeans = None
        self.n_clusters = None
        self.segment_profiles = None
        self.feature_importance = None
        
    def fit(
        self,
        features: pd.DataFrame,
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (3, 8)
    ) -> 'RFMSegmenter':
        """
        Fit K-Means clustering on RFM features.
        
        Args:
            features: DataFrame with RFM features (must include rfm_* columns)
            n_clusters: Number of clusters (if None, will auto-select)
            k_range: Range of K values to test for auto-selection
            
        Returns:
            Self for method chaining
        """
        self.logger.info("=" * 80)
        self.logger.info("FITTING RFM SEGMENTATION MODEL")
        self.logger.info("=" * 80)
        
        # Get latest snapshot only (for segmentation, we want current state)
        if 'snapshot_date' in features.columns:
            latest_snapshot = features['snapshot_date'].max()
            features = features[features['snapshot_date'] == latest_snapshot].copy()
            self.logger.info(f"Using latest snapshot: {latest_snapshot.date()}")
        
        # Select RFM features
        rfm_features = self._select_rfm_features(features)
        self.logger.info(f"Selected {len(rfm_features.columns)} RFM features")
        self.logger.info(f"Customers: {len(rfm_features):,}")
        
        # Standardize features
        self.logger.info("\n--- Standardizing Features ---")
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Auto-select K if not provided
        if n_clusters is None:
            self.logger.info(f"\n--- Auto-Selecting Optimal K (range: {k_range}) ---")
            n_clusters = self._select_optimal_k(rfm_scaled, k_range)
            self.logger.info(f"Selected K = {n_clusters}")
        
        self.n_clusters = n_clusters
        
        # Fit K-Means
        self.logger.info(f"\n--- Fitting K-Means (K={n_clusters}) ---")
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.config.get('random_seed', 42)
        )
        
        cluster_labels = self.kmeans.fit_predict(rfm_scaled)
        
        # Add cluster labels to features
        rfm_features['segment'] = cluster_labels
        
        # Compute metrics
        silhouette = silhouette_score(rfm_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(rfm_scaled, cluster_labels)
        inertia = self.kmeans.inertia_
        
        self.logger.info(f"Clustering metrics:")
        self.logger.info(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        self.logger.info(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        self.logger.info(f"  Inertia: {inertia:.2f}")
        
        # Profile segments
        self.logger.info("\n--- Profiling Segments ---")
        self.segment_profiles = self._profile_segments(rfm_features, features)
        
        # Compute feature importance
        self.feature_importance = self._compute_feature_importance(rfm_features)
        
        self.logger.info("\n✓ RFM segmentation model trained successfully")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict segment for new customers.
        
        Args:
            features: DataFrame with RFM features
            
        Returns:
            Array of segment labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Select RFM features
        rfm_features = self._select_rfm_features(features)
        
        # Standardize
        rfm_scaled = self.scaler.transform(rfm_features)
        
        # Predict
        segments = self.kmeans.predict(rfm_scaled)
        
        return segments
    
    def _select_rfm_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Select RFM and related features for clustering.
        
        Args:
            features: Full feature DataFrame
            
        Returns:
            DataFrame with selected features
        """
        # Core RFM features
        rfm_cols = [
            'rfm_recency_days',
            'rfm_frequency_count',
            'rfm_monetary_avg',
            'rfm_monetary_total'
        ]
        
        # Additional valuable features
        additional_cols = [
            'engagement_score',
            'purchase_frequency_per_month',
            'days_since_first_purchase'
        ]
        
        # Select available columns
        selected_cols = []
        for col in rfm_cols + additional_cols:
            if col in features.columns:
                selected_cols.append(col)
        
        if len(selected_cols) == 0:
            raise ValueError("No RFM features found in DataFrame")
        
        # Return selected features with customer ID as index
        rfm_features = features[selected_cols].copy()
        
        # Handle missing values (fill with median)
        for col in rfm_features.columns:
            if rfm_features[col].isna().any():
                median_val = rfm_features[col].median()
                rfm_features[col].fillna(median_val, inplace=True)
                self.logger.warning(f"Filled {rfm_features[col].isna().sum()} missing values in {col} with median")
        
        return rfm_features
    
    def _select_optimal_k(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int]
    ) -> int:
        """
        Select optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Standardized feature matrix
            k_range: Range of K values to test
            
        Returns:
            Optimal K
        """
        k_min, k_max = k_range
        inertias = []
        silhouettes = []
        k_values = range(k_min, k_max + 1)
        
        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.config.get('random_seed', 42)
            )
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            
            self.logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")
        
        # Find elbow using second derivative
        inertias_arr = np.array(inertias)
        second_derivative = np.diff(inertias_arr, n=2)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because of double diff
        
        # Find best silhouette
        best_silhouette_idx = np.argmax(silhouettes)
        
        self.logger.info(f"\nElbow method suggests K={k_min + elbow_idx}")
        self.logger.info(f"Silhouette method suggests K={k_min + best_silhouette_idx}")
        
        # Prefer silhouette score, but bounded by reasonable range
        optimal_k = k_min + best_silhouette_idx
        
        # Sanity check: if silhouette suggests too many clusters, use elbow
        if optimal_k > 6:
            optimal_k = min(k_min + elbow_idx, 6)
            self.logger.info(f"Limiting to K={optimal_k} for interpretability")
        
        return optimal_k
    
    def _profile_segments(
        self,
        rfm_features: pd.DataFrame,
        full_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive segment profiles.
        
        Args:
            rfm_features: RFM features with segment labels
            full_features: Full feature DataFrame
            
        Returns:
            DataFrame with segment profiles
        """
        profiles = []
        
        for segment in sorted(rfm_features['segment'].unique()):
            segment_data = rfm_features[rfm_features['segment'] == segment]
            segment_customers = segment_data.index
            
            # Get full features for this segment
            segment_full = full_features.loc[segment_customers]
            
            profile = {
                'segment': segment,
                'size': len(segment_data),
                'size_pct': (len(segment_data) / len(rfm_features)) * 100,
                
                # RFM metrics
                'avg_recency': segment_data['rfm_recency_days'].mean(),
                'avg_frequency': segment_data['rfm_frequency_count'].mean(),
                'avg_monetary': segment_data['rfm_monetary_avg'].mean(),
                'total_revenue': segment_data['rfm_monetary_total'].sum() if 'rfm_monetary_total' in segment_data.columns else 0,
                
                # Engagement
                'avg_engagement_score': segment_data['engagement_score'].mean() if 'engagement_score' in segment_data.columns else 0,
                
                # Customer lifetime
                'avg_customer_age_days': segment_data['days_since_first_purchase'].mean() if 'days_since_first_purchase' in segment_data.columns else 0,
            }
            
            # Add churn risk if available
            if 'churn_probability' in segment_full.columns:
                profile['avg_churn_risk'] = segment_full['churn_probability'].mean()
            
            # Add CLV if available
            if 'clv_prediction' in segment_full.columns:
                profile['avg_clv'] = segment_full['clv_prediction'].mean()
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        # Add segment names
        profiles_df['segment_name'] = profiles_df.apply(self._name_segment, axis=1)
        
        # Log segment summary
        self.logger.info("\nSegment Summary:")
        for _, row in profiles_df.iterrows():
            self.logger.info(f"\nSegment {row['segment']} - {row['segment_name']}:")
            self.logger.info(f"  Size: {row['size']:,} ({row['size_pct']:.1f}%)")
            self.logger.info(f"  Avg Recency: {row['avg_recency']:.1f} days")
            self.logger.info(f"  Avg Frequency: {row['avg_frequency']:.1f} transactions")
            self.logger.info(f"  Avg Monetary: ${row['avg_monetary']:.2f}")
            self.logger.info(f"  Total Revenue: ${row['total_revenue']:,.2f}")
        
        return profiles_df
    
    def _name_segment(self, row: pd.Series) -> str:
        """
        Assign business-friendly name to segment based on RFM characteristics.
        
        Args:
            row: Row from segment profile DataFrame
            
        Returns:
            Segment name
        """
        recency = row['avg_recency']
        frequency = row['avg_frequency']
        monetary = row['avg_monetary']
        
        # Champions: Recent, frequent, high value
        if recency < 30 and frequency > 10 and monetary > 100:
            return "Champions"
        
        # Loyal: Not super recent but frequent and high value
        elif frequency > 8 and monetary > 80:
            return "Loyal Customers"
        
        # At Risk: Were valuable but haven't purchased recently
        elif recency > 60 and frequency > 5 and monetary > 60:
            return "At Risk"
        
        # Hibernating: Long time since purchase, was somewhat active
        elif recency > 90 and frequency > 3:
            return "Hibernating"
        
        # New Customers: Recent but low frequency
        elif recency < 30 and frequency < 3:
            return "New Customers"
        
        # Promising: Recent, moderate activity
        elif recency < 45 and frequency >= 3:
            return "Promising"
        
        # Need Attention: Moderate on all dimensions
        elif recency < 60 and frequency >= 2:
            return "Need Attention"
        
        # Lost: Long time, low activity
        else:
            return "Lost/Churned"
    
    def _compute_feature_importance(self, rfm_features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature importance based on cluster center distances.
        
        Args:
            rfm_features: RFM features with segment labels
            
        Returns:
            DataFrame with feature importance scores
        """
        # Get feature columns (exclude 'segment')
        feature_cols = [col for col in rfm_features.columns if col != 'segment']
        
        # Standardize features
        rfm_scaled = self.scaler.transform(rfm_features[feature_cols])
        
        # Compute cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Compute variance of centers for each feature
        feature_variance = np.var(centers, axis=0)
        
        # Normalize to get importance scores (0-1)
        importance_scores = feature_variance / feature_variance.sum()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\nFeature Importance for Segmentation:")
        for _, row in importance_df.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def get_segment_strategies(self) -> Dict[str, Dict[str, str]]:
        """
        Get business strategies for each segment.
        
        Returns:
            Dictionary mapping segment names to strategy recommendations
        """
        strategies = {
            "Champions": {
                "priority": "High",
                "goal": "Retention & Advocacy",
                "actions": [
                    "Reward with exclusive benefits",
                    "Ask for reviews and referrals",
                    "Early access to new products",
                    "VIP customer service"
                ],
                "campaign_type": "Loyalty program, referral incentives",
                "budget_allocation": "Medium (focus on experience)"
            },
            
            "Loyal Customers": {
                "priority": "High",
                "goal": "Upsell & Cross-sell",
                "actions": [
                    "Recommend premium products",
                    "Bundle offers",
                    "Exclusive discounts",
                    "Appreciation campaigns"
                ],
                "campaign_type": "Product recommendations, premium offers",
                "budget_allocation": "Medium-High"
            },
            
            "At Risk": {
                "priority": "Critical",
                "goal": "Win-back before churn",
                "actions": [
                    "Personalized re-engagement campaigns",
                    "Limited-time offers",
                    "Survey to understand issues",
                    "Special discounts"
                ],
                "campaign_type": "Win-back emails, discount codes",
                "budget_allocation": "High (prevent churn)"
            },
            
            "Hibernating": {
                "priority": "Medium",
                "goal": "Reactivation",
                "actions": [
                    "Remind of past purchases",
                    "New product announcements",
                    "Aggressive discounts",
                    "Limited-time comeback offers"
                ],
                "campaign_type": "Reactivation campaigns, deep discounts",
                "budget_allocation": "Medium (ROI-dependent)"
            },
            
            "New Customers": {
                "priority": "High",
                "goal": "Onboarding & Second Purchase",
                "actions": [
                    "Welcome series",
                    "Product education",
                    "First-repeat purchase incentive",
                    "Build engagement"
                ],
                "campaign_type": "Onboarding emails, next-purchase coupon",
                "budget_allocation": "High (critical period)"
            },
            
            "Promising": {
                "priority": "Medium-High",
                "goal": "Accelerate to Champions",
                "actions": [
                    "Encourage higher frequency",
                    "Category expansion offers",
                    "Loyalty program enrollment",
                    "Engagement campaigns"
                ],
                "campaign_type": "Frequency incentives, cross-category offers",
                "budget_allocation": "Medium"
            },
            
            "Need Attention": {
                "priority": "Medium",
                "goal": "Prevent decline",
                "actions": [
                    "Engagement campaigns",
                    "Value propositions",
                    "Special offers",
                    "Feedback surveys"
                ],
                "campaign_type": "Engagement emails, moderate discounts",
                "budget_allocation": "Low-Medium"
            },
            
            "Lost/Churned": {
                "priority": "Low",
                "goal": "Low-cost reactivation attempts",
                "actions": [
                    "Minimal-cost reactivation",
                    "Only if high CLV",
                    "Generic re-engagement",
                    "Opt-out option"
                ],
                "campaign_type": "Automated win-back (low cost)",
                "budget_allocation": "Low (only high-CLV customers)"
            }
        }
        
        return strategies
    
    def assign_segments(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Assign segments to all customers in feature DataFrame.
        
        Args:
            features: DataFrame with RFM features
            
        Returns:
            DataFrame with segment assignments and names
        """
        # Predict segments
        segments = self.predict(features)
        
        # Create result DataFrame
        result = features.copy()
        result['segment'] = segments
        
        # Add segment names
        if self.segment_profiles is not None:
            segment_name_map = dict(zip(
                self.segment_profiles['segment'],
                self.segment_profiles['segment_name']
            ))
            result['segment_name'] = result['segment'].map(segment_name_map)
        
        return result
    
    def save_model(self, path: str):
        """
        Save fitted model to disk.
        
        Args:
            path: Output path (without extension)
        """
        if self.kmeans is None:
            raise ValueError("No model to save. Call fit() first.")
        
        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'n_clusters': self.n_clusters,
            'segment_profiles': self.segment_profiles,
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
        Load fitted model from disk.
        
        Args:
            path: Model path
            config: Configuration object
            logger: Logger instance
            
        Returns:
            Loaded RFMSegmenter instance
        """
        model_path = Path(path).with_suffix('.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(config, logger)
        
        # Restore model components
        instance.scaler = model_data['scaler']
        instance.kmeans = model_data['kmeans']
        instance.n_clusters = model_data['n_clusters']
        instance.segment_profiles = model_data['segment_profiles']
        instance.feature_importance = model_data['feature_importance']
        
        if logger:
            logger.info(f"Model loaded from: {model_path}")
        
        return instance


if __name__ == "__main__":
    print("RFM Segmentation Module")
    print("=" * 80)
    print("This module provides K-Means clustering for customer segmentation")
    print("Use the RFMSegmenter class to fit and predict customer segments")
