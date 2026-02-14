"""
RFM Segmentation Visualization Module
======================================
Creates publication-quality visualizations for RFM segment analysis.

Visualizations:
1. Segment distribution (bar chart)
2. RFM heatmap by segment
3. 3D scatter plot (R, F, M)
4. Segment revenue contribution
5. Feature importance
6. Segment comparison radar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import logging
from pathlib import Path


class RFMVisualizer:
    """Create visualizations for RFM segmentation analysis."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def plot_segment_distribution(
        self,
        segment_profiles: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot segment size distribution.
        
        Args:
            segment_profiles: DataFrame with segment profiles
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by size
        profiles_sorted = segment_profiles.sort_values('size', ascending=False)
        
        # Bar chart
        colors = sns.color_palette("husl", len(profiles_sorted))
        ax1.bar(
            profiles_sorted['segment_name'],
            profiles_sorted['size'],
            color=colors
        )
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Number of Customers')
        ax1.set_title('Customer Count by Segment')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (idx, row) in enumerate(profiles_sorted.iterrows()):
            ax1.text(
                i, row['size'], f"{row['size']:,}\n({row['size_pct']:.1f}%)",
                ha='center', va='bottom', fontsize=9
            )
        
        # Pie chart
        ax2.pie(
            profiles_sorted['size'],
            labels=profiles_sorted['segment_name'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax2.set_title('Segment Distribution (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved segment distribution plot to: {save_path}")
        
        return fig
    
    def plot_rfm_heatmap(
        self,
        segment_profiles: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of RFM metrics by segment.
        
        Args:
            segment_profiles: DataFrame with segment profiles
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        heatmap_data = segment_profiles[[
            'segment_name', 'avg_recency', 'avg_frequency', 'avg_monetary'
        ]].set_index('segment_name')
        
        # Normalize for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        heatmap_normalized.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            heatmap_normalized.T,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score (0-1)'},
            ax=ax
        )
        
        ax.set_title('RFM Metrics by Segment (Normalized)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved RFM heatmap to: {save_path}")
        
        return fig
    
    def plot_3d_scatter(
        self,
        features: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 3D scatter of customers colored by segment.
        
        Args:
            features: DataFrame with RFM features and segment assignments
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique segments
        segments = sorted(features['segment'].unique())
        colors = sns.color_palette("husl", len(segments))
        
        # Plot each segment
        for segment, color in zip(segments, colors):
            segment_data = features[features['segment'] == segment]
            segment_name = segment_data['segment_name'].iloc[0] if 'segment_name' in segment_data.columns else f"Segment {segment}"
            
            ax.scatter(
                segment_data['rfm_recency_days'],
                segment_data['rfm_frequency_count'],
                segment_data['rfm_monetary_avg'],
                c=[color],
                label=segment_name,
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('Recency (days)', fontsize=10)
        ax.set_ylabel('Frequency (transactions)', fontsize=10)
        ax.set_zlabel('Monetary (avg value)', fontsize=10)
        ax.set_title('Customer Segmentation in RFM Space', fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved 3D scatter plot to: {save_path}")
        
        return fig
    
    def plot_revenue_contribution(
        self,
        segment_profiles: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot revenue contribution by segment.
        
        Args:
            segment_profiles: DataFrame with segment profiles
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate revenue percentage
        total_revenue = segment_profiles['total_revenue'].sum()
        segment_profiles['revenue_pct'] = (segment_profiles['total_revenue'] / total_revenue) * 100
        
        # Sort by revenue
        profiles_sorted = segment_profiles.sort_values('total_revenue', ascending=False)
        
        # Create bar chart
        colors = sns.color_palette("viridis", len(profiles_sorted))
        bars = ax.bar(
            profiles_sorted['segment_name'],
            profiles_sorted['total_revenue'],
            color=colors
        )
        
        ax.set_xlabel('Segment', fontsize=12)
        ax.set_ylabel('Total Revenue ($)', fontsize=12)
        ax.set_title('Revenue Contribution by Segment', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (idx, row) in enumerate(profiles_sorted.iterrows()):
            ax.text(
                i, row['total_revenue'],
                f"${row['total_revenue']:,.0f}\n({row['revenue_pct']:.1f}%)",
                ha='center', va='bottom', fontsize=9
            )
        
        # Add horizontal line for average
        avg_revenue = segment_profiles['total_revenue'].mean()
        ax.axhline(avg_revenue, color='red', linestyle='--', linewidth=2, label=f'Average: ${avg_revenue:,.0f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved revenue contribution plot to: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance for segmentation.
        
        Args:
            feature_importance: DataFrame with feature importance scores
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        importance_sorted = feature_importance.sort_values('importance', ascending=True)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(importance_sorted['importance'] / importance_sorted['importance'].max())
        ax.barh(importance_sorted['feature'], importance_sorted['importance'], color=colors)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance for Segmentation', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (idx, row) in enumerate(importance_sorted.iterrows()):
            ax.text(
                row['importance'], i,
                f"{row['importance']:.4f}",
                va='center', ha='left', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance plot to: {save_path}")
        
        return fig
    
    def plot_segment_comparison(
        self,
        segment_profiles: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot radar chart comparing segments across multiple metrics.
        
        Args:
            segment_profiles: DataFrame with segment profiles
            metrics: List of metrics to compare (if None, uses default)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['avg_recency', 'avg_frequency', 'avg_monetary', 'avg_engagement_score']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in segment_profiles.columns]
        
        if len(available_metrics) < 3:
            self.logger.warning("Not enough metrics for radar chart")
            return None
        
        # Normalize metrics to 0-1 scale
        data_normalized = segment_profiles[available_metrics].copy()
        for col in available_metrics:
            data_normalized[col] = (data_normalized[col] - data_normalized[col].min()) / \
                                   (data_normalized[col].max() - data_normalized[col].min())
        
        # Create radar chart
        num_vars = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(segment_profiles))
        
        for i, (idx, row) in enumerate(segment_profiles.iterrows()):
            values = data_normalized.iloc[i].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=row['segment_name'])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Fix axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('avg_', '').replace('_', ' ').title() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Segment Comparison Across Key Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved segment comparison plot to: {save_path}")
        
        return fig
    
    def create_segment_report(
        self,
        segment_profiles: pd.DataFrame,
        features: pd.DataFrame,
        feature_importance: pd.DataFrame,
        output_dir: str
    ):
        """
        Create comprehensive visualization report.
        
        Args:
            segment_profiles: DataFrame with segment profiles
            features: DataFrame with customer features and segments
            feature_importance: DataFrame with feature importance
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating segment visualization report...")
        
        # 1. Distribution
        self.plot_segment_distribution(
            segment_profiles,
            save_path=output_path / 'segment_distribution.png'
        )
        plt.close()
        
        # 2. RFM Heatmap
        self.plot_rfm_heatmap(
            segment_profiles,
            save_path=output_path / 'rfm_heatmap.png'
        )
        plt.close()
        
        # 3. 3D Scatter (sample if too many points)
        if len(features) > 5000:
            features_sample = features.sample(n=5000, random_state=42)
        else:
            features_sample = features
        
        self.plot_3d_scatter(
            features_sample,
            save_path=output_path / 'rfm_3d_scatter.png'
        )
        plt.close()
        
        # 4. Revenue Contribution
        self.plot_revenue_contribution(
            segment_profiles,
            save_path=output_path / 'revenue_contribution.png'
        )
        plt.close()
        
        # 5. Feature Importance
        self.plot_feature_importance(
            feature_importance,
            save_path=output_path / 'feature_importance.png'
        )
        plt.close()
        
        # 6. Segment Comparison
        self.plot_segment_comparison(
            segment_profiles,
            save_path=output_path / 'segment_comparison_radar.png'
        )
        plt.close()
        
        self.logger.info(f"✓ Visualization report saved to: {output_dir}")


if __name__ == "__main__":
    print("RFM Visualization Module")
    print("=" * 80)
    print("Creates publication-quality visualizations for segment analysis")
