"""
Business Representation Layer
==============================
Creates executive dashboards and business reports showing the impact
of the ML system on your actual customer data.

This translates technical outputs into business-friendly formats that
stakeholders can understand and act upon.

Usage:
    python src/business/create_reports.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BusinessReportGenerator:
    """Generates business-friendly reports and visualizations."""
    
    def __init__(self):
        self.output_dir = Path('outputs/business_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all outputs
        self.load_data()
    
    def load_data(self):
        """Load all system outputs."""
        print("Loading system outputs...")
        
        # Load data
        self.cleaned_data = pd.read_parquet('data/processed/cleaned_data.parquet')
        self.features = pd.read_parquet('data/features/customer_features.parquet')
        self.segments = pd.read_parquet('outputs/segment_assignments.parquet')
        self.churn = pd.read_parquet('outputs/churn_predictions.parquet')
        self.clv = pd.read_parquet('outputs/clv_predictions.parquet')
        self.allocation = pd.read_parquet('outputs/optimal_campaign_allocation.parquet')
        
        # Load metrics
        with open('outputs/churn_model_metrics.json') as f:
            self.churn_metrics = json.load(f)
        
        with open('outputs/clv_model_metrics.json') as f:
            self.clv_metrics = json.load(f)
        
        with open('outputs/optimization_metrics.json') as f:
            self.optimization_metrics = json.load(f)
        
        print(f"✓ Loaded data for {len(self.cleaned_data):,} transactions")
        print(f"✓ Loaded {len(self.segments):,} customer segments")
        print(f"✓ Loaded {len(self.churn):,} churn predictions")
        print(f"✓ Loaded {len(self.clv):,} CLV predictions")
        print(f"✓ Loaded {len(self.allocation):,} campaign allocations")
    
    def create_executive_summary(self):
        """Create executive summary dashboard."""
        print("\nCreating Executive Summary...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Executive Summary: Customer Retention Intelligence System', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Customer Distribution by Segment
        ax = axes[0, 0]
        if 'segment_name' in self.segments.columns:
            segment_counts = self.segments['segment_name'].value_counts()
            colors = plt.cm.Set3(range(len(segment_counts)))
            ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title('Customer Distribution by Segment', fontsize=14, fontweight='bold')
        
        # 2. Churn Risk Distribution
        ax = axes[0, 1]
        if 'risk_level' in self.churn.columns:
            risk_counts = self.churn['risk_level'].value_counts()
            risk_order = ['Low', 'Medium', 'High', 'Very High']
            risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
            colors = ['green', 'yellow', 'orange', 'red'][:len(risk_counts)]
            ax.bar(range(len(risk_counts)), risk_counts.values, color=colors)
            ax.set_xticks(range(len(risk_counts)))
            ax.set_xticklabels(risk_counts.index, rotation=45)
            ax.set_ylabel('Number of Customers')
            ax.set_title('Churn Risk Distribution', fontsize=14, fontweight='bold')
            
            # Add values on bars
            for i, v in enumerate(risk_counts.values):
                ax.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # 3. CLV Distribution
        ax = axes[0, 2]
        clv_values = self.clv['clv_prediction'].values
        ax.hist(clv_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('CLV Distribution', fontsize=14, fontweight='bold')
        ax.axvline(clv_values.mean(), color='red', linestyle='--', 
                   label=f'Mean: ${clv_values.mean():.2f}')
        ax.legend()
        
        # 4. Campaign Budget Allocation
        ax = axes[1, 0]
        if 'campaign_tier' in self.allocation.columns:
            campaign_counts = self.allocation['campaign_tier'].value_counts()
            campaign_costs = self.allocation.groupby('campaign_tier')['campaign_cost'].sum()
            
            x = range(len(campaign_counts))
            width = 0.35
            ax.bar([i - width/2 for i in x], campaign_counts.values, width, 
                   label='Customers', color='skyblue')
            ax2 = ax.twinx()
            ax2.bar([i + width/2 for i in x], campaign_costs.values, width, 
                    label='Budget ($)', color='lightcoral')
            
            ax.set_xticks(x)
            ax.set_xticklabels(campaign_counts.index)
            ax.set_ylabel('Number of Customers', color='skyblue')
            ax2.set_ylabel('Budget Allocated ($)', color='lightcoral')
            ax.set_title('Campaign Allocation', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 5. Expected Revenue Impact
        ax = axes[1, 1]
        metrics = self.optimization_metrics
        
        categories = ['Baseline\nChurn Loss', 'Campaign\nCost', 'Revenue\nSaved', 'Net\nValue']
        values = [
            -metrics.get('total_expected_revenue_saved', 0) / 0.6,  # Approximate baseline
            -metrics.get('total_campaign_cost', 0),
            metrics.get('total_expected_revenue_saved', 0),
            metrics.get('total_expected_net_value', 0)
        ]
        colors = ['red', 'orange', 'green', 'darkgreen']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Amount ($)')
        ax.set_title('Financial Impact Analysis', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${abs(val):,.0f}',
                    ha='center', va='bottom' if val > 0 else 'top')
        
        # 6. ROI Summary
        ax = axes[1, 2]
        ax.axis('off')
        
        # Key metrics
        roi = metrics.get('portfolio_roi_pct', 0)
        customers_saved = metrics.get('expected_customers_saved', 0)
        total_cost = metrics.get('total_campaign_cost', 0)
        net_value = metrics.get('total_expected_net_value', 0)
        
        summary_text = f"""
        KEY PERFORMANCE INDICATORS
        
        Portfolio ROI: {roi:.1f}%
        
        Customers Saved: {customers_saved:.0f}
        
        Campaign Investment: ${total_cost:,.0f}
        
        Expected Net Value: ${net_value:,.0f}
        
        Model Performance:
        • Churn ROC AUC: {self.churn_metrics.get('roc_auc', 0):.3f}
        • CLV R²: {self.clv_metrics.get('r2_score', 0):.3f}
        
        Customers Targeted: {len(self.allocation):,}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {self.output_dir / 'executive_summary.png'}")
        plt.close()
    
    def create_customer_action_list(self):
        """Create prioritized customer action list."""
        print("\nCreating Customer Action List...")
        
        # Merge all customer data
        customer_data = self.allocation.copy()
        
        # Add segment info
        if hasattr(self, 'segments') and 'segment_name' in self.segments.columns:
            customer_data = customer_data.join(
                self.segments[['segment_name']], 
                how='left'
            )
        
        # Sort by expected value (highest first)
        customer_data = customer_data.sort_values('expected_value', ascending=False)
        
        # Create action list
        action_list = customer_data[[
            'campaign_tier', 'campaign_cost', 'churn_probability', 
            'clv_prediction', 'expected_value', 'roi'
        ]].copy()
        
        # Add segment if available
        if 'segment_name' in customer_data.columns:
            action_list['segment'] = customer_data['segment_name']
        
        # Add priority rank
        action_list['priority_rank'] = range(1, len(action_list) + 1)
        
        # Add recommended action
        action_list['action'] = action_list['campaign_tier'].map({
            'premium': 'Personal Call + 30% Discount + VIP Support',
            'standard': 'Email Campaign + 20% Discount',
            'light': 'Automated Email + 10% Discount'
        })
        
        # Format for readability
        action_list['churn_risk'] = (action_list['churn_probability'] * 100).round(1).astype(str) + '%'
        action_list['expected_clv'] = '$' + action_list['clv_prediction'].round(2).astype(str)
        action_list['expected_return'] = '$' + action_list['expected_value'].round(2).astype(str)
        action_list['roi_pct'] = (action_list['roi'] * 100).round(1).astype(str) + '%'
        
        # Select final columns
        final_cols = ['priority_rank', 'campaign_tier', 'action', 'churn_risk', 
                     'expected_clv', 'expected_return', 'roi_pct']
        
        if 'segment' in action_list.columns:
            final_cols.insert(2, 'segment')
        
        action_list_final = action_list[final_cols]
        
        # Save top 100 to CSV
        action_list_final.head(100).to_csv(
            self.output_dir / 'top_100_customer_actions.csv'
        )
        
        # Save full list
        action_list_final.to_csv(
            self.output_dir / 'complete_customer_action_list.csv'
        )
        
        print(f"✓ Saved: {self.output_dir / 'top_100_customer_actions.csv'}")
        print(f"✓ Saved: {self.output_dir / 'complete_customer_action_list.csv'}")
        
        return action_list_final
    
    def create_segment_profiles(self):
        """Create detailed profiles for each segment."""
        print("\nCreating Segment Profiles...")
        
        if 'segment_name' not in self.segments.columns:
            print("No segment names found, skipping...")
            return
        
        # Merge segment data with churn and CLV
        segment_data = self.segments.copy()
        segment_data = segment_data.join(self.churn[['churn_probability']], how='left')
        segment_data = segment_data.join(self.clv[['clv_prediction']], how='left')
        
        # Create segment profiles
        profiles = []
        
        for segment in segment_data['segment_name'].unique():
            seg_customers = segment_data[segment_data['segment_name'] == segment]
            
            profile = {
                'Segment': segment,
                'Customer Count': len(seg_customers),
                'Percentage': f"{len(seg_customers)/len(segment_data)*100:.1f}%",
                'Avg Churn Risk': f"{seg_customers['churn_probability'].mean()*100:.1f}%",
                'Avg CLV': f"${seg_customers['clv_prediction'].mean():.2f}",
                'Total CLV': f"${seg_customers['clv_prediction'].sum():,.2f}",
                'High Risk Count': len(seg_customers[seg_customers['churn_probability'] > 0.7])
            }
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        profiles_df = profiles_df.sort_values('Customer Count', ascending=False)
        
        profiles_df.to_csv(self.output_dir / 'segment_profiles.csv', index=False)
        print(f"✓ Saved: {self.output_dir / 'segment_profiles.csv'}")
        
        return profiles_df
    
    def create_impact_report(self):
        """Create before/after comparison report."""
        print("\nCreating Impact Report...")
        
        # Calculate baseline (no intervention)
        total_customers = len(self.churn)
        baseline_churners = (self.churn['churn_probability'] > 0.5).sum()
        baseline_churn_rate = baseline_churners / total_customers
        
        # Calculate with intervention
        customers_targeted = len(self.allocation)
        expected_saved = self.optimization_metrics.get('expected_customers_saved', 0)
        new_churners = baseline_churners - expected_saved
        new_churn_rate = new_churners / total_customers
        
        # Revenue impact
        baseline_loss = (self.churn['churn_probability'] * self.clv['clv_prediction']).sum()
        campaign_cost = self.optimization_metrics.get('total_campaign_cost', 0)
        revenue_saved = self.optimization_metrics.get('total_expected_revenue_saved', 0)
        net_value = self.optimization_metrics.get('total_expected_net_value', 0)
        
        # Create comparison
        impact_data = {
            'Metric': [
                'Total Customers',
                'Customers Churning',
                'Churn Rate',
                'Revenue at Risk',
                'Retention Investment',
                'Revenue Saved',
                'Net Impact',
                'ROI'
            ],
            'Baseline (No Action)': [
                f'{total_customers:,}',
                f'{baseline_churners:,}',
                f'{baseline_churn_rate*100:.1f}%',
                f'${baseline_loss:,.0f}',
                '$0',
                '$0',
                f'-${baseline_loss:,.0f}',
                'N/A'
            ],
            'With ML Optimization': [
                f'{total_customers:,}',
                f'{new_churners:,.0f}',
                f'{new_churn_rate*100:.1f}%',
                f'${(baseline_loss - revenue_saved):,.0f}',
                f'${campaign_cost:,.0f}',
                f'${revenue_saved:,.0f}',
                f'${net_value:,.0f}',
                f"{self.optimization_metrics.get('portfolio_roi_pct', 0):.0f}%"
            ],
            'Improvement': [
                '→',
                f'-{expected_saved:.0f} ({expected_saved/baseline_churners*100:.0f}%)',
                f'-{(baseline_churn_rate - new_churn_rate)*100:.1f}%',
                f'${revenue_saved:,.0f}',
                f'-${campaign_cost:,.0f}',
                f'+${revenue_saved:,.0f}',
                f'+${(net_value + baseline_loss):,.0f}',
                f"+{self.optimization_metrics.get('portfolio_roi_pct', 0):.0f}%"
            ]
        }
        
        impact_df = pd.DataFrame(impact_data)
        impact_df.to_csv(self.output_dir / 'impact_analysis.csv', index=False)
        print(f"✓ Saved: {self.output_dir / 'impact_analysis.csv'}")
        
        return impact_df
    
    def create_business_summary_report(self):
        """Create comprehensive business summary."""
        print("\nCreating Business Summary Report...")
        
        summary = {
            'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Data Overview': {
                'Total Transactions Analyzed': f"{len(self.cleaned_data):,}",
                'Total Customers': f"{self.cleaned_data['CustomerID'].nunique():,}",
                'Date Range': f"{self.cleaned_data['InvoiceDate'].min().date()} to {self.cleaned_data['InvoiceDate'].max().date()}",
                'Total Revenue': f"${self.cleaned_data['Revenue'].sum():,.2f}"
            },
            'Segmentation Results': {
                'Number of Segments': f"{self.segments['segment_name'].nunique() if 'segment_name' in self.segments.columns else 'N/A'}",
                'Largest Segment': f"{self.segments['segment_name'].mode()[0] if 'segment_name' in self.segments.columns else 'N/A'}",
                'Smallest Segment': f"{self.segments['segment_name'].value_counts().idxmin() if 'segment_name' in self.segments.columns else 'N/A'}"
            },
            'Churn Analysis': {
                'Overall Churn Risk': f"{self.churn['churn_probability'].mean()*100:.1f}%",
                'High Risk Customers (>70%)': f"{(self.churn['churn_probability'] > 0.7).sum():,}",
                'Model ROC AUC': f"{self.churn_metrics.get('roc_auc', 0):.3f}",
                'Model Recall': f"{self.churn_metrics.get('recall', 0)*100:.1f}%"
            },
            'CLV Predictions': {
                'Average CLV': f"${self.clv['clv_prediction'].mean():.2f}",
                'Median CLV': f"${self.clv['clv_prediction'].median():.2f}",
                'Total Portfolio CLV': f"${self.clv['clv_prediction'].sum():,.2f}",
                'Model R²': f"{self.clv_metrics.get('r2_score', 0):.3f}"
            },
            'Campaign Optimization': {
                'Budget Allocated': f"${self.optimization_metrics.get('total_campaign_cost', 0):,.2f}",
                'Customers Targeted': f"{self.optimization_metrics.get('total_customers_targeted', 0):,}",
                'Expected Revenue Saved': f"${self.optimization_metrics.get('total_expected_revenue_saved', 0):,.2f}",
                'Expected Net Value': f"${self.optimization_metrics.get('total_expected_net_value', 0):,.2f}",
                'Portfolio ROI': f"{self.optimization_metrics.get('portfolio_roi_pct', 0):.1f}%",
                'Expected Customers Saved': f"{self.optimization_metrics.get('expected_customers_saved', 0):.0f}"
            },
            'Campaign Breakdown': {}
        }
        
        # Add campaign breakdown
        if 'campaign_tier' in self.allocation.columns:
            for tier in self.allocation['campaign_tier'].unique():
                tier_data = self.allocation[self.allocation['campaign_tier'] == tier]
                summary['Campaign Breakdown'][f'{tier.capitalize()} Campaign'] = {
                    'Customers': f"{len(tier_data):,}",
                    'Cost': f"${tier_data['campaign_cost'].sum():,.2f}",
                    'Expected Value': f"${tier_data['expected_value'].sum():,.2f}"
                }
        
        # Save as JSON
        with open(self.output_dir / 'business_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved: {self.output_dir / 'business_summary.json'}")
        
        return summary
    
    def generate_all_reports(self):
        """Generate all business reports."""
        print("=" * 80)
        print("GENERATING BUSINESS REPORTS")
        print("=" * 80)
        
        # Create all reports
        self.create_executive_summary()
        action_list = self.create_customer_action_list()
        segment_profiles = self.create_segment_profiles()
        impact_report = self.create_impact_report()
        summary = self.create_business_summary_report()
        
        print("\n" + "=" * 80)
        print("ALL REPORTS GENERATED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nReports saved to: {self.output_dir}")
        print("\nFiles created:")
        print("  1. executive_summary.png - Visual dashboard")
        print("  2. top_100_customer_actions.csv - Priority action list")
        print("  3. complete_customer_action_list.csv - Full customer list")
        print("  4. segment_profiles.csv - Segment analysis")
        print("  5. impact_analysis.csv - Before/after comparison")
        print("  6. business_summary.json - Complete metrics")
        
        return {
            'action_list': action_list,
            'segment_profiles': segment_profiles,
            'impact_report': impact_report,
            'summary': summary
        }


if __name__ == "__main__":
    generator = BusinessReportGenerator()
    reports = generator.generate_all_reports()
    
    print("\n✓ Business representation layer complete!")
    print("\nNext steps:")
    print("  1. Review executive_summary.png")
    print("  2. Use top_100_customer_actions.csv for immediate campaigns")
    print("  3. Share impact_analysis.csv with stakeholders")
