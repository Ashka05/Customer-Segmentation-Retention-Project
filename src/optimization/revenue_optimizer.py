"""
Revenue Optimization Module
===========================
Budget-constrained revenue maximization using churn and CLV predictions.

Key Components:
1. Expected value calculation (churn × CLV × retention_rate - cost)
2. Greedy budget allocation algorithm
3. Multi-tier campaign assignment
4. ROI calculation and reporting

Business Goal: Maximize retention revenue subject to budget constraint
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class Campaign:
    """Campaign definition with costs and expected retention rates."""
    name: str
    cost_per_customer: float
    expected_retention_rate: float  # Probability customer is saved if targeted
    description: str
    priority: int = 1  # Higher = more aggressive


class RevenueOptimizer:
    """
    Revenue optimization through optimal customer targeting.
    
    Methodology:
    1. Calculate expected value for each customer
    2. Rank customers by ROI
    3. Allocate budget greedily to maximize revenue
    4. Assign appropriate campaigns
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize revenue optimizer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Define campaign tiers
        self.campaigns = self._define_campaigns()
        
    def _define_campaigns(self) -> Dict[str, Campaign]:
        """
        Define campaign tiers with costs and retention rates.
        
        Returns:
            Dictionary of campaign definitions
        """
        campaigns = {
            'premium': Campaign(
                name='Premium Retention',
                cost_per_customer=50.0,
                expected_retention_rate=0.60,
                description='Personal outreach, 30% discount, dedicated support',
                priority=3
            ),
            'standard': Campaign(
                name='Standard Retention',
                cost_per_customer=25.0,
                expected_retention_rate=0.40,
                description='Automated email, 20% discount, standard support',
                priority=2
            ),
            'light': Campaign(
                name='Light Touch',
                cost_per_customer=10.0,
                expected_retention_rate=0.20,
                description='Generic email, 10% discount',
                priority=1
            ),
            'none': Campaign(
                name='No Intervention',
                cost_per_customer=0.0,
                expected_retention_rate=0.0,
                description='No retention campaign',
                priority=0
            )
        }
        
        self.logger.info("Campaign definitions:")
        for key, camp in campaigns.items():
            self.logger.info(
                f"  {camp.name}: Cost=${camp.cost_per_customer}, "
                f"Retention={camp.expected_retention_rate:.0%}"
            )
        
        return campaigns
    
    def calculate_expected_value(
        self,
        clv_predictions: pd.DataFrame,
        churn_predictions: pd.DataFrame,
        campaign: Campaign
    ) -> pd.DataFrame:
        """
        Calculate expected value for each customer under a campaign.
        
        Expected Value = (Churn_prob × CLV × Retention_rate) - Cost
        
        Args:
            clv_predictions: DataFrame with CLV predictions
            churn_predictions: DataFrame with churn probabilities
            campaign: Campaign to evaluate
            
        Returns:
            DataFrame with expected value calculations
        """
        # Merge predictions
        data = clv_predictions.join(
            churn_predictions[['churn_probability', 'risk_level']], 
            how='inner'
        )
        
        # Calculate expected revenue saved
        data['expected_loss_baseline'] = data['churn_probability'] * data['clv_prediction']
        
        data['expected_revenue_saved'] = (
            data['churn_probability'] * 
            data['clv_prediction'] * 
            campaign.expected_retention_rate
        )
        
        data['campaign_cost'] = campaign.cost_per_customer
        
        data['expected_value'] = data['expected_revenue_saved'] - data['campaign_cost']
        
        # ROI per customer
        data['roi'] = np.where(
            data['campaign_cost'] > 0,
            (data['expected_revenue_saved'] - data['campaign_cost']) / data['campaign_cost'],
            0
        )
        
        data['campaign_name'] = campaign.name
        data['expected_retention_rate'] = campaign.expected_retention_rate
        
        return data
    
    def optimize_budget_allocation(
        self,
        clv_predictions: pd.DataFrame,
        churn_predictions: pd.DataFrame,
        total_budget: float,
        min_roi: float = 0.0
    ) -> pd.DataFrame:
        """
        Optimize budget allocation across customers using greedy algorithm.
        
        Args:
            clv_predictions: DataFrame with CLV predictions
            churn_predictions: DataFrame with churn probabilities
            total_budget: Total marketing budget
            min_roi: Minimum ROI threshold
            
        Returns:
            DataFrame with optimal campaign assignments
        """
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZING BUDGET ALLOCATION")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Total budget: ${total_budget:,.2f}")
        self.logger.info(f"Minimum ROI threshold: {min_roi:.1%}")
        
        # Calculate expected value for each campaign tier
        all_options = []
        
        for campaign_key in ['premium', 'standard', 'light']:
            campaign = self.campaigns[campaign_key]
            ev_data = self.calculate_expected_value(
                clv_predictions, churn_predictions, campaign
            )
            ev_data['campaign_tier'] = campaign_key
            all_options.append(ev_data)
        
        # Combine all options
        all_options_df = pd.concat(all_options, ignore_index=False)
        
        # Filter by minimum ROI
        all_options_df = all_options_df[all_options_df['roi'] >= min_roi].copy()
        
        self.logger.info(f"\nCustomers with ROI >= {min_roi:.0%}: {len(all_options_df.index.unique()):,}")
        
        # Sort by expected value (descending)
        all_options_df = all_options_df.sort_values('expected_value', ascending=False).reset_index()
        
        # Greedy allocation
        allocated = []
        remaining_budget = total_budget
        allocated_customers = set()
        
        for idx, row in all_options_df.iterrows():
            customer_id = row['CustomerID']
            
            # Skip if customer already allocated
            if customer_id in allocated_customers:
                continue
            
            # Check if budget allows this campaign
            if remaining_budget >= row['campaign_cost']:
                allocated.append({
                    'CustomerID': customer_id,
                    'campaign_tier': row['campaign_tier'],
                    'campaign_name': row['campaign_name'],
                    'campaign_cost': row['campaign_cost'],
                    'churn_probability': row['churn_probability'],
                    'clv_prediction': row['clv_prediction'],
                    'expected_revenue_saved': row['expected_revenue_saved'],
                    'expected_value': row['expected_value'],
                    'roi': row['roi'],
                    'expected_retention_rate': row['expected_retention_rate']
                })
                
                remaining_budget -= row['campaign_cost']
                allocated_customers.add(customer_id)
                
                # Stop if budget exhausted
                if remaining_budget <= 0:
                    break
        
        # Create allocation DataFrame
        allocation_df = pd.DataFrame(allocated)
        
        if len(allocation_df) > 0:
            allocation_df.set_index('CustomerID', inplace=True)
        
        # Statistics
        budget_used = total_budget - remaining_budget
        total_expected_value = allocation_df['expected_value'].sum() if len(allocation_df) > 0 else 0
        total_expected_revenue = allocation_df['expected_revenue_saved'].sum() if len(allocation_df) > 0 else 0
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ALLOCATION RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Customers targeted: {len(allocation_df):,}")
        self.logger.info(f"Budget used: ${budget_used:,.2f} ({budget_used/total_budget*100:.1f}%)")
        self.logger.info(f"Budget remaining: ${remaining_budget:,.2f}")
        self.logger.info(f"Expected revenue saved: ${total_expected_revenue:,.2f}")
        self.logger.info(f"Expected net value: ${total_expected_value:,.2f}")
        self.logger.info(f"Expected ROI: {total_expected_value/budget_used*100:.1f}%" if budget_used > 0 else "N/A")
        
        # Campaign distribution
        if len(allocation_df) > 0:
            campaign_dist = allocation_df['campaign_tier'].value_counts()
            self.logger.info("\nCampaign distribution:")
            for tier, count in campaign_dist.items():
                cost = allocation_df[allocation_df['campaign_tier'] == tier]['campaign_cost'].sum()
                revenue = allocation_df[allocation_df['campaign_tier'] == tier]['expected_revenue_saved'].sum()
                self.logger.info(f"  {tier}: {count:,} customers, ${cost:,.2f} cost, ${revenue:,.2f} expected revenue")
        
        return allocation_df
    
    def calculate_portfolio_metrics(
        self,
        allocation: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio-level metrics and ROI.
        
        Args:
            allocation: Campaign allocation DataFrame
            
        Returns:
            Dictionary of portfolio metrics
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PORTFOLIO METRICS")
        self.logger.info("=" * 80)
        
        if len(allocation) == 0:
            self.logger.warning("No customers allocated")
            return {}
        
        # Total metrics
        total_cost = allocation['campaign_cost'].sum()
        total_expected_revenue = allocation['expected_revenue_saved'].sum()
        total_expected_value = allocation['expected_value'].sum()
        
        # Expected customers saved
        expected_customers_saved = (
            allocation['churn_probability'] * 
            allocation['expected_retention_rate']
        ).sum()
        
        # Portfolio ROI
        portfolio_roi = (total_expected_value / total_cost) if total_cost > 0 else 0
        
        metrics = {
            'total_customers_targeted': len(allocation),
            'total_campaign_cost': float(total_cost),
            'total_expected_revenue_saved': float(total_expected_revenue),
            'total_expected_net_value': float(total_expected_value),
            'portfolio_roi_pct': float(portfolio_roi * 100),
            'expected_customers_saved': float(expected_customers_saved),
            'cost_per_customer_saved': float(total_cost / expected_customers_saved) if expected_customers_saved > 0 else 0
        }
        
        # By campaign tier
        tier_metrics = {}
        for tier in allocation['campaign_tier'].unique():
            tier_data = allocation[allocation['campaign_tier'] == tier]
            tier_metrics[tier] = {
                'count': len(tier_data),
                'cost': float(tier_data['campaign_cost'].sum()),
                'expected_revenue': float(tier_data['expected_revenue_saved'].sum()),
                'expected_value': float(tier_data['expected_value'].sum())
            }
        
        metrics['by_campaign_tier'] = tier_metrics
        
        # Log results
        self.logger.info(f"Customers targeted: {metrics['total_customers_targeted']:,}")
        self.logger.info(f"Total cost: ${metrics['total_campaign_cost']:,.2f}")
        self.logger.info(f"Expected revenue saved: ${metrics['total_expected_revenue_saved']:,.2f}")
        self.logger.info(f"Expected net value: ${metrics['total_expected_net_value']:,.2f}")
        self.logger.info(f"Portfolio ROI: {metrics['portfolio_roi_pct']:.1f}%")
        self.logger.info(f"Expected customers saved: {metrics['expected_customers_saved']:.0f}")
        self.logger.info(f"Cost per customer saved: ${metrics['cost_per_customer_saved']:.2f}")
        
        return metrics


if __name__ == "__main__":
    print("Revenue Optimization Module")
    print("=" * 80)
    print("Budget-constrained optimization for retention campaigns")
    print("Maximizes expected value using churn and CLV predictions")
