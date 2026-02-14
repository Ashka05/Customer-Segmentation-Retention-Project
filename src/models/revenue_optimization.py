"""
Revenue Optimization Module
===========================
Budget-constrained customer targeting for revenue maximization.

Combines:
- Churn predictions (Pillar 2)
- CLV predictions (Pillar 3)
- Customer segments (Pillar 1)

Optimizes:
- Which customers to target
- Which campaigns to assign
- How to allocate limited budget

Mathematical Formulation:
  Maximize: Σ (Expected Value per customer)
  Subject to: Σ (Campaign cost) ≤ Budget
  
  Where: Expected Value = Churn_prob × CLV × Retention_rate - Cost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path


class CampaignOptimizer:
    """
    Optimize customer targeting and campaign assignment for revenue maximization.
    
    Approach:
    1. Calculate expected value for each customer × campaign combination
    2. Use greedy algorithm to select optimal assignments within budget
    3. Generate campaign lists with expected ROI
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize campaign optimizer.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Define campaign tiers
        self.campaigns = self._define_campaigns()
        
    def _define_campaigns(self) -> pd.DataFrame:
        """
        Define campaign tiers with costs and retention rates.
        
        Returns:
            DataFrame with campaign definitions
        """
        # Default campaign definitions
        # In production, these would come from A/B test results
        campaigns = pd.DataFrame([
            {
                'tier': 1,
                'name': 'VIP Personal Outreach',
                'description': 'Personal call + 20% discount',
                'cost': 25.0,
                'retention_rate': 0.50,  # 50% success rate
                'min_clv': 500,
                'min_churn_prob': 0.6
            },
            {
                'tier': 2,
                'name': 'Premium Email Campaign',
                'description': 'Personalized email + 10% discount',
                'cost': 5.0,
                'retention_rate': 0.30,
                'min_clv': 200,
                'min_churn_prob': 0.5
            },
            {
                'tier': 3,
                'name': 'Standard Retention',
                'description': 'Email campaign + small incentive',
                'cost': 2.0,
                'retention_rate': 0.20,
                'min_clv': 100,
                'min_churn_prob': 0.3
            },
            {
                'tier': 4,
                'name': 'Basic Email',
                'description': 'Generic retention email',
                'cost': 1.0,
                'retention_rate': 0.10,
                'min_clv': 50,
                'min_churn_prob': 0.2
            },
            {
                'tier': 0,
                'name': 'No Action',
                'description': 'Do not target',
                'cost': 0.0,
                'retention_rate': 0.0,
                'min_clv': 0,
                'min_churn_prob': 0.0
            }
        ])
        
        return campaigns
    
    def calculate_expected_value(
        self,
        churn_prob: float,
        clv: float,
        campaign_cost: float,
        retention_rate: float
    ) -> float:
        """
        Calculate expected value of targeting a customer.
        
        Args:
            churn_prob: Probability customer will churn (0-1)
            clv: Customer lifetime value ($)
            campaign_cost: Cost of campaign ($)
            retention_rate: Probability campaign prevents churn (0-1)
            
        Returns:
            Expected value ($)
        """
        # Expected value = (Revenue saved by retention) - Cost
        # Revenue saved = Churn_prob × CLV × Retention_rate
        expected_revenue = churn_prob * clv * retention_rate
        expected_value = expected_revenue - campaign_cost
        
        return expected_value
    
    def assign_optimal_campaigns(
        self,
        customers: pd.DataFrame,
        budget: float,
        strategy: str = 'expected_value'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Assign optimal campaigns to customers within budget constraint.
        
        Args:
            customers: DataFrame with CustomerID, churn_probability, clv_prediction
            budget: Total budget available ($)
            strategy: 'expected_value' (default) or 'maximize_retention'
            
        Returns:
            Tuple of (campaign assignments DataFrame, optimization stats)
        """
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZING CAMPAIGN ASSIGNMENTS")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Total customers: {len(customers):,}")
        self.logger.info(f"Total budget: ${budget:,.2f}")
        self.logger.info(f"Optimization strategy: {strategy}")
        
        # Calculate expected value for each customer × campaign combination
        all_options = []
        
        for idx, customer in customers.iterrows():
            customer_id = idx
            churn_prob = customer['churn_probability']
            clv = customer.get('clv_prediction', customer.get('clv_risk_adjusted', 0))
            
            # Try each campaign tier
            for _, campaign in self.campaigns.iterrows():
                # Check eligibility
                if clv < campaign['min_clv']:
                    continue
                if churn_prob < campaign['min_churn_prob']:
                    continue
                
                # Calculate expected value
                ev = self.calculate_expected_value(
                    churn_prob=churn_prob,
                    clv=clv,
                    campaign_cost=campaign['cost'],
                    retention_rate=campaign['retention_rate']
                )
                
                # Only consider if EV > 0 (profitable)
                if ev > 0 or campaign['tier'] == 0:  # Always include "No Action"
                    all_options.append({
                        'CustomerID': customer_id,
                        'tier': campaign['tier'],
                        'campaign_name': campaign['name'],
                        'cost': campaign['cost'],
                        'retention_rate': campaign['retention_rate'],
                        'expected_value': ev,
                        'churn_prob': churn_prob,
                        'clv': clv
                    })
        
        options_df = pd.DataFrame(all_options)
        
        self.logger.info(f"\nGenerated {len(options_df):,} customer × campaign options")
        self.logger.info(f"Profitable options (EV > 0): {(options_df['expected_value'] > 0).sum():,}")
        
        # Greedy optimization
        assignments = self._greedy_allocation(options_df, budget, strategy)
        
        # Calculate statistics
        stats = self._calculate_optimization_stats(assignments, budget)
        
        return assignments, stats
    
    def _greedy_allocation(
        self,
        options: pd.DataFrame,
        budget: float,
        strategy: str
    ) -> pd.DataFrame:
        """
        Greedy algorithm: Select highest EV options until budget exhausted.
        
        Args:
            options: All customer × campaign options with expected values
            strategy: Optimization strategy
            
        Returns:
            Selected assignments
        """
        self.logger.info("\n--- Running Greedy Allocation Algorithm ---")
        
        # Sort by expected value (descending)
        if strategy == 'expected_value':
            options_sorted = options.sort_values('expected_value', ascending=False)
        elif strategy == 'maximize_retention':
            # Sort by (churn_prob × retention_rate) to maximize customers saved
            options['retention_impact'] = options['churn_prob'] * options['retention_rate']
            options_sorted = options.sort_values('retention_impact', ascending=False)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # For each customer, select best campaign
        selected = []
        spent = 0.0
        customers_assigned = set()
        
        for idx, row in options_sorted.iterrows():
            customer_id = row['CustomerID']
            
            # Skip if customer already assigned
            if customer_id in customers_assigned:
                continue
            
            # Check if we have budget
            if spent + row['cost'] <= budget:
                selected.append(row)
                spent += row['cost']
                customers_assigned.add(customer_id)
        
        # Assign "No Action" to unassigned customers
        all_customers = set(options['CustomerID'].unique())
        unassigned = all_customers - customers_assigned
        
        for customer_id in unassigned:
            customer_data = options[options['CustomerID'] == customer_id].iloc[0]
            selected.append({
                'CustomerID': customer_id,
                'tier': 0,
                'campaign_name': 'No Action',
                'cost': 0.0,
                'retention_rate': 0.0,
                'expected_value': 0.0,
                'churn_prob': customer_data['churn_prob'],
                'clv': customer_data['clv']
            })
        
        assignments = pd.DataFrame(selected)
        
        self.logger.info(f"\nAllocation complete:")
        self.logger.info(f"  Customers targeted: {len(customers_assigned):,}")
        self.logger.info(f"  Customers not targeted: {len(unassigned):,}")
        self.logger.info(f"  Budget spent: ${spent:,.2f} / ${budget:,.2f} ({spent/budget*100:.1f}%)")
        
        return assignments
    
    def _calculate_optimization_stats(
        self,
        assignments: pd.DataFrame,
        budget: float
    ) -> Dict:
        """
        Calculate optimization statistics and expected outcomes.
        
        Args:
            assignments: Campaign assignments
            budget: Total budget
            
        Returns:
            Statistics dictionary
        """
        # Filter to targeted customers only
        targeted = assignments[assignments['tier'] > 0]
        
        # Calculate expected outcomes
        total_cost = targeted['cost'].sum()
        total_expected_value = targeted['expected_value'].sum()
        expected_roi = (total_expected_value / total_cost * 100) if total_cost > 0 else 0
        
        # Expected customers retained
        expected_retained = (targeted['churn_prob'] * targeted['retention_rate']).sum()
        
        # Expected revenue saved
        expected_revenue_saved = (
            targeted['churn_prob'] * targeted['clv'] * targeted['retention_rate']
        ).sum()
        
        # Campaign distribution
        campaign_dist = assignments['campaign_name'].value_counts().to_dict()
        
        stats = {
            'total_customers': len(assignments),
            'customers_targeted': len(targeted),
            'customers_not_targeted': len(assignments) - len(targeted),
            'total_budget': float(budget),
            'budget_spent': float(total_cost),
            'budget_utilization_pct': float(total_cost / budget * 100) if budget > 0 else 0,
            'total_expected_value': float(total_expected_value),
            'expected_roi_pct': float(expected_roi),
            'expected_customers_retained': float(expected_retained),
            'expected_revenue_saved': float(expected_revenue_saved),
            'campaign_distribution': campaign_dist
        }
        
        # Log summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("OPTIMIZATION RESULTS")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\nBudget:")
        self.logger.info(f"  Total budget: ${stats['total_budget']:,.2f}")
        self.logger.info(f"  Spent: ${stats['budget_spent']:,.2f} ({stats['budget_utilization_pct']:.1f}%)")
        
        self.logger.info(f"\nTargeting:")
        self.logger.info(f"  Customers targeted: {stats['customers_targeted']:,}")
        self.logger.info(f"  Customers not targeted: {stats['customers_not_targeted']:,}")
        
        self.logger.info(f"\nExpected Outcomes:")
        self.logger.info(f"  Expected value: ${stats['total_expected_value']:,.2f}")
        self.logger.info(f"  Expected ROI: {stats['expected_roi_pct']:.1f}%")
        self.logger.info(f"  Expected customers retained: {stats['expected_customers_retained']:.0f}")
        self.logger.info(f"  Expected revenue saved: ${stats['expected_revenue_saved']:,.2f}")
        
        self.logger.info(f"\nCampaign Distribution:")
        for campaign, count in sorted(campaign_dist.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(assignments) * 100
            self.logger.info(f"  {campaign}: {count:,} ({pct:.1f}%)")
        
        return stats
    
    def compare_strategies(
        self,
        customers: pd.DataFrame,
        budget: float
    ) -> pd.DataFrame:
        """
        Compare different targeting strategies.
        
        Args:
            customers: Customer DataFrame
            budget: Budget amount
            
        Returns:
            Comparison DataFrame
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARING TARGETING STRATEGIES")
        self.logger.info("=" * 80)
        
        strategies = []
        
        # Strategy 1: Optimize for expected value
        assignments_ev, stats_ev = self.assign_optimal_campaigns(
            customers, budget, strategy='expected_value'
        )
        strategies.append({
            'strategy': 'Expected Value',
            **stats_ev
        })
        
        # Strategy 2: Naive - target all high churn
        high_churn = customers[customers['churn_probability'] > 0.5].copy()
        if len(high_churn) > 0:
            cost_per_customer = 5.0  # Average cost
            n_targeted = min(len(high_churn), int(budget / cost_per_customer))
            
            strategies.append({
                'strategy': 'Naive (All High Churn)',
                'total_customers': len(customers),
                'customers_targeted': n_targeted,
                'budget_spent': n_targeted * cost_per_customer,
                'expected_roi_pct': 50.0  # Assumed lower ROI
            })
        
        # Strategy 3: Random targeting
        strategies.append({
            'strategy': 'Random Targeting',
            'total_customers': len(customers),
            'customers_targeted': int(budget / 5.0),
            'budget_spent': budget,
            'expected_roi_pct': 20.0  # Assumed very low ROI
        })
        
        comparison = pd.DataFrame(strategies)
        
        self.logger.info("\nStrategy Comparison:")
        self.logger.info(comparison.to_string(index=False))
        
        return comparison
    
    def generate_campaign_lists(
        self,
        assignments: pd.DataFrame,
        output_dir: str
    ):
        """
        Generate campaign execution lists for each tier.
        
        Args:
            assignments: Campaign assignments
            output_dir: Directory to save campaign lists
        """
        self.logger.info("\n--- Generating Campaign Lists ---")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group by campaign
        for campaign_name in assignments['campaign_name'].unique():
            if campaign_name == 'No Action':
                continue
            
            campaign_customers = assignments[assignments['campaign_name'] == campaign_name].copy()
            
            # Sort by expected value (highest first)
            campaign_customers = campaign_customers.sort_values('expected_value', ascending=False)
            
            # Create execution list
            execution_list = campaign_customers[[
                'CustomerID', 'churn_prob', 'clv', 'expected_value'
            ]].copy()
            
            execution_list.columns = [
                'CustomerID', 'Churn Probability', 'CLV', 'Expected Value'
            ]
            
            # Save
            filename = campaign_name.lower().replace(' ', '_') + '_campaign_list.csv'
            filepath = output_path / filename
            execution_list.to_csv(filepath, index=False)
            
            self.logger.info(f"  {campaign_name}: {len(execution_list):,} customers → {filepath.name}")


if __name__ == "__main__":
    print("Revenue Optimization Module")
    print("=" * 80)
    print("Budget-constrained customer targeting for revenue maximization")
    print("Combines churn predictions, CLV predictions, and campaign economics")
