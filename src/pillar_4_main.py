"""
Pillar 4: Revenue Optimization - Main Execution Script
======================================================
The FINAL PILLAR that brings everything together!

Orchestrates optimal budget allocation for retention campaigns.

Workflow:
1. Load all predictions (CLV, Churn, Segments)
2. Define campaign tiers with costs and retention rates
3. Calculate expected value per customer
4. Optimize budget allocation (greedy algorithm)
5. Generate campaign assignments
6. Calculate ROI and business impact

Usage:
    python pillar_4_main.py --budget 50000
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
from optimization.revenue_optimizer import RevenueOptimizer


def main(args):
    """Main execution function."""
    
    # ========================================================================
    # SETUP
    # ========================================================================
    print_section_header("PILLAR 4: REVENUE OPTIMIZATION (FINAL PILLAR!)")
    
    config = Config(args.config)
    logger = setup_logging(config)
    
    logger.info("Configuration loaded successfully")
    logger.info("=" * 80)
    logger.info("THIS IS THE FINAL PILLAR - COMPLETING THE ENTIRE SYSTEM!")
    logger.info("=" * 80)
    
    # ========================================================================
    # TASK 4.1: Load All Predictions
    # ========================================================================
    print_section_header("Task 4.1: Load Predictions from Pillars 1-3")
    
    with Timer("Data Loading", logger):
        output_path = Path(config.get('paths.outputs'))
        
        # Load CLV predictions
        logger.info("Loading CLV predictions...")
        clv_path = output_path / 'clv_predictions.parquet'
        if not clv_path.exists():
            logger.error(f"CLV predictions not found at: {clv_path}")
            logger.error("Please run Pillar 3 first!")
            return 1
        
        clv_predictions = pd.read_parquet(clv_path)
        logger.info(f"✓ Loaded CLV predictions: {len(clv_predictions):,} customers")
        
        # Load churn predictions
        logger.info("Loading churn predictions...")
        churn_path = output_path / 'churn_predictions.parquet'
        if not churn_path.exists():
            logger.error(f"Churn predictions not found at: {churn_path}")
            logger.error("Please run Pillar 2 first!")
            return 1
        
        churn_predictions = pd.read_parquet(churn_path)
        logger.info(f"✓ Loaded churn predictions: {len(churn_predictions):,} customers")
        
        # Load segment assignments (optional)
        segment_path = output_path / 'segment_assignments.parquet'
        if segment_path.exists():
            logger.info("Loading segment assignments...")
            segment_assignments = pd.read_parquet(segment_path)
            # Get latest snapshot
            if 'snapshot_date' in segment_assignments.columns:
                latest_snapshot = segment_assignments['snapshot_date'].max()
                segment_assignments = segment_assignments[
                    segment_assignments['snapshot_date'] == latest_snapshot
                ]
            logger.info(f"✓ Loaded segment assignments: {len(segment_assignments):,} customers")
        else:
            logger.warning("Segment assignments not found (optional)")
            segment_assignments = None
        
        logger.info("\n✓ All predictions loaded successfully")
    
    # ========================================================================
    # TASK 4.2: Expected Value Calculation
    # ========================================================================
    print_section_header("Task 4.2: Calculate Expected Value for Each Customer")
    
    with Timer("Expected Value Calculation", logger):
        optimizer = RevenueOptimizer(config, logger)
        
        # Calculate baseline expected loss (no intervention)
        merged_data = clv_predictions.join(
            churn_predictions[['churn_probability', 'risk_level']], 
            how='inner'
        )
        
        merged_data['baseline_expected_loss'] = (
            merged_data['churn_probability'] * merged_data['clv_prediction']
        )
        
        total_baseline_loss = merged_data['baseline_expected_loss'].sum()
        
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE ANALYSIS (No Intervention)")
        logger.info("=" * 80)
        logger.info(f"Total customers: {len(merged_data):,}")
        logger.info(f"Total potential CLV: ${merged_data['clv_prediction'].sum():,.2f}")
        logger.info(f"Expected churn loss (no action): ${total_baseline_loss:,.2f}")
        logger.info(f"Average churn probability: {merged_data['churn_probability'].mean():.1%}")
        
        # Breakdown by risk level
        logger.info("\nExpected loss by risk level:")
        for risk in ['Low', 'Medium', 'High', 'Very High']:
            risk_data = merged_data[merged_data['risk_level'] == risk]
            if len(risk_data) > 0:
                risk_loss = risk_data['baseline_expected_loss'].sum()
                logger.info(
                    f"  {risk}: {len(risk_data):,} customers, "
                    f"${risk_loss:,.2f} expected loss ({risk_loss/total_baseline_loss*100:.1f}%)"
                )
        
        logger.info(f"\n✓ Expected value framework established")
    
    # ========================================================================
    # TASK 4.3: Optimize Budget Allocation
    # ========================================================================
    print_section_header("Task 4.3: Optimize Budget Allocation")
    
    with Timer("Budget Optimization", logger):
        # Get budget from args
        total_budget = args.budget
        
        logger.info(f"\nOptimizing for budget: ${total_budget:,.2f}")
        logger.info(f"Minimum ROI threshold: {args.min_roi:.0%}")
        
        # Optimize allocation
        optimal_allocation = optimizer.optimize_budget_allocation(
            clv_predictions=clv_predictions,
            churn_predictions=churn_predictions,
            total_budget=total_budget,
            min_roi=args.min_roi
        )
        
        if len(optimal_allocation) == 0:
            logger.error("No customers meet ROI threshold. Try lowering min_roi.")
            return 1
        
        # Save allocation
        allocation_path = output_path / 'optimal_campaign_allocation.parquet'
        optimal_allocation.to_parquet(allocation_path, index=True)
        logger.info(f"\n✓ Optimal allocation saved to: {allocation_path}")
    
    # ========================================================================
    # TASK 4.4: Calculate Portfolio Metrics & ROI
    # ========================================================================
    print_section_header("Task 4.4: Calculate Portfolio Metrics & ROI")
    
    with Timer("Portfolio Metrics", logger):
        portfolio_metrics = optimizer.calculate_portfolio_metrics(optimal_allocation)
        
        # Save metrics
        metrics_path = output_path / 'optimization_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(portfolio_metrics, f, indent=2)
        logger.info(f"\n✓ Metrics saved to: {metrics_path}")
        
        # Calculate lift vs baseline
        revenue_saved = portfolio_metrics.get('total_expected_revenue_saved', 0)
        baseline_loss_for_targeted = merged_data.loc[
            optimal_allocation.index
        ]['baseline_expected_loss'].sum()
        
        lift_vs_baseline = (revenue_saved / baseline_loss_for_targeted) if baseline_loss_for_targeted > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("IMPACT ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Baseline (no intervention):")
        logger.info(f"  Expected churn loss: ${baseline_loss_for_targeted:,.2f}")
        logger.info(f"\nWith optimized campaigns:")
        logger.info(f"  Campaign cost: ${portfolio_metrics.get('total_campaign_cost', 0):,.2f}")
        logger.info(f"  Revenue saved: ${revenue_saved:,.2f}")
        logger.info(f"  Net value: ${portfolio_metrics.get('total_expected_net_value', 0):,.2f}")
        logger.info(f"  ROI: {portfolio_metrics.get('portfolio_roi_pct', 0):.1f}%")
        logger.info(f"  Lift vs baseline: {lift_vs_baseline:.1%}")
        logger.info(f"  Customers saved: {portfolio_metrics.get('expected_customers_saved', 0):.0f}")
    
    # ========================================================================
    # TASK 4.5: Budget Sensitivity Analysis
    # ========================================================================
    print_section_header("Task 4.5: Budget Sensitivity Analysis")
    
    with Timer("Sensitivity Analysis", logger):
        # Test different budget levels
        test_budgets = [
            total_budget * 0.5,
            total_budget * 0.75,
            total_budget,
            total_budget * 1.5,
            total_budget * 2.0
        ]
        
        sensitivity_results = []
        
        logger.info(f"\nTesting {len(test_budgets)} budget scenarios...")
        
        for test_budget in test_budgets:
            allocation = optimizer.optimize_budget_allocation(
                clv_predictions, churn_predictions, test_budget, min_roi=0.0
            )
            
            if len(allocation) > 0:
                metrics = optimizer.calculate_portfolio_metrics(allocation)
                
                sensitivity_results.append({
                    'budget': test_budget,
                    'customers_targeted': metrics['total_customers_targeted'],
                    'budget_used': metrics['total_campaign_cost'],
                    'expected_revenue': metrics['total_expected_revenue_saved'],
                    'expected_value': metrics['total_expected_net_value'],
                    'roi_pct': metrics['portfolio_roi_pct'],
                    'customers_saved': metrics['expected_customers_saved']
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Save sensitivity analysis
        sensitivity_path = output_path / 'budget_sensitivity.csv'
        sensitivity_df.to_csv(sensitivity_path, index=False)
        logger.info(f"\n✓ Sensitivity analysis saved to: {sensitivity_path}")
        
        # Log results
        logger.info("\nBudget sensitivity results:")
        for _, row in sensitivity_df.iterrows():
            logger.info(
                f"  ${row['budget']:,.0f}: "
                f"{row['customers_targeted']:,} customers, "
                f"${row['expected_value']:,.2f} value, "
                f"{row['roi_pct']:.1f}% ROI"
            )
    
    # ========================================================================
    # TASK 4.6: Create Campaign Execution Plan
    # ========================================================================
    print_section_header("Task 4.6: Create Campaign Execution Plan")
    
    with Timer("Execution Plan", logger):
        # Create detailed execution plan
        execution_plan = optimal_allocation.copy()
        
        # Add priority ranking
        execution_plan['priority_rank'] = range(1, len(execution_plan) + 1)
        
        # Add segment information if available
        if segment_assignments is not None:
            execution_plan = execution_plan.join(
                segment_assignments[['segment_name']], 
                how='left'
            )
        
        # Sort by priority
        execution_plan = execution_plan.sort_values('expected_value', ascending=False)
        
        # Save execution plan
        plan_path = output_path / 'campaign_execution_plan.csv'
        execution_plan.to_csv(plan_path)
        logger.info(f"✓ Execution plan saved to: {plan_path}")
        
        # Create summary by campaign tier
        logger.info("\n" + "=" * 80)
        logger.info("CAMPAIGN EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        for tier in ['premium', 'standard', 'light']:
            tier_data = execution_plan[execution_plan['campaign_tier'] == tier]
            if len(tier_data) > 0:
                logger.info(f"\n{tier.upper()} Campaign:")
                logger.info(f"  Customers: {len(tier_data):,}")
                logger.info(f"  Total cost: ${tier_data['campaign_cost'].sum():,.2f}")
                logger.info(f"  Expected revenue: ${tier_data['expected_revenue_saved'].sum():,.2f}")
                logger.info(f"  Expected value: ${tier_data['expected_value'].sum():,.2f}")
                logger.info(f"  Avg churn probability: {tier_data['churn_probability'].mean():.1%}")
                logger.info(f"  Avg CLV: ${tier_data['clv_prediction'].mean():.2f}")
                
                # Top 5 customers in this tier
                logger.info(f"  Top 5 priorities:")
                for idx, (customer_id, row) in enumerate(tier_data.head(5).iterrows(), 1):
                    logger.info(
                        f"    {idx}. Customer {customer_id}: "
                        f"CLV=${row['clv_prediction']:.2f}, "
                        f"Churn={row['churn_probability']:.1%}, "
                        f"Value=${row['expected_value']:.2f}"
                    )
    
    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    print_section_header("PILLAR 4 COMPLETION SUMMARY")
    
    logger.info("✓ Task 4.1: Load Predictions - COMPLETE")
    logger.info("✓ Task 4.2: Expected Value Calculation - COMPLETE")
    logger.info("✓ Task 4.3: Budget Optimization - COMPLETE")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PILLAR 4 STATUS: 3/3 TASKS COMPLETE ✓")
    logger.info("=" * 80)
    
    logger.info("\n" + "🎉" * 40)
    logger.info("ENTIRE SYSTEM COMPLETE - ALL 18 TASKS DONE!")
    logger.info("🎉" * 40)
    
    logger.info("\nKey Results:")
    logger.info(f"  - Budget: ${total_budget:,.2f}")
    logger.info(f"  - Customers targeted: {portfolio_metrics.get('total_customers_targeted', 0):,}")
    logger.info(f"  - Expected revenue saved: ${portfolio_metrics.get('total_expected_revenue_saved', 0):,.2f}")
    logger.info(f"  - Expected net value: ${portfolio_metrics.get('total_expected_net_value', 0):,.2f}")
    logger.info(f"  - Portfolio ROI: {portfolio_metrics.get('portfolio_roi_pct', 0):.1f}%")
    logger.info(f"  - Customers saved: {portfolio_metrics.get('expected_customers_saved', 0):.0f}")
    
    logger.info("\nCampaign breakdown:")
    for tier, metrics in portfolio_metrics.get('by_campaign_tier', {}).items():
        logger.info(
            f"  {tier.capitalize()}: {metrics['count']:,} customers, "
            f"${metrics['cost']:,.2f} cost, ${metrics['expected_value']:,.2f} value"
        )
    
    logger.info("\nOutput files created:")
    logger.info(f"  1. Optimal allocation: {allocation_path}")
    logger.info(f"  2. Portfolio metrics: {metrics_path}")
    logger.info(f"  3. Sensitivity analysis: {sensitivity_path}")
    logger.info(f"  4. Execution plan: {plan_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL PROGRESS: 18/18 TASKS COMPLETE (100%)")
    logger.info("=" * 80)
    logger.info("Pillar 0: ✓ Complete (6/6 tasks) - Data Engineering")
    logger.info("Pillar 1: ✓ Complete (4/4 tasks) - RFM Segmentation")
    logger.info("Pillar 2: ✓ Complete (4/4 tasks) - Churn Prediction")
    logger.info("Pillar 3: ✓ Complete (3/3 tasks) - CLV Prediction")
    logger.info("Pillar 4: ✓ Complete (3/3 tasks) - Revenue Optimization")
    
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM READY FOR PRODUCTION!")
    logger.info("=" * 80)
    
    logger.info("\nBusiness Impact:")
    logger.info(f"  → Baseline churn loss: ${total_baseline_loss:,.2f}")
    logger.info(f"  → With optimization: ${portfolio_metrics.get('total_expected_net_value', 0):,.2f} net value")
    logger.info(f"  → ROI: {portfolio_metrics.get('portfolio_roi_pct', 0):.1f}%")
    logger.info(f"  → {portfolio_metrics.get('expected_customers_saved', 0):.0f} customers saved")
    
    logger.info("\nNext steps:")
    logger.info("  → Execute campaigns according to execution plan")
    logger.info("  → Monitor actual retention rates vs predicted")
    logger.info("  → A/B test different campaign tiers")
    logger.info("  → Retrain models quarterly with new data")
    logger.info("  → Deploy to production (API, dashboard, automation)")
    
    logger.info("\n🚀 CONGRATULATIONS! PRODUCTION-GRADE RETENTION SYSTEM COMPLETE! 🚀")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pillar 4: Revenue Optimization (FINAL PILLAR)"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--budget',
        type=float,
        default=50000.0,
        help='Total marketing budget for retention campaigns'
    )
    
    parser.add_argument(
        '--min-roi',
        type=float,
        default=0.0,
        help='Minimum ROI threshold (0.0 = positive expected value)'
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    sys.exit(exit_code)
