"""
System Validation Tests
=======================
Comprehensive tests to verify the entire retention system works correctly.

Run this to validate your system before deployment or GitHub upload.

Usage:
    python tests/validate_system.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.utils import Config


class SystemValidator:
    """Validates the complete retention system."""
    
    def __init__(self):
        self.config = Config('config/config.yaml')
        self.results = {}
        self.passed = 0
        self.failed = 0
        
    def print_header(self, text):
        """Print section header."""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    
    def test(self, name, condition, message=""):
        """Run a single test."""
        if condition:
            print(f"✅ PASS: {name}")
            self.passed += 1
            self.results[name] = {"status": "PASS", "message": message}
        else:
            print(f"❌ FAIL: {name}")
            if message:
                print(f"   {message}")
            self.failed += 1
            self.results[name] = {"status": "FAIL", "message": message}
    
    def validate_pillar_0(self):
        """Validate Pillar 0: Data Engineering."""
        self.print_header("PILLAR 0: DATA ENGINEERING")
        
        # Test 1: Cleaned data exists
        cleaned_path = Path('data/processed/cleaned_data.parquet')
        self.test(
            "Cleaned data file exists",
            cleaned_path.exists(),
            f"Expected: {cleaned_path}"
        )
        
        if cleaned_path.exists():
            df = pd.read_parquet(cleaned_path)
            
            # Test 2: Data shape
            self.test(
                "Cleaned data has rows",
                len(df) > 0,
                f"Found {len(df):,} rows"
            )
            
            # Test 3: Required columns
            required_cols = ['CustomerID', 'InvoiceDate', 'Revenue']
            has_cols = all(col in df.columns for col in required_cols)
            self.test(
                "Required columns present",
                has_cols,
                f"Has: {', '.join(required_cols)}"
            )
            
            # Test 4: No null CustomerIDs
            self.test(
                "No null CustomerIDs",
                df['CustomerID'].notna().all(),
                f"Nulls: {df['CustomerID'].isna().sum()}"
            )
        
        # Test 5: Features exist
        features_path = Path('data/features/customer_features.parquet')
        self.test(
            "Feature store exists",
            features_path.exists(),
            f"Expected: {features_path}"
        )
        
        if features_path.exists():
            features = pd.read_parquet(features_path)
            
            # Test 6: Features shape
            self.test(
                "Features have customers",
                len(features) > 0,
                f"Found {len(features):,} feature rows"
            )
            
            # Test 7: Feature count
            feature_cols = [col for col in features.columns if col not in ['snapshot_date', 'observation_start_date']]
            self.test(
                "Has ~33 features",
                25 <= len(feature_cols) <= 40,
                f"Found {len(feature_cols)} features"
            )
    
    def validate_pillar_1(self):
        """Validate Pillar 1: RFM Segmentation."""
        self.print_header("PILLAR 1: RFM SEGMENTATION")
        
        # Test 1: Segments exist
        segments_path = Path('outputs/segment_assignments.parquet')
        self.test(
            "Segment assignments exist",
            segments_path.exists(),
            f"Expected: {segments_path}"
        )
        
        if segments_path.exists():
            segments = pd.read_parquet(segments_path)
            
            # Test 2: Has segments
            self.test(
                "Customers are segmented",
                len(segments) > 0,
                f"Segmented {len(segments):,} customers"
            )
            
            # Test 3: Segment names
            if 'segment_name' in segments.columns:
                unique_segments = segments['segment_name'].nunique()
                self.test(
                    "Has 5-10 segments",
                    5 <= unique_segments <= 10,
                    f"Found {unique_segments} segments"
                )
        
        # Test 4: Strategies exist
        strategies_path = Path('outputs/segment_strategies.json')
        self.test(
            "Segment strategies exist",
            strategies_path.exists(),
            f"Expected: {strategies_path}"
        )
        
        # Test 5: Visualizations exist
        viz_dir = Path('outputs/visualizations/rfm')
        if viz_dir.exists():
            viz_files = list(viz_dir.glob('*.png'))
            self.test(
                "RFM visualizations created",
                len(viz_files) >= 4,
                f"Found {len(viz_files)} visualization files"
            )
        else:
            self.test("RFM visualizations exist", False, "Directory not found")
        
        # Test 6: Model saved
        model_path = Path('models/rfm_segmenter.pkl')
        self.test(
            "RFM model saved",
            model_path.exists(),
            f"Expected: {model_path}"
        )
    
    def validate_pillar_2(self):
        """Validate Pillar 2: Churn Prediction."""
        self.print_header("PILLAR 2: CHURN PREDICTION")
        
        # Test 1: Churn labels exist
        labels_path = Path('outputs/churn_labels.parquet')
        self.test(
            "Churn labels created",
            labels_path.exists(),
            f"Expected: {labels_path}"
        )
        
        if labels_path.exists():
            labels = pd.read_parquet(labels_path)
            
            # Test 2: Labels have churn column
            self.test(
                "Has churn_label column",
                'churn_label' in labels.columns,
                "churn_label column present"
            )
            
            # Test 3: Churn rate reasonable
            if 'churn_label' in labels.columns:
                churn_rate = labels['churn_label'].mean()
                self.test(
                    "Churn rate is reasonable",
                    0.05 <= churn_rate <= 0.95,
                    f"Churn rate: {churn_rate:.1%}"
                )
        
        # Test 4: Predictions exist
        predictions_path = Path('outputs/churn_predictions.parquet')
        self.test(
            "Churn predictions exist",
            predictions_path.exists(),
            f"Expected: {predictions_path}"
        )
        
        if predictions_path.exists():
            predictions = pd.read_parquet(predictions_path)
            
            # Test 5: Has probability
            self.test(
                "Has churn probabilities",
                'churn_probability' in predictions.columns,
                "churn_probability column present"
            )
            
            # Test 6: Probabilities valid
            if 'churn_probability' in predictions.columns:
                valid_probs = (predictions['churn_probability'] >= 0) & (predictions['churn_probability'] <= 1)
                self.test(
                    "Probabilities in [0, 1]",
                    valid_probs.all(),
                    f"{valid_probs.sum()}/{len(predictions)} valid"
                )
        
        # Test 7: Model saved
        model_path = Path('models/churn_predictor.pkl')
        self.test(
            "Churn model saved",
            model_path.exists(),
            f"Expected: {model_path}"
        )
        
        # Test 8: Metrics exist
        metrics_path = Path('outputs/churn_model_metrics.json')
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Test 9: ROC AUC reasonable
            if 'roc_auc' in metrics:
                self.test(
                    "ROC AUC > 0.5",
                    metrics['roc_auc'] > 0.5,
                    f"ROC AUC: {metrics['roc_auc']:.4f}"
                )
        else:
            self.test("Metrics file exists", False, "File not found")
    
    def validate_pillar_3(self):
        """Validate Pillar 3: CLV Prediction."""
        self.print_header("PILLAR 3: CLV PREDICTION")
        
        # Test 1: CLV targets exist
        targets_path = Path('outputs/clv_targets.parquet')
        self.test(
            "CLV targets created",
            targets_path.exists(),
            f"Expected: {targets_path}"
        )
        
        # Test 2: Predictions exist
        predictions_path = Path('outputs/clv_predictions.parquet')
        self.test(
            "CLV predictions exist",
            predictions_path.exists(),
            f"Expected: {predictions_path}"
        )
        
        if predictions_path.exists():
            predictions = pd.read_parquet(predictions_path)
            
            # Test 3: Has CLV column
            self.test(
                "Has clv_prediction column",
                'clv_prediction' in predictions.columns,
                "clv_prediction column present"
            )
            
            # Test 4: CLV values reasonable
            if 'clv_prediction' in predictions.columns:
                clv_positive = (predictions['clv_prediction'] >= 0).all()
                self.test(
                    "CLV predictions non-negative",
                    clv_positive,
                    f"All {len(predictions)} predictions >= 0"
                )
        
        # Test 5: Model saved
        model_path = Path('models/clv_predictor.pkl')
        self.test(
            "CLV model saved",
            model_path.exists(),
            f"Expected: {model_path}"
        )
        
        # Test 6: Metrics exist
        metrics_path = Path('outputs/clv_model_metrics.json')
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Test 7: R² reasonable
            if 'r2_score' in metrics:
                self.test(
                    "R² score reasonable",
                    -1 <= metrics['r2_score'] <= 1,
                    f"R²: {metrics['r2_score']:.4f}"
                )
        else:
            self.test("Metrics file exists", False, "File not found")
    
    def validate_pillar_4(self):
        """Validate Pillar 4: Revenue Optimization."""
        self.print_header("PILLAR 4: REVENUE OPTIMIZATION")
        
        # Test 1: Allocation exists
        allocation_path = Path('outputs/optimal_campaign_allocation.parquet')
        self.test(
            "Campaign allocation exists",
            allocation_path.exists(),
            f"Expected: {allocation_path}"
        )
        
        if allocation_path.exists():
            allocation = pd.read_parquet(allocation_path)
            
            # Test 2: Has customers
            self.test(
                "Customers allocated to campaigns",
                len(allocation) > 0,
                f"Allocated {len(allocation):,} customers"
            )
            
            # Test 3: Has campaign column
            self.test(
                "Has campaign_tier column",
                'campaign_tier' in allocation.columns,
                "campaign_tier column present"
            )
        
        # Test 4: Execution plan exists
        plan_path = Path('outputs/campaign_execution_plan.csv')
        self.test(
            "Execution plan exists",
            plan_path.exists(),
            f"Expected: {plan_path}"
        )
        
        # Test 5: Metrics exist
        metrics_path = Path('outputs/optimization_metrics.json')
        self.test(
            "Optimization metrics exist",
            metrics_path.exists(),
            f"Expected: {metrics_path}"
        )
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Test 6: ROI calculated
            if 'portfolio_roi_pct' in metrics:
                self.test(
                    "Portfolio ROI calculated",
                    metrics['portfolio_roi_pct'] is not None,
                    f"ROI: {metrics['portfolio_roi_pct']:.1f}%"
                )
    
    def validate_integration(self):
        """Validate cross-pillar integration."""
        self.print_header("CROSS-PILLAR INTEGRATION")
        
        # Load all outputs
        segments_path = Path('outputs/segment_assignments.parquet')
        churn_path = Path('outputs/churn_predictions.parquet')
        clv_path = Path('outputs/clv_predictions.parquet')
        allocation_path = Path('outputs/optimal_campaign_allocation.parquet')
        
        if all(p.exists() for p in [segments_path, churn_path, clv_path]):
            segments = pd.read_parquet(segments_path)
            churn = pd.read_parquet(churn_path)
            clv = pd.read_parquet(clv_path)
            
            # Test 1: Customer overlap
            seg_customers = set(segments.index)
            churn_customers = set(churn.index)
            clv_customers = set(clv.index)
            
            overlap = len(seg_customers & churn_customers & clv_customers)
            total = len(seg_customers | churn_customers | clv_customers)
            
            self.test(
                "Customer IDs align across pillars",
                overlap / total > 0.5,
                f"{overlap}/{total} customers in all datasets"
            )
            
            # Test 2: Risk-adjusted CLV
            if allocation_path.exists():
                allocation = pd.read_parquet(allocation_path)
                
                self.test(
                    "Allocation uses all predictions",
                    len(allocation) > 0,
                    f"{len(allocation):,} customers in final allocation"
                )
    
    def run_all(self):
        """Run all validation tests."""
        print("\n" + "🔍" * 40)
        print("RETENTION SYSTEM VALIDATION")
        print("🔍" * 40)
        
        self.validate_pillar_0()
        self.validate_pillar_1()
        self.validate_pillar_2()
        self.validate_pillar_3()
        self.validate_pillar_4()
        self.validate_integration()
        
        # Summary
        self.print_header("VALIDATION SUMMARY")
        print(f"\n✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Total: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for deployment!")
            return 0
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Review issues above.")
            return 1


if __name__ == "__main__":
    validator = SystemValidator()
    exit_code = validator.run_all()
    sys.exit(exit_code)
