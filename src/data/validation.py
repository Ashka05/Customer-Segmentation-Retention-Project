"""
Data Validation Module
======================
Comprehensive data quality checks and validation for the UCI Online Retail dataset.
Implements Great Expectations-style validation without the dependency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class ValidationResult:
    """Container for validation results."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    severity: str = "ERROR"  # ERROR, WARNING, INFO


class DataValidator:
    """
    Validates data quality for the retention intelligence system.
    
    Performs comprehensive checks on:
    - Schema (columns, data types)
    - Completeness (missing values)
    - Validity (value ranges, business rules)
    - Consistency (relationships, duplicates)
    - Timeliness (date ranges)
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
        
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (overall_success, list of validation results)
        """
        self.logger.info("Starting comprehensive data validation...")
        self.validation_results = []
        
        # Run all validation checks
        self._validate_schema(df)
        self._validate_completeness(df)
        self._validate_data_types(df)
        self._validate_value_ranges(df)
        self._validate_business_rules(df)
        self._validate_duplicates(df)
        self._validate_dates(df)
        self._validate_customer_consistency(df)
        
        # Determine overall success
        errors = [r for r in self.validation_results if r.severity == "ERROR" and not r.passed]
        overall_success = len(errors) == 0
        
        # Log summary
        self._log_validation_summary()
        
        return overall_success, self.validation_results
    
    def _add_result(self, result: ValidationResult):
        """Add validation result to collection."""
        self.validation_results.append(result)
        
        # Log based on severity
        if not result.passed:
            if result.severity == "ERROR":
                self.logger.error(f"❌ {result.check_name}: {result.message}")
            elif result.severity == "WARNING":
                self.logger.warning(f"⚠️  {result.check_name}: {result.message}")
        else:
            self.logger.info(f"✓ {result.check_name}: {result.message}")
    
    def _validate_schema(self, df: pd.DataFrame):
        """Validate that all required columns exist."""
        required_columns = self.config.get('data_cleaning.required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            self._add_result(ValidationResult(
                check_name="Schema Validation",
                passed=False,
                message=f"Missing required columns: {missing_columns}",
                severity="ERROR"
            ))
        else:
            self._add_result(ValidationResult(
                check_name="Schema Validation",
                passed=True,
                message=f"All {len(required_columns)} required columns present"
            ))
    
    def _validate_completeness(self, df: pd.DataFrame):
        """Validate missing value percentages."""
        max_missing_pct = self.config.get('data_quality.max_missing_pct', 5.0)
        
        missing_stats = {}
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            missing_stats[col] = missing_pct
            
            if missing_pct > max_missing_pct:
                self._add_result(ValidationResult(
                    check_name=f"Completeness - {col}",
                    passed=False,
                    message=f"{missing_pct:.2f}% missing (threshold: {max_missing_pct}%)",
                    severity="WARNING",
                    details={'missing_pct': missing_pct}
                ))
        
        # Overall completeness check
        avg_missing = np.mean(list(missing_stats.values()))
        self._add_result(ValidationResult(
            check_name="Overall Completeness",
            passed=True,
            message=f"Average missing: {avg_missing:.2f}%",
            details=missing_stats
        ))
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate data types can be converted properly."""
        
        # Check numeric columns
        numeric_cols = ['Quantity', 'UnitPrice']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    self._add_result(ValidationResult(
                        check_name=f"Data Type - {col}",
                        passed=True,
                        message=f"{col} is numeric-compatible"
                    ))
                except Exception as e:
                    self._add_result(ValidationResult(
                        check_name=f"Data Type - {col}",
                        passed=False,
                        message=f"{col} cannot be converted to numeric: {e}",
                        severity="ERROR"
                    ))
        
        # Check date column
        if 'InvoiceDate' in df.columns:
            try:
                pd.to_datetime(df['InvoiceDate'], errors='coerce')
                self._add_result(ValidationResult(
                    check_name="Data Type - InvoiceDate",
                    passed=True,
                    message="InvoiceDate is datetime-compatible"
                ))
            except Exception as e:
                self._add_result(ValidationResult(
                    check_name="Data Type - InvoiceDate",
                    passed=False,
                    message=f"InvoiceDate cannot be converted to datetime: {e}",
                    severity="ERROR"
                ))
    
    def _validate_value_ranges(self, df: pd.DataFrame):
        """Validate that values are within expected ranges."""
        
        # UnitPrice validation
        if 'UnitPrice' in df.columns:
            min_price = self.config.get('data_cleaning.filters.min_unit_price', 0.01)
            
            negative_prices = (df['UnitPrice'] < 0).sum()
            zero_prices = (df['UnitPrice'] == 0).sum()
            valid_prices = (df['UnitPrice'] >= min_price).sum()
            
            self._add_result(ValidationResult(
                check_name="Value Range - UnitPrice",
                passed=negative_prices == 0,
                message=f"Negative: {negative_prices:,}, Zero: {zero_prices:,}, Valid: {valid_prices:,}",
                severity="WARNING" if negative_prices > 0 else "INFO",
                details={
                    'negative_count': int(negative_prices),
                    'zero_count': int(zero_prices),
                    'valid_count': int(valid_prices)
                }
            ))
        
        # Quantity validation
        if 'Quantity' in df.columns:
            min_qty = self.config.get('data_cleaning.filters.min_quantity', -100000)
            max_qty = self.config.get('data_cleaning.filters.max_quantity', 100000)
            
            out_of_range = ((df['Quantity'] < min_qty) | (df['Quantity'] > max_qty)).sum()
            zero_qty = (df['Quantity'] == 0).sum()
            
            self._add_result(ValidationResult(
                check_name="Value Range - Quantity",
                passed=True,
                message=f"Out of range: {out_of_range:,}, Zero: {zero_qty:,}",
                severity="INFO",
                details={
                    'out_of_range': int(out_of_range),
                    'zero_count': int(zero_qty)
                }
            ))
    
    def _validate_business_rules(self, df: pd.DataFrame):
        """Validate business logic and rules."""
        
        # Check for cancellation invoices
        if 'InvoiceNo' in df.columns:
            cancellation_prefix = self.config.get('data_cleaning.cancellation_prefix', 'C')
            cancellations = df['InvoiceNo'].astype(str).str.startswith(cancellation_prefix).sum()
            cancellation_pct = (cancellations / len(df)) * 100
            
            self._add_result(ValidationResult(
                check_name="Business Rule - Cancellations",
                passed=True,
                message=f"Found {cancellations:,} cancellations ({cancellation_pct:.2f}%)",
                severity="INFO",
                details={'cancellation_count': int(cancellations)}
            ))
        
        # Check for negative quantities (returns)
        if 'Quantity' in df.columns:
            returns = (df['Quantity'] < 0).sum()
            return_pct = (returns / len(df)) * 100
            
            self._add_result(ValidationResult(
                check_name="Business Rule - Returns",
                passed=True,
                message=f"Found {returns:,} returns ({return_pct:.2f}%)",
                severity="INFO",
                details={'return_count': int(returns)}
            ))
        
        # Calculate revenue validity
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df_temp = df.copy()
            df_temp['Revenue'] = df_temp['Quantity'] * df_temp['UnitPrice']
            
            positive_revenue = (df_temp['Revenue'] > 0).sum()
            negative_revenue = (df_temp['Revenue'] < 0).sum()
            zero_revenue = (df_temp['Revenue'] == 0).sum()
            
            self._add_result(ValidationResult(
                check_name="Business Rule - Revenue Distribution",
                passed=True,
                message=f"Positive: {positive_revenue:,}, Negative: {negative_revenue:,}, Zero: {zero_revenue:,}",
                severity="INFO",
                details={
                    'positive_revenue': int(positive_revenue),
                    'negative_revenue': int(negative_revenue),
                    'zero_revenue': int(zero_revenue)
                }
            ))
    
    def _validate_duplicates(self, df: pd.DataFrame):
        """Check for duplicate records."""
        if not self.config.get('data_quality.check_duplicates', True):
            return
        
        duplicate_subset = self.config.get('data_quality.duplicate_subset', None)
        
        if duplicate_subset:
            available_cols = [col for col in duplicate_subset if col in df.columns]
            if available_cols:
                duplicates = df.duplicated(subset=available_cols, keep=False).sum()
                duplicate_pct = (duplicates / len(df)) * 100
                
                self._add_result(ValidationResult(
                    check_name="Duplicate Records",
                    passed=duplicates == 0,
                    message=f"Found {duplicates:,} duplicate rows ({duplicate_pct:.2f}%)",
                    severity="WARNING" if duplicates > 0 else "INFO",
                    details={'duplicate_count': int(duplicates)}
                ))
    
    def _validate_dates(self, df: pd.DataFrame):
        """Validate date ranges and consistency."""
        if 'InvoiceDate' not in df.columns:
            return
        
        # Convert to datetime if not already
        try:
            dates = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            
            min_date = dates.min()
            max_date = dates.max()
            date_range_days = (max_date - min_date).days
            
            # Check if dates are in expected range
            expected_min = pd.to_datetime(self.config.get('data_quality.min_date', '2000-01-01'))
            expected_max = pd.to_datetime(self.config.get('data_quality.max_date', '2030-12-31'))
            
            dates_in_range = ((dates >= expected_min) & (dates <= expected_max)).sum()
            dates_out_of_range = len(dates) - dates_in_range
            
            self._add_result(ValidationResult(
                check_name="Date Range Validation",
                passed=dates_out_of_range == 0,
                message=f"Date range: {min_date.date()} to {max_date.date()} ({date_range_days} days)",
                severity="WARNING" if dates_out_of_range > 0 else "INFO",
                details={
                    'min_date': str(min_date.date()),
                    'max_date': str(max_date.date()),
                    'range_days': date_range_days,
                    'out_of_range': int(dates_out_of_range)
                }
            ))
            
            # Check for null dates
            null_dates = dates.isna().sum()
            if null_dates > 0:
                self._add_result(ValidationResult(
                    check_name="Date Completeness",
                    passed=False,
                    message=f"Found {null_dates:,} null dates",
                    severity="WARNING",
                    details={'null_count': int(null_dates)}
                ))
                
        except Exception as e:
            self._add_result(ValidationResult(
                check_name="Date Validation",
                passed=False,
                message=f"Error validating dates: {e}",
                severity="ERROR"
            ))
    
    def _validate_customer_consistency(self, df: pd.DataFrame):
        """Validate customer-level consistency."""
        if 'CustomerID' not in df.columns:
            return
        
        # Count unique customers
        total_rows = len(df)
        null_customers = df['CustomerID'].isna().sum()
        unique_customers = df['CustomerID'].nunique()
        
        null_pct = (null_customers / total_rows) * 100
        
        self._add_result(ValidationResult(
            check_name="Customer Data",
            passed=True,
            message=f"Unique customers: {unique_customers:,}, Null CustomerID: {null_customers:,} ({null_pct:.2f}%)",
            severity="INFO",
            details={
                'unique_customers': int(unique_customers),
                'null_customers': int(null_customers),
                'null_pct': float(null_pct)
            }
        ))
        
        # Check minimum transactions per customer
        min_transactions = self.config.get('data_cleaning.filters.min_customer_transactions', 2)
        customer_txn_counts = df[df['CustomerID'].notna()].groupby('CustomerID').size()
        
        customers_below_threshold = (customer_txn_counts < min_transactions).sum()
        
        self._add_result(ValidationResult(
            check_name="Customer Transaction Count",
            passed=True,
            message=f"{customers_below_threshold:,} customers with < {min_transactions} transactions",
            severity="INFO",
            details={
                'below_threshold': int(customers_below_threshold),
                'threshold': min_transactions
            }
        ))
    
    def _log_validation_summary(self):
        """Log validation summary."""
        total_checks = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.passed)
        failed = total_checks - passed
        
        errors = sum(1 for r in self.validation_results if r.severity == "ERROR" and not r.passed)
        warnings = sum(1 for r in self.validation_results if r.severity == "WARNING" and not r.passed)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Checks: {total_checks}")
        self.logger.info(f"Passed: {passed} ✓")
        self.logger.info(f"Failed: {failed} ✗")
        self.logger.info(f"  - Errors: {errors}")
        self.logger.info(f"  - Warnings: {warnings}")
        self.logger.info("=" * 80 + "\n")
    
    def get_validation_report(self) -> pd.DataFrame:
        """
        Get validation results as a DataFrame.
        
        Returns:
            DataFrame with validation results
        """
        report_data = []
        for result in self.validation_results:
            report_data.append({
                'Check': result.check_name,
                'Status': 'PASS' if result.passed else 'FAIL',
                'Severity': result.severity,
                'Message': result.message
            })
        
        return pd.DataFrame(report_data)


# Standalone validation function
def validate_raw_data(df: pd.DataFrame, config, logger=None) -> Tuple[bool, pd.DataFrame]:
    """
    Validate raw data and return validation report.
    
    Args:
        df: DataFrame to validate
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Tuple of (validation_passed, validation_report)
    """
    validator = DataValidator(config, logger)
    success, results = validator.validate_all(df)
    report = validator.get_validation_report()
    
    return success, report


if __name__ == "__main__":
    # Test with sample data
    print("Data Validation Module - Test Mode")
    print("=" * 80)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'InvoiceNo': ['536365', '536366', '536367'],
        'StockCode': ['85123A', '71053', '84406B'],
        'Description': ['WHITE HANGING HEART', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS'],
        'Quantity': [6, 6, 8],
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 8:26', '12/1/2010 8:26'],
        'UnitPrice': [2.55, 3.39, 2.75],
        'CustomerID': [17850.0, 17850.0, 17850.0],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom']
    })
    
    print("Sample data created successfully")
    print(f"Shape: {sample_data.shape}")
