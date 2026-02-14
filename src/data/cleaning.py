"""
Data Cleaning Pipeline
======================
Production-grade data cleaning for UCI Online Retail dataset.
Handles missing values, outliers, returns, cancellations, and business logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for transactional retail data.
    
    Cleaning stages:
    1. Structural cleaning (nulls, types, schema)
    2. Business logic cleaning (returns, cancellations, invalid transactions)
    3. Outlier detection and removal
    4. Customer-level validation
    5. Feature creation (Revenue, flags)
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize data cleaner.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cleaning_stats = {}
        
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning statistics)
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING DATA CLEANING PIPELINE")
        self.logger.info("=" * 80)
        
        initial_rows = len(df)
        self.cleaning_stats['initial_rows'] = initial_rows
        
        # Stage 1: Structural cleaning
        df = self._structural_cleaning(df)
        
        # Stage 2: Business logic cleaning
        df = self._business_logic_cleaning(df)
        
        # Stage 3: Outlier detection
        df = self._outlier_detection(df)
        
        # Stage 4: Customer-level validation
        df = self._customer_level_validation(df)
        
        # Stage 5: Feature creation
        df = self._create_features(df)
        
        # Final statistics
        final_rows = len(df)
        self.cleaning_stats['final_rows'] = final_rows
        self.cleaning_stats['rows_removed'] = initial_rows - final_rows
        self.cleaning_stats['removal_pct'] = ((initial_rows - final_rows) / initial_rows) * 100
        
        self._log_summary()
        
        return df, self.cleaning_stats
    
    def _structural_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 1: Structural cleaning.
        
        - Remove rows with null CustomerID
        - Remove null StockCode/Description
        - Parse dates
        - Convert types
        """
        self.logger.info("\n--- Stage 1: Structural Cleaning ---")
        
        initial_rows = len(df)
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # 1. Remove null CustomerID (B2B/guest transactions)
        null_customer_count = df['CustomerID'].isna().sum()
        df = df[df['CustomerID'].notna()].copy()
        self.cleaning_stats['null_customer_removed'] = int(null_customer_count)
        self.logger.info(f"✓ Removed {null_customer_count:,} rows with null CustomerID")
        
        # 2. Remove null StockCode
        null_stockcode = df['StockCode'].isna().sum()
        df = df[df['StockCode'].notna()].copy()
        self.cleaning_stats['null_stockcode_removed'] = int(null_stockcode)
        if null_stockcode > 0:
            self.logger.info(f"✓ Removed {null_stockcode:,} rows with null StockCode")
        
        # 3. Remove null Description (optional, but clean data)
        null_description = df['Description'].isna().sum()
        df = df[df['Description'].notna()].copy()
        self.cleaning_stats['null_description_removed'] = int(null_description)
        if null_description > 0:
            self.logger.info(f"✓ Removed {null_description:,} rows with null Description")
        
        # 4. Parse InvoiceDate
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            null_dates = df['InvoiceDate'].isna().sum()
            
            if null_dates > 0:
                df = df[df['InvoiceDate'].notna()].copy()
                self.cleaning_stats['null_date_removed'] = int(null_dates)
                self.logger.info(f"✓ Removed {null_dates:,} rows with unparseable dates")
            else:
                self.logger.info(f"✓ Parsed InvoiceDate successfully")
        except Exception as e:
            self.logger.error(f"✗ Error parsing InvoiceDate: {e}")
            raise
        
        # 5. Convert numeric columns
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
        
        # Remove rows where numeric conversion failed
        numeric_null_count = df[['Quantity', 'UnitPrice']].isna().any(axis=1).sum()
        df = df[~df[['Quantity', 'UnitPrice']].isna().any(axis=1)].copy()
        self.cleaning_stats['numeric_conversion_failed'] = int(numeric_null_count)
        if numeric_null_count > 0:
            self.logger.info(f"✓ Removed {numeric_null_count:,} rows with invalid numeric values")
        
        rows_removed = initial_rows - len(df)
        self.logger.info(f"Stage 1 complete: Removed {rows_removed:,} rows ({(rows_removed/initial_rows)*100:.2f}%)")
        
        return df
    
    def _business_logic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Business logic cleaning.
        
        - Identify returns and cancellations
        - Remove invalid transactions
        - Flag special transaction types
        """
        self.logger.info("\n--- Stage 2: Business Logic Cleaning ---")
        
        initial_rows = len(df)
        
        # Make a copy
        df = df.copy()
        
        # 1. Identify cancellations (InvoiceNo starts with 'C')
        cancellation_prefix = self.config.get('data_cleaning.cancellation_prefix', 'C')
        df['is_cancellation'] = df['InvoiceNo'].astype(str).str.startswith(cancellation_prefix)
        cancellation_count = df['is_cancellation'].sum()
        self.cleaning_stats['cancellations_flagged'] = int(cancellation_count)
        self.logger.info(f"✓ Flagged {cancellation_count:,} cancellations")
        
        # 2. Identify returns (negative quantity)
        df['is_return'] = df['Quantity'] < 0
        return_count = df['is_return'].sum()
        self.cleaning_stats['returns_flagged'] = int(return_count)
        self.logger.info(f"✓ Flagged {return_count:,} returns (negative quantity)")
        
        # 3. Remove invalid transactions
        # - Zero or negative UnitPrice
        # - Zero Quantity
        min_price = self.config.get('data_cleaning.filters.min_unit_price', 0.01)
        
        invalid_price = (df['UnitPrice'] < min_price).sum()
        invalid_qty = (df['Quantity'] == 0).sum()
        
        df = df[df['UnitPrice'] >= min_price].copy()
        df = df[df['Quantity'] != 0].copy()
        
        self.cleaning_stats['invalid_price_removed'] = int(invalid_price)
        self.cleaning_stats['zero_quantity_removed'] = int(invalid_qty)
        
        if invalid_price > 0:
            self.logger.info(f"✓ Removed {invalid_price:,} rows with UnitPrice < {min_price}")
        if invalid_qty > 0:
            self.logger.info(f"✓ Removed {invalid_qty:,} rows with Quantity = 0")
        
        # 4. Optionally remove cancellations (or keep for return analysis)
        # For now, we'll keep them but flag them
        # If you want to remove: df = df[~df['is_cancellation']].copy()
        
        rows_removed = initial_rows - len(df)
        self.logger.info(f"Stage 2 complete: Removed {rows_removed:,} rows ({(rows_removed/initial_rows)*100:.2f}%)")
        
        return df
    
    def _outlier_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Outlier detection and removal.
        
        - Remove extreme price outliers
        - Remove extreme quantity outliers
        """
        self.logger.info("\n--- Stage 3: Outlier Detection ---")
        
        initial_rows = len(df)
        
        # Make a copy
        df = df.copy()
        
        # 1. Price outliers
        if self.config.get('data_cleaning.remove_price_outliers', True):
            max_price_pct = self.config.get('data_cleaning.filters.max_unit_price_percentile', 99)
            price_threshold = df['UnitPrice'].quantile(max_price_pct / 100)
            
            price_outliers = (df['UnitPrice'] > price_threshold).sum()
            df = df[df['UnitPrice'] <= price_threshold].copy()
            
            self.cleaning_stats['price_outliers_removed'] = int(price_outliers)
            if price_outliers > 0:
                self.logger.info(f"✓ Removed {price_outliers:,} price outliers (>{price_threshold:.2f})")
        
        # 2. Quantity outliers (using IQR method for positive quantities)
        if self.config.get('data_cleaning.remove_quantity_outliers', True):
            # Only check positive quantities (returns are expected to be negative)
            positive_qty = df[df['Quantity'] > 0]['Quantity']
            
            Q1 = positive_qty.quantile(0.25)
            Q3 = positive_qty.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # Allow some negative for edge cases
            upper_bound = Q3 + 3 * IQR
            
            qty_outliers = ((df['Quantity'] > 0) & 
                           ((df['Quantity'] < lower_bound) | (df['Quantity'] > upper_bound))).sum()
            
            df = df[~((df['Quantity'] > 0) & 
                     ((df['Quantity'] < lower_bound) | (df['Quantity'] > upper_bound)))].copy()
            
            self.cleaning_stats['quantity_outliers_removed'] = int(qty_outliers)
            if qty_outliers > 0:
                self.logger.info(f"✓ Removed {qty_outliers:,} quantity outliers (IQR method)")
        
        rows_removed = initial_rows - len(df)
        self.logger.info(f"Stage 3 complete: Removed {rows_removed:,} rows ({(rows_removed/initial_rows)*100:.2f}%)")
        
        return df
    
    def _customer_level_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 4: Customer-level validation.
        
        - Remove customers with < min_transactions
        - Calculate customer-level statistics
        """
        self.logger.info("\n--- Stage 4: Customer-Level Validation ---")
        
        initial_rows = len(df)
        initial_customers = df['CustomerID'].nunique()
        
        # Make a copy
        df = df.copy()
        
        # 1. Count transactions per customer
        customer_txn_counts = df.groupby('CustomerID').size()
        min_transactions = self.config.get('data_cleaning.filters.min_customer_transactions', 2)
        
        # Identify valid customers
        valid_customers = customer_txn_counts[customer_txn_counts >= min_transactions].index
        
        # Filter data
        df = df[df['CustomerID'].isin(valid_customers)].copy()
        
        customers_removed = initial_customers - len(valid_customers)
        self.cleaning_stats['customers_removed_min_txn'] = int(customers_removed)
        
        if customers_removed > 0:
            self.logger.info(f"✓ Removed {customers_removed:,} customers with < {min_transactions} transactions")
        
        # 2. Calculate return rate per customer (optional: flag high return customers)
        customer_return_rates = df.groupby('CustomerID').apply(
            lambda x: (x['is_return'].sum() / len(x)) if len(x) > 0 else 0
        )
        
        high_return_threshold = 0.8  # 80% return rate
        high_return_customers = (customer_return_rates > high_return_threshold).sum()
        
        self.cleaning_stats['high_return_customers'] = int(high_return_customers)
        if high_return_customers > 0:
            self.logger.info(f"⚠️  Flagged {high_return_customers:,} customers with >{high_return_threshold*100}% return rate")
        
        rows_removed = initial_rows - len(df)
        final_customers = df['CustomerID'].nunique()
        
        self.logger.info(f"Stage 4 complete: Removed {rows_removed:,} rows ({(rows_removed/initial_rows)*100:.2f}%)")
        self.logger.info(f"Final customer count: {final_customers:,} (removed {customers_removed:,})")
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 5: Create derived features.
        
        - Revenue = Quantity * UnitPrice
        - Extract date components
        - Create derived flags
        """
        self.logger.info("\n--- Stage 5: Feature Creation ---")
        
        # Make a copy
        df = df.copy()
        
        # 1. Revenue calculation
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        self.logger.info("✓ Created Revenue feature")
        
        # 2. Date components
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        self.logger.info("✓ Extracted date components")
        
        # 3. Absolute values for analysis
        df['AbsQuantity'] = df['Quantity'].abs()
        df['AbsRevenue'] = df['Revenue'].abs()
        
        # 4. Transaction type classification
        df['transaction_type'] = 'purchase'
        df.loc[df['is_return'], 'transaction_type'] = 'return'
        df.loc[df['is_cancellation'], 'transaction_type'] = 'cancellation'
        self.logger.info("✓ Classified transaction types")
        
        self.logger.info(f"Final feature count: {len(df.columns)} columns")
        
        return df
    
    def _log_summary(self):
        """Log cleaning summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DATA CLEANING SUMMARY")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Initial rows: {self.cleaning_stats['initial_rows']:,}")
        self.logger.info(f"Final rows: {self.cleaning_stats['final_rows']:,}")
        self.logger.info(f"Rows removed: {self.cleaning_stats['rows_removed']:,} "
                        f"({self.cleaning_stats['removal_pct']:.2f}%)")
        self.logger.info("\nBreakdown:")
        self.logger.info(f"  - Null CustomerID: {self.cleaning_stats.get('null_customer_removed', 0):,}")
        self.logger.info(f"  - Invalid prices: {self.cleaning_stats.get('invalid_price_removed', 0):,}")
        self.logger.info(f"  - Zero quantity: {self.cleaning_stats.get('zero_quantity_removed', 0):,}")
        self.logger.info(f"  - Price outliers: {self.cleaning_stats.get('price_outliers_removed', 0):,}")
        self.logger.info(f"  - Quantity outliers: {self.cleaning_stats.get('quantity_outliers_removed', 0):,}")
        self.logger.info(f"  - Min transaction filter: {self.cleaning_stats.get('customers_removed_min_txn', 0):,} customers")
        
        self.logger.info("\nData characteristics:")
        self.logger.info(f"  - Cancellations flagged: {self.cleaning_stats.get('cancellations_flagged', 0):,}")
        self.logger.info(f"  - Returns flagged: {self.cleaning_stats.get('returns_flagged', 0):,}")
        
        self.logger.info("=" * 80 + "\n")


def load_and_clean_data(
    file_path: str,
    config,
    logger: Optional[logging.Logger] = None,
    save_output: bool = True,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load raw data, clean it, and optionally save.
    
    Args:
        file_path: Path to raw data CSV
        config: Configuration object
        logger: Logger instance
        save_output: Whether to save cleaned data
        output_path: Where to save cleaned data
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load raw data
    logger.info(f"Loading raw data from: {file_path}")
    
    try:
        df_raw = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 encoding failed, trying ISO-8859-1")
        df_raw = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    logger.info(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
    
    # Clean data
    cleaner = DataCleaner(config, logger)
    df_clean, stats = cleaner.clean(df_raw)
    
    # Save if requested
    if save_output:
        if output_path is None:
            output_path = config.get('paths.processed_data') + 'cleaned_data.parquet'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved cleaned data to: {output_path}")
    
    return df_clean, stats


if __name__ == "__main__":
    print("Data Cleaning Module - Test Mode")
    print("=" * 80)
    print("This module provides production-grade data cleaning for UCI Online Retail dataset")
    print("Use load_and_clean_data() function to process your data")
