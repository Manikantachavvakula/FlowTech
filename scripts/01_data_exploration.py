"""
FlowTech Customer Intelligence Project
Step 1: Initial Data Exploration

What this does (ELI5): 
This script is like a detective looking at clues. It opens each of our 
three data files and tells us what's inside - how many rows, what columns 
exist, if anything is missing, and shows us a few examples.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set up paths (ELI5: Tell Python where to find our files)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
REPORTS_DIR = BASE_DIR / 'reports'

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FLOWTECH DATA EXPLORATION REPORT")
print("=" * 80)
print("\n")

# -------------------------------------------------------------------
# PART 1: Load the datasets
# -------------------------------------------------------------------
print("📂 LOADING DATASETS...")
print("-" * 80)

try:
    customers_df = pd.read_csv(DATA_DIR / 'flowtech_customers.csv')
    usage_df = pd.read_csv(DATA_DIR / 'flowtech_usage.csv')
    tickets_df = pd.read_csv(DATA_DIR / 'flowtech_tickets.csv')
    print("✅ All datasets loaded successfully!\n")
except FileNotFoundError as e:
    print(f"❌ ERROR: Could not find file - {e}")
    print("Please make sure all CSV files are in the data/raw/ folder")
    exit()

# -------------------------------------------------------------------
# PART 2: Dataset Overview
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 DATASET OVERVIEW")
print("=" * 80)

datasets = {
    'Customers': customers_df,
    'Usage': usage_df,
    'Tickets': tickets_df
}

for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(f"  • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  • Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}")

# -------------------------------------------------------------------
# PART 3: Detailed Column Analysis
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔍 DETAILED COLUMN ANALYSIS")
print("=" * 80)

for name, df in datasets.items():
    print(f"\n{'─' * 80}")
    print(f"{name} Dataset - Column Details:")
    print(f"{'─' * 80}")
    
    # Create info summary
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
        'Unique': df.nunique().values
    })
    
    print(info_df.to_string(index=False))

# -------------------------------------------------------------------
# PART 4: Sample Data Preview
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("👀 SAMPLE DATA PREVIEW (First 3 rows)")
print("=" * 80)

for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print(df.head(3).to_string())

# -------------------------------------------------------------------
# PART 5: Key Statistics for Customers Dataset
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📈 CUSTOMERS DATASET - KEY STATISTICS")
print("=" * 80)

print("\n🎯 Churn Distribution:")
churn_counts = customers_df['Churn'].value_counts()
print(churn_counts)
print(f"\nChurn Rate: {(churn_counts.get('Yes', 0) / len(customers_df) * 100):.2f}%")

print("\n💰 Financial Metrics:")
print(f"  • Avg Monthly Charges: ${customers_df['MonthlyCharges'].mean():.2f}")
print(f"  • Avg Total Charges: ${customers_df['TotalCharges'].mean():.2f}")
print(f"  • Avg Tenure: {customers_df['tenure'].mean():.1f} months")

print("\n📊 Categorical Features:")
for col in ['gender', 'Contract', 'InternetService', 'PaymentMethod']:
    if col in customers_df.columns:
        print(f"\n{col}:")
        print(customers_df[col].value_counts().head())

# -------------------------------------------------------------------
# PART 6: Data Quality Issues
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("⚠️  DATA QUALITY CHECKS")
print("=" * 80)

all_issues = []

for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  ⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"     - {col}: {count} ({count/len(df)*100:.2f}%)")
            all_issues.append(f"{name}: {col} has {count} missing values")
    else:
        print(f"  ✅ No missing values")
    
    # Check for duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"  ⚠️  {dupes} duplicate rows found")
        all_issues.append(f"{name}: {dupes} duplicate rows")
    else:
        print(f"  ✅ No duplicate rows")

# -------------------------------------------------------------------
# PART 7: Summary Report
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📋 SUMMARY")
print("=" * 80)

print(f"\n✅ Data Loading: SUCCESS")
print(f"✅ Total Records: {len(customers_df):,} customers, {len(usage_df):,} usage records, {len(tickets_df):,} tickets")
print(f"\n{'⚠️ ' if all_issues else '✅'} Data Quality: {len(all_issues)} issues found")

if all_issues:
    print("\nIssues to address:")
    for issue in all_issues:
        print(f"  • {issue}")

print("\n" + "=" * 80)
print("🎉 EXPLORATION COMPLETE!")
print("=" * 80)