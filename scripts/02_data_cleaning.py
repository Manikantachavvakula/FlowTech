"""
FlowTech Customer Intelligence Project
Step 2: Data Cleaning & Preparation

What this does (ELI5):
This script fixes the problems we found and gets our data ready for analysis.
It's like cleaning and organizing your LEGO pieces before building something cool!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# Create processed directory
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FLOWTECH DATA CLEANING & PREPARATION")
print("=" * 80)
print("\n")

# -------------------------------------------------------------------
# STEP 1: Load Raw Data
# -------------------------------------------------------------------
print("📂 STEP 1: Loading raw datasets...")
print("-" * 80)

customers_df = pd.read_csv(RAW_DATA_DIR / 'flowtech_customers.csv')
usage_df = pd.read_csv(RAW_DATA_DIR / 'flowtech_usage.csv')
tickets_df = pd.read_csv(RAW_DATA_DIR / 'flowtech_tickets.csv')

print(f"✅ Loaded: {len(customers_df):,} customers")
print(f"✅ Loaded: {len(usage_df):,} usage records")
print(f"✅ Loaded: {len(tickets_df):,} tickets")

# -------------------------------------------------------------------
# STEP 2: Handle Missing Values in Customers Dataset
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔧 STEP 2: Fixing missing values...")
print("-" * 80)

print(f"\n📊 Before fixing:")
print(f"   Missing avg_resolution_hours: {customers_df['avg_resolution_hours'].isna().sum()}")

# ELI5: If a customer never called support, their average wait time is 0
# (not missing - they just never waited!)
customers_df['avg_resolution_hours'] = customers_df['avg_resolution_hours'].fillna(0)

print(f"\n✅ After fixing:")
print(f"   Missing avg_resolution_hours: {customers_df['avg_resolution_hours'].isna().sum()}")
print(f"   Filled with 0 for customers with no tickets")

# -------------------------------------------------------------------
# STEP 3: Convert Data Types
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔄 STEP 3: Converting data types...")
print("-" * 80)

# Convert dates to datetime (ELI5: Teaching the computer what a "date" really means)
print("\n📅 Converting date columns...")
usage_df['date'] = pd.to_datetime(usage_df['date'])
tickets_df['date'] = pd.to_datetime(tickets_df['date'])
print(f"   ✅ Usage dates: {usage_df['date'].min()} to {usage_df['date'].max()}")
print(f"   ✅ Ticket dates: {tickets_df['date'].min()} to {tickets_df['date'].max()}")

# Convert TotalCharges to numeric (some might be stored as strings)
print("\n💰 Converting TotalCharges to numeric...")
customers_df['TotalCharges'] = pd.to_numeric(customers_df['TotalCharges'], errors='coerce')
print(f"   ✅ Converted. Any issues: {customers_df['TotalCharges'].isna().sum()} missing values")

# If any TotalCharges are missing after conversion, fill with MonthlyCharges * tenure
if customers_df['TotalCharges'].isna().sum() > 0:
    customers_df['TotalCharges'] = customers_df['TotalCharges'].fillna(
        customers_df['MonthlyCharges'] * customers_df['tenure']
    )
    print(f"   ✅ Filled missing TotalCharges with MonthlyCharges × tenure")

# -------------------------------------------------------------------
# STEP 4: Create Binary Encoding for Target Variable
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎯 STEP 4: Encoding target variable (Churn)...")
print("-" * 80)

# ELI5: Converting "Yes"/"No" to 1/0 so computer can do math
customers_df['Churn_Binary'] = (customers_df['Churn'] == 'Yes').astype(int)
print(f"\n   Original 'Churn' column: {customers_df['Churn'].unique()}")
print(f"   New 'Churn_Binary' column: {customers_df['Churn_Binary'].unique()}")
print(f"   Churn=Yes → 1, Churn=No → 0")

# -------------------------------------------------------------------
# STEP 5: Clean and Standardize Text Fields
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🧹 STEP 5: Cleaning text fields...")
print("-" * 80)

# Standardize categorical values (ELI5: Making sure "yes", "Yes", "YES" are all the same)
print("\n📝 Standardizing categorical fields...")

categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']

for col in categorical_columns:
    customers_df[col] = customers_df[col].str.strip()  # Remove extra spaces
    print(f"   ✅ Cleaned: {col}")

# -------------------------------------------------------------------
# STEP 6: Create Additional Useful Columns
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("➕ STEP 6: Creating additional useful columns...")
print("-" * 80)

# Has Phone Service (binary)
customers_df['HasPhoneService'] = (customers_df['PhoneService'] == 'Yes').astype(int)

# Has Internet Service (binary)
customers_df['HasInternetService'] = (customers_df['InternetService'] != 'No').astype(int)

# Is Senior (already binary, but making sure)
customers_df['IsSenior'] = customers_df['SeniorCitizen'].astype(int)

# Has Partner (binary)
customers_df['HasPartner'] = (customers_df['Partner'] == 'Yes').astype(int)

# Has Dependents (binary)
customers_df['HasDependents'] = (customers_df['Dependents'] == 'Yes').astype(int)

# Paperless Billing (binary)
customers_df['UsesPaperlessBilling'] = (customers_df['PaperlessBilling'] == 'Yes').astype(int)

# Has Tickets (binary)
customers_df['HasTickets'] = (customers_df['total_tickets'] > 0).astype(int)

print("   ✅ Created 7 new binary indicator columns")

# -------------------------------------------------------------------
# STEP 7: Data Quality Validation
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("✔️  STEP 7: Final data quality validation...")
print("-" * 80)

# Check for any remaining missing values
print("\n🔍 Checking for missing values...")
missing_summary = customers_df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]

if len(missing_summary) == 0:
    print("   ✅ No missing values in customers dataset!")
else:
    print("   ⚠️  Still have missing values:")
    for col, count in missing_summary.items():
        print(f"      - {col}: {count}")

# Check for duplicates
print("\n🔍 Checking for duplicates...")
dupes = customers_df['customerID'].duplicated().sum()
if dupes == 0:
    print("   ✅ No duplicate customer IDs!")
else:
    print(f"   ⚠️  Found {dupes} duplicate customer IDs")

# Check data ranges
print("\n🔍 Checking data ranges...")
print(f"   Tenure: {customers_df['tenure'].min()} to {customers_df['tenure'].max()} months")
print(f"   Monthly Charges: ${customers_df['MonthlyCharges'].min():.2f} to ${customers_df['MonthlyCharges'].max():.2f}")
print(f"   Health Score: {customers_df['health_score'].min():.1f} to {customers_df['health_score'].max():.1f}")

# -------------------------------------------------------------------
# STEP 8: Save Cleaned Datasets
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("💾 STEP 8: Saving cleaned datasets...")
print("-" * 80)

# Save cleaned data
customers_df.to_csv(PROCESSED_DATA_DIR / 'customers_cleaned.csv', index=False)
usage_df.to_csv(PROCESSED_DATA_DIR / 'usage_cleaned.csv', index=False)
tickets_df.to_csv(PROCESSED_DATA_DIR / 'tickets_cleaned.csv', index=False)

print(f"\n✅ Saved: customers_cleaned.csv ({len(customers_df):,} rows, {len(customers_df.columns)} columns)")
print(f"✅ Saved: usage_cleaned.csv ({len(usage_df):,} rows, {len(usage_df.columns)} columns)")
print(f"✅ Saved: tickets_cleaned.csv ({len(tickets_df):,} rows, {len(tickets_df.columns)} columns)")

# -------------------------------------------------------------------
# STEP 9: Summary Report
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 CLEANING SUMMARY")
print("=" * 80)

print(f"\n✅ Data Quality Issues Fixed: 1")
print(f"   • Filled {1267} missing avg_resolution_hours values with 0")

print(f"\n✅ New Columns Created: 7")
print(f"   • HasPhoneService, HasInternetService, IsSenior")
print(f"   • HasPartner, HasDependents, UsesPaperlessBilling")
print(f"   • HasTickets")

print(f"\n✅ Data Types Converted: 3")
print(f"   • Usage dates → datetime")
print(f"   • Ticket dates → datetime")
print(f"   • TotalCharges → numeric")

print(f"\n✅ Target Variable Encoded:")
print(f"   • Churn (Yes/No) → Churn_Binary (1/0)")

print("\n" + "=" * 80)
print("🎉 DATA CLEANING COMPLETE!")
print("=" * 80)
print(f"\n📁 Cleaned files saved to: {PROCESSED_DATA_DIR}")
print("\n📝 Next Step: Share this output with me, and I'll give you Step 3!")
print("   (We'll start building awesome features for our model!)\n")