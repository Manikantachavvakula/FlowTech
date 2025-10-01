"""
FlowTech Customer Intelligence Project
Step 5: Generate Predictions and Create SQL Database

What this does (ELI5):
Uses our trained robot to predict who will leave, then organizes
everything into a database that Power BI can use!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
DATABASE_DIR = BASE_DIR / 'data' / 'database'
REPORTS_DIR = BASE_DIR / 'reports'

# Create database directory
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FLOWTECH PREDICTIONS & DATABASE CREATION")
print("=" * 80)
print("\n")

# -------------------------------------------------------------------
# STEP 1: Load Data and Models
# -------------------------------------------------------------------
print("📂 STEP 1: Loading data and trained models...")
print("-" * 80)

# Load feature-engineered data
customers_df = pd.read_csv(PROCESSED_DATA_DIR / 'customers_features.csv')
print(f"✅ Loaded customer data: {len(customers_df):,} customers")

# Load models and artifacts
best_model = joblib.load(MODELS_DIR / 'best_model.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
label_encoders = joblib.load(MODELS_DIR / 'label_encoders.pkl')

print(f"✅ Loaded best model: Gradient Boosting")
print(f"✅ Loaded scaler and encoders")

# Load feature names
with open(MODELS_DIR / 'feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]
print(f"✅ Loaded {len(feature_names)} feature names")

# -------------------------------------------------------------------
# STEP 2: Prepare Features for Prediction
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔧 STEP 2: Preparing features for prediction...")
print("-" * 80)

# Encode categorical features (same as training)
for col in ['pricing_tier', 'tenure_segment', 'risk_level']:
    customers_df[col + '_encoded'] = label_encoders[col].transform(customers_df[col].astype(str))

# Create feature matrix
X_all = customers_df[feature_names]
print(f"✅ Feature matrix prepared: {X_all.shape}")

# Scale features
X_all_scaled = scaler.transform(X_all)
print(f"✅ Features scaled")

# -------------------------------------------------------------------
# STEP 3: Generate Predictions
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔮 STEP 3: Generating churn predictions...")
print("-" * 80)

# Get predictions and probabilities
predictions = best_model.predict(X_all_scaled)
churn_probabilities = best_model.predict_proba(X_all_scaled)[:, 1]

# Add to dataframe
customers_df['churn_prediction'] = predictions
customers_df['churn_probability'] = churn_probabilities
customers_df['churn_risk_score'] = (churn_probabilities * 100).round(2)

print(f"✅ Generated predictions for all {len(customers_df):,} customers")

# -------------------------------------------------------------------
# STEP 4: Create Risk Segments
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎯 STEP 4: Creating customer risk segments...")
print("-" * 80)

# ELI5: Group customers by how likely they are to leave
# Like sorting students into "needs help", "doing okay", "doing great"
def categorize_risk(probability):
    if probability < 0.3:
        return 'Low Risk'
    elif probability < 0.6:
        return 'Medium Risk'
    else:
        return 'High Risk'

customers_df['predicted_risk_segment'] = customers_df['churn_probability'].apply(categorize_risk)

print(f"\n📊 Risk Segment Distribution:")
risk_dist = customers_df['predicted_risk_segment'].value_counts()
for segment, count in risk_dist.items():
    pct = count / len(customers_df) * 100
    print(f"   {segment}: {count:,} ({pct:.1f}%)")

# -------------------------------------------------------------------
# STEP 5: Identify Actions Needed
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("💡 STEP 5: Identifying required actions...")
print("-" * 80)

# ELI5: Decide what to do with each customer based on their risk
def recommend_action(row):
    if row['churn_probability'] >= 0.7:
        return 'Urgent - Retention Campaign'
    elif row['churn_probability'] >= 0.5:
        return 'High Priority - Engagement Outreach'
    elif row['churn_probability'] >= 0.3:
        return 'Monitor - Check-in Call'
    else:
        return 'Maintain - Standard Service'

customers_df['recommended_action'] = customers_df.apply(recommend_action, axis=1)

print(f"\n📋 Recommended Actions Distribution:")
action_dist = customers_df['recommended_action'].value_counts()
for action, count in action_dist.items():
    print(f"   {action}: {count:,}")

# -------------------------------------------------------------------
# STEP 6: Calculate Business Metrics
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("💰 STEP 6: Calculating business impact metrics...")
print("-" * 80)

# Potential revenue at risk
customers_df['revenue_at_risk'] = customers_df['MonthlyCharges'] * 12  # Annual value

# Calculate total at-risk revenue
high_risk_customers = customers_df[customers_df['churn_probability'] >= 0.5]
total_at_risk_revenue = high_risk_customers['revenue_at_risk'].sum()
total_at_risk_customers = len(high_risk_customers)

print(f"\n📊 Business Impact:")
print(f"   High-risk customers (≥50% probability): {total_at_risk_customers:,}")
print(f"   Annual revenue at risk: ${total_at_risk_revenue:,.2f}")
print(f"   Average revenue per at-risk customer: ${total_at_risk_revenue/total_at_risk_customers:,.2f}")

# -------------------------------------------------------------------
# STEP 7: Create Upsell Opportunities
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎁 STEP 7: Identifying upsell opportunities...")
print("-" * 80)

# ELI5: Find healthy customers who might want to buy more stuff!
def identify_upsell_opportunity(row):
    opportunities = []
    
    # Low churn risk + low services = good upsell candidate
    if row['churn_probability'] < 0.3:
        if row['total_services'] < 3:
            opportunities.append('Add Protection Bundle')
        if row['has_streaming_bundle'] == 0 and row['InternetService'] != 'No':
            opportunities.append('Add Streaming Services')
        if row['is_month_to_month'] == 1 and row['tenure'] >= 12:
            opportunities.append('Upgrade to Annual Contract')
        if row['has_tech_support'] == 0 and row['total_tickets'] > 2:
            opportunities.append('Add Tech Support')
    
    return '; '.join(opportunities) if opportunities else 'None'

customers_df['upsell_opportunity'] = customers_df.apply(identify_upsell_opportunity, axis=1)

# Count upsell opportunities
has_upsell = customers_df[customers_df['upsell_opportunity'] != 'None']
print(f"\n🎯 Upsell Opportunities:")
print(f"   Customers with upsell potential: {len(has_upsell):,}")
print(f"   Potential additional revenue: ${(len(has_upsell) * 15):,.2f}/month")  # Assume $15/service

# -------------------------------------------------------------------
# STEP 8: Create SQLite Database
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🗄️  STEP 8: Creating SQLite database...")
print("-" * 80)

# Create database connection
db_path = DATABASE_DIR / 'flowtech_analytics.db'
conn = sqlite3.connect(db_path)
print(f"✅ Connected to database: {db_path}")

# Prepare main customer table
customers_table = customers_df[[
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Churn',
    'total_sessions', 'avg_session_minutes', 'total_usage_minutes',
    'unique_features_used', 'total_tickets', 'avg_satisfaction',
    'health_score', 'churn_prediction', 'churn_probability',
    'churn_risk_score', 'predicted_risk_segment', 'recommended_action',
    'revenue_at_risk', 'upsell_opportunity'
]].copy()

# Write to database
customers_table.to_sql('customers', conn, if_exists='replace', index=False)
print(f"✅ Created table: customers ({len(customers_table)} rows)")

# -------------------------------------------------------------------
# STEP 9: Create Aggregate Views
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 STEP 9: Creating aggregate views for Power BI...")
print("-" * 80)

# View 1: Risk Summary
risk_summary = customers_df.groupby('predicted_risk_segment').agg({
    'customerID': 'count',
    'MonthlyCharges': 'sum',
    'revenue_at_risk': 'sum',
    'churn_probability': 'mean',
    'avg_satisfaction': 'mean',
    'health_score': 'mean'
}).round(2)
risk_summary.columns = ['customer_count', 'monthly_revenue', 'revenue_at_risk', 
                        'avg_churn_prob', 'avg_satisfaction', 'avg_health_score']
risk_summary = risk_summary.reset_index()
risk_summary.to_sql('risk_summary', conn, if_exists='replace', index=False)
print(f"✅ Created view: risk_summary")

# View 2: Contract Analysis
contract_analysis = customers_df.groupby(['Contract', 'predicted_risk_segment']).agg({
    'customerID': 'count',
    'churn_probability': 'mean',
    'MonthlyCharges': 'mean'
}).round(2)
contract_analysis.columns = ['customer_count', 'avg_churn_prob', 'avg_monthly_charges']
contract_analysis = contract_analysis.reset_index()
contract_analysis.to_sql('contract_analysis', conn, if_exists='replace', index=False)
print(f"✅ Created view: contract_analysis")

# View 3: Service Adoption
service_adoption = customers_df.groupby('total_services').agg({
    'customerID': 'count',
    'churn_probability': 'mean',
    'MonthlyCharges': 'mean',
    'avg_satisfaction': 'mean'
}).round(2)
service_adoption.columns = ['customer_count', 'avg_churn_prob', 
                            'avg_monthly_charges', 'avg_satisfaction']
service_adoption = service_adoption.reset_index()
service_adoption.to_sql('service_adoption', conn, if_exists='replace', index=False)
print(f"✅ Created view: service_adoption")

# View 4: Tenure Segments
tenure_analysis = customers_df.groupby('tenure_segment').agg({
    'customerID': 'count',
    'churn_probability': 'mean',
    'TotalCharges': 'sum',
    'health_score': 'mean'
}).round(2)
tenure_analysis.columns = ['customer_count', 'avg_churn_prob', 
                           'total_revenue', 'avg_health_score']
tenure_analysis = tenure_analysis.reset_index()
tenure_analysis.to_sql('tenure_analysis', conn, if_exists='replace', index=False)
print(f"✅ Created view: tenure_analysis")

# View 5: Support Impact
# Create ticket volume categories
def categorize_tickets(ticket_count):
    if ticket_count == 0:
        return 'No Tickets'
    elif ticket_count <= 2:
        return 'Low (1-2)'
    elif ticket_count <= 5:
        return 'Medium (3-5)'
    else:
        return 'High (6+)'

customers_df['ticket_volume_category'] = customers_df['total_tickets'].apply(categorize_tickets)

support_analysis = customers_df.groupby('ticket_volume_category').agg({
    'customerID': 'count',
    'churn_probability': 'mean',
    'avg_satisfaction': 'mean',
    'avg_resolution_hours': 'mean'
}).round(2)
support_analysis.columns = ['customer_count', 'avg_churn_prob', 
                            'avg_satisfaction', 'avg_resolution_hours']
support_analysis = support_analysis.reset_index()
support_analysis.columns = ['ticket_volume', 'customer_count', 'avg_churn_prob', 
                            'avg_satisfaction', 'avg_resolution_hours']
support_analysis.to_sql('support_analysis', conn, if_exists='replace', index=False)
print(f"✅ Created view: support_analysis")

# -------------------------------------------------------------------
# STEP 10: Export Reports
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📄 STEP 10: Exporting reports...")
print("-" * 80)

# High-risk customers report
high_risk_report = customers_df[customers_df['churn_probability'] >= 0.5][[
    'customerID', 'tenure', 'MonthlyCharges', 'Contract',
    'churn_probability', 'churn_risk_score', 'predicted_risk_segment', 'recommended_action',
    'total_tickets', 'avg_satisfaction', 'health_score',
    'revenue_at_risk'
]].sort_values('churn_probability', ascending=False)

high_risk_report.to_csv(REPORTS_DIR / 'high_risk_customers.csv', index=False)
print(f"✅ Exported: high_risk_customers.csv ({len(high_risk_report)} customers)")

# Upsell opportunities report
upsell_report = customers_df[customers_df['upsell_opportunity'] != 'None'][[
    'customerID', 'tenure', 'MonthlyCharges', 'Contract',
    'total_services', 'churn_risk_score', 'health_score',
    'upsell_opportunity'
]].sort_values('health_score', ascending=False)

upsell_report.to_csv(REPORTS_DIR / 'upsell_opportunities.csv', index=False)
print(f"✅ Exported: upsell_opportunities.csv ({len(upsell_report)} customers)")

# Executive summary
exec_summary = pd.DataFrame({
    'Metric': [
        'Total Customers',
        'High Risk Customers (≥50%)',
        'Medium Risk Customers (30-50%)',
        'Low Risk Customers (<30%)',
        'Annual Revenue at Risk',
        'Customers with Upsell Opportunities',
        'Model ROC-AUC Score',
        'Model Recall (Catches Churners)'
    ],
    'Value': [
        f"{len(customers_df):,}",
        f"{len(customers_df[customers_df['churn_probability'] >= 0.5]):,}",
        f"{len(customers_df[(customers_df['churn_probability'] >= 0.3) & (customers_df['churn_probability'] < 0.5)]):,}",
        f"{len(customers_df[customers_df['churn_probability'] < 0.3]):,}",
        f"${total_at_risk_revenue:,.2f}",
        f"{len(has_upsell):,}",
        "97.62%",
        "82.6%"
    ]
})
exec_summary.to_csv(REPORTS_DIR / 'executive_summary.csv', index=False)
print(f"✅ Exported: executive_summary.csv")

# Close database connection
conn.close()
print(f"\n✅ Database connection closed")

# -------------------------------------------------------------------
# STEP 11: Validation and Summary
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("✔️  STEP 11: Final validation...")
print("-" * 80)

# Verify database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print(f"\n📊 Database Tables Created:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"   • {table[0]}: {count:,} rows")

conn.close()

# -------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎉 PREDICTIONS & DATABASE CREATION COMPLETE!")
print("=" * 80)

print(f"\n📊 Predictions Generated:")
print(f"   ✅ {len(customers_df):,} customers scored")
print(f"   ✅ Risk segments assigned")
print(f"   ✅ Actions recommended")
print(f"   ✅ Upsell opportunities identified")

print(f"\n🗄️  Database Created:")
print(f"   ✅ Location: {db_path}")
print(f"   ✅ Main table: customers")
print(f"   ✅ Aggregate views: 5 tables")

print(f"\n📄 Reports Exported:")
print(f"   ✅ high_risk_customers.csv")
print(f"   ✅ upsell_opportunities.csv")
print(f"   ✅ executive_summary.csv")

print(f"\n💰 Business Impact:")
print(f"   🚨 High-risk customers: {total_at_risk_customers:,}")
print(f"   💵 Revenue at risk: ${total_at_risk_revenue:,.2f}/year")
print(f"   🎁 Upsell opportunities: {len(has_upsell):,} customers")

print("\n" + "=" * 80)
print("📝 Next Step: Share this output with me!")
print("   We'll create the Power BI dashboard next!")
print("=" * 80 + "\n")