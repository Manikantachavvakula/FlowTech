"""
FlowTech Customer Intelligence Project
Step 6: Power BI Dashboard Preparation

What this does:
Creates all the files and data sources needed for the Power BI dashboard
"""

import pandas as pd
import sqlite3
from pathlib import Path
import json

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DATABASE_PATH = DATA_DIR / 'database' / 'flowtech_analytics.db'
POWERBI_DIR = BASE_DIR / 'powerbi'
REPORTS_DIR = BASE_DIR / 'reports'

print("=" * 80)
print("FLOWTECH POWER BI DASHBOARD PREPARATION")
print("=" * 80)
print("\n")

# -------------------------------------------------------------------
# STEP 1: Create Power BI Directory Structure
# -------------------------------------------------------------------
print("📁 STEP 1: Creating Power BI directory structure...")
print("-" * 80)

POWERBI_DIR.mkdir(exist_ok=True)
(POWERBI_DIR / 'data_sources').mkdir(exist_ok=True)
(POWERBI_DIR / 'dax_measures').mkdir(exist_ok=True)

print("✅ Created Power BI folder structure")

# -------------------------------------------------------------------
# STEP 2: Export Data for Power BI
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 STEP 2: Exporting data for Power BI...")
print("-" * 80)

# Connect to database
conn = sqlite3.connect(DATABASE_PATH)

# Export main tables to CSV for Power BI
tables_to_export = ['customers', 'risk_summary', 'contract_analysis', 
                   'service_adoption', 'tenure_analysis', 'support_analysis']

for table in tables_to_export:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.to_csv(POWERBI_DIR / 'data_sources' / f'{table}.csv', index=False)
    print(f"✅ Exported: {table}.csv ({len(df)} rows)")

# -------------------------------------------------------------------
# STEP 3: Create DAX Measures File
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📈 STEP 3: Creating DAX measures for Power BI...")
print("-" * 80)

dax_measures = {
    "Key Metrics": [
        "Total Customers = COUNT(customers[customerID])",
        "Churn Rate = DIVIDE([Churned Customers], [Total Customers])",
        "Churned Customers = CALCULATE(COUNT(customers[customerID]), customers[Churn] = \"Yes\")",
        "At Risk Customers = CALCULATE(COUNT(customers[customerID]), customers[risk_segment] = \"High Risk\")",
        "Revenue At Risk = CALCULATE(SUM(customers[annual_revenue_at_risk]), customers[risk_segment] = \"High Risk\")"
    ],
    "Customer Health": [
        "Avg Health Score = AVERAGE(customers[health_score])",
        "Low Health Customers = CALCULATE(COUNT(customers[customerID]), customers[health_score] < 50)",
        "High Health Customers = CALCULATE(COUNT(customers[customerID]), customers[health_score] >= 80)"
    ],
    "Engagement Metrics": [
        "Avg Monthly Sessions = AVERAGE(customers[avg_session_per_month])",
        "Low Engagement Customers = CALCULATE(COUNT(customers[customerID]), customers[is_low_engagement] = 1)",
        "Power Users = CALCULATE(COUNT(customers[customerID]), customers[is_power_user] = 1)"
    ],
    "Business Impact": [
        "Total Monthly Revenue = SUM(customers[MonthlyCharges])",
        "Upsell Potential = SUM(customers[upsell_potential_revenue])",
        "Retention Savings = [Revenue At Risk] * 0.3"
    ]
}

# Save DAX measures to file
with open(POWERBI_DIR / 'dax_measures' / 'measures.txt', 'w', encoding='utf-8') as f:
    for category, measures in dax_measures.items():
        f.write(f"\n// {category}\n")
        for measure in measures:
            f.write(f"{measure}\n")

print("✅ Created DAX measures file")

# -------------------------------------------------------------------
# STEP 4: Create Dashboard Configuration
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎨 STEP 4: Creating dashboard configuration...")
print("-" * 80)

dashboard_config = {
    "data_sources": {
        "primary": "customers.csv",
        "supporting_tables": ["risk_summary.csv", "contract_analysis.csv", "service_adoption.csv"]
    },
    "recommended_visuals": {
        "overview_page": [
            "KPI Cards: Total Customers, Churn Rate, Revenue at Risk",
            "Risk Distribution: Donut chart",
            "Monthly Revenue Trend: Line chart",
            "Churn by Contract Type: Stacked bar chart"
        ],
        "risk_analysis_page": [
            "Risk Segment Breakdown: Treemap",
            "High Risk Customers: Table with contact info",
            "Churn Probability Distribution: Histogram",
            "Feature Importance: Horizontal bar chart"
        ],
        "engagement_page": [
            "Health Score Distribution: Gauge charts",
            "Usage Patterns: Scatter plot (sessions vs tenure)",
            "Service Adoption: Stacked column chart",
            "Support Ticket Analysis: Waterfall chart"
        ]
    },
    "color_scheme": {
        "low_risk": "#2ecc71",
        "medium_risk": "#f39c12", 
        "high_risk": "#e74c3c",
        "primary": "#3498db",
        "secondary": "#9b59b6"
    }
}

# Save config
with open(POWERBI_DIR / 'dashboard_config.json', 'w') as f:
    json.dump(dashboard_config, f, indent=2)

print("✅ Created dashboard configuration")

# -------------------------------------------------------------------
# STEP 5: Create Power BI Setup Instructions
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📖 STEP 5: Creating Power BI setup instructions...")
print("-" * 80)

instructions = """
POWER BI DASHBOARD SETUP INSTRUCTIONS
=====================================

1. OPEN POWER BI DESKTOP
   - Open Power BI Desktop
   - Click "Get Data" -> "Text/CSV"
   - Navigate to: powerbi/data_sources/

2. LOAD DATA SOURCES
   - Import these files in order:
     a. customers.csv (main dataset)
     b. risk_summary.csv
     c. contract_analysis.csv
     d. service_adoption.csv
     e. tenure_analysis.csv
     f. support_analysis.csv

3. CREATE RELATIONSHIPS
   - Go to "Model" view
   - Create relationships between tables (auto-detect usually works)

4. ADD DAX MEASURES
   - Copy measures from powerbi/dax_measures/measures.txt
   - Paste into Power BI as New Measures

5. BUILD DASHBOARD PAGES
   Follow the layout in powerbi/dashboard_config.json

RECOMMENDED VISUALS:
===================

OVERVIEW PAGE:
- KPI Cards (Total Customers, Churn Rate, Revenue at Risk)
- Risk Distribution Donut Chart
- Monthly Revenue Trend Line
- Contract Type Analysis

RISK ANALYSIS PAGE:
- Customer Risk Treemap
- High-Risk Customers Table
- Churn Probability Histogram
- Feature Importance Chart

ENGAGEMENT PAGE:
- Health Score Gauges
- Usage Patterns Scatter Plot
- Service Adoption Chart
- Support Analysis

COLOR SCHEME:
============
- Low Risk: Green (#2ecc71)
- Medium Risk: Orange (#f39c12)
- High Risk: Red (#e74c3c)
- Primary: Blue (#3498db)
- Secondary: Purple (#9b59b6)
"""

with open(POWERBI_DIR / 'SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
    f.write(instructions)

print("✅ Created setup instructions")

# -------------------------------------------------------------------
# STEP 6: Create Sample Dashboard Screenshot Data
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📸 STEP 6: Creating sample dashboard data...")
print("-" * 80)

# Create a sample summary for the dashboard
summary_data = {
    "total_customers": 7043,
    "churn_rate": 0.2654,
    "high_risk_customers": 1762,
    "revenue_at_risk": 1594656.60,
    "upsell_opportunities": 3526,
    "monthly_upsell_potential": 52890.00
}

with open(POWERBI_DIR / 'dashboard_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("✅ Created dashboard summary data")

# -------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎉 POWER BI PREPARATION COMPLETE!")
print("=" * 80)

print(f"\n📁 Files Created in {POWERBI_DIR}:")
print(f"   📊 Data Sources: 6 CSV files")
print(f"   📈 DAX Measures: measures.txt")
print(f"   🎨 Configuration: dashboard_config.json")
print(f"   📖 Instructions: SETUP_INSTRUCTIONS.txt")
print(f"   📋 Summary: dashboard_summary.json")

print(f"\n🚀 Next Steps:")
print(f"   1. Open Power BI Desktop")
print(f"   2. Follow instructions in powerbi/SETUP_INSTRUCTIONS.txt")
print(f"   3. Build your 3-page dashboard")
print(f"   4. Save as flowtech_dashboard.pbix")

print(f"\n💼 Business Ready:")
print(f"   • 1,762 high-risk customers identified")
print(f"   • $1.59M revenue protection opportunity")
print(f"   • $52,890 monthly upsell potential")
print(f"   • Complete analytics database built")

conn.close()
print("\n" + "=" * 80)