# FlowTech Customer Intelligence: Churn Prediction & Analytics

**An end-to-end machine learning project for predicting customer churn and identifying business opportunities in a SaaS company.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

FlowTech is a fictional SaaS company facing customer churn challenges. This project builds a complete data science pipeline to:

- **Predict customer churn** with 97.6% ROC-AUC accuracy
- **Identify at-risk customers** for targeted retention campaigns
- **Discover upsell opportunities** among healthy customer segments
- **Visualize insights** through interactive Power BI dashboards

### Key Results

- **1,653 high-risk customers** identified (≥50% churn probability)
- **$5.47M annual revenue** at risk
- **97.6% model accuracy** (Gradient Boosting)
- **82.6% recall rate** - catches 8 out of 10 potential churners
- **2,800+ upsell opportunities** identified

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, imbalanced-learn, XGBoost |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn, Power BI |
| **Deployment** | Joblib (model serialization) |

---

## Project Structure

```
flowtech_churn_project/
│
├── data/
│   ├── raw/                          # Original CSV files
│   ├── processed/                    # Cleaned datasets
│   │   ├── customers_cleaned.csv
│   │   ├── customers_features.csv
│   │   ├── usage_cleaned.csv
│   │   └── tickets_cleaned.csv
│   └── database/
│       └── flowtech_analytics.db     # SQLite database
│
├── scripts/
│   ├── 01_data_exploration.py        # Initial EDA
│   ├── 02_data_cleaning.py           # Data cleaning
│   ├── 03_feature_engineering.py     # Feature creation
│   ├── 04_modeling.py                # Model training
│   └── 05_predictions.py             # Generate predictions
│
├── models/                           # Trained ML models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── reports/
│   ├── figures/                      # Visualizations
│   │   ├── model_comparison.png
│   │   ├── roc_curves.png
│   │   └── feature_importance.png
│   ├── high_risk_customers.csv
│   ├── upsell_opportunities.csv
│   ├── executive_summary.csv
│   └── feature_importance.csv
│
├── powerbi/                          # Power BI dashboards
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flowtech-churn-prediction.git
cd flowtech-churn-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Data Exploration

```bash
python scripts/01_data_exploration.py
```

Analyzes the raw datasets and identifies data quality issues.

### Step 2: Data Cleaning

```bash
python scripts/02_data_cleaning.py
```

Handles missing values, converts data types, and creates binary encodings.

### Step 3: Feature Engineering

```bash
python scripts/03_feature_engineering.py
```

Creates 31 new predictive features including:
- Financial metrics (charges per tenure, pricing tiers)
- Service combinations (protection bundle, streaming bundle)
- Risk indicators (contract type, payment method)
- Engagement scores (usage patterns, feature adoption)
- Support metrics (ticket volume, satisfaction scores)

### Step 4: Model Training

```bash
python scripts/04_modeling.py
```

Trains and evaluates three models:
- Logistic Regression
- Random Forest
- **Gradient Boosting (Best: 97.6% ROC-AUC)**

### Step 5: Generate Predictions

```bash
python scripts/05_predictions.py
```

Creates predictions for all customers and builds the SQLite database.

---

## Machine Learning Models

### Model Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| **Gradient Boosting** | **0.9762** | **92.5%** | **88.5%** | **82.6%** | **85.5%** |
| Random Forest | 0.9753 | 92.3% | 84.1% | 87.7% | 85.9% |
| Logistic Regression | 0.9680 | 90.1% | 77.0% | 89.6% | 82.8% |

### Top Predictive Features

1. **total_tickets** (34.6%) - Support ticket volume
2. **tickets_per_month** (9.5%) - Support frequency
3. **high_ticket_slow_resolution** (6.2%) - Support quality
4. **is_month_to_month** (5.9%) - Contract flexibility
5. **support_experience_score** (4.8%) - Overall support satisfaction

---

## Database Schema

### Main Tables

**`customers`** - Complete customer profiles with predictions
- Demographics, services, contract details
- Usage metrics, support history
- Churn predictions and risk scores
- Recommended actions

**`risk_summary`** - Aggregated risk metrics by segment

**`contract_analysis`** - Churn analysis by contract type

**`service_adoption`** - Impact of service bundles on churn

**`tenure_analysis`** - Customer lifecycle patterns

**`support_analysis`** - Support quality impact on retention

---

## Power BI Dashboard

The project includes three interactive dashboards:

### 1. Risk Analysis Dashboard
- Customer risk distribution
- Revenue at risk metrics
- Top at-risk customers table
- Risk trends by services

### 2. Engagement Analytics
- Session and usage metrics
- Feature adoption rates
- Customer activity patterns
- Engagement gauge

### 3. Executive Overview
- KPI cards (total customers, churned, at-risk, revenue)
- Contract analysis
- Customer segmentation
- Support impact visualization

### Connecting Power BI to Database

1. Open Power BI Desktop
2. Get Data → SQLite Database
3. Navigate to `data/database/flowtech_analytics.db`
4. Select tables to import
5. Create visualizations

---

## Key Insights

### Customer Churn Drivers

1. **Support Issues Dominate** - Customers with 6+ tickets have 59% churn risk
2. **Contract Type Matters** - Month-to-month contracts have 3x higher churn
3. **Low Engagement = High Risk** - Customers using <3 features churn more
4. **Payment Method Impact** - Electronic check users churn 40% more
5. **Early Tenure Critical** - First 6 months have highest churn risk

### Business Recommendations

**Immediate Actions (High-Risk Customers):**
- Deploy urgent retention campaigns for 1,653 high-risk customers
- Prioritize customers with high ticket volume + low satisfaction
- Offer contract upgrades with incentives

**Strategic Initiatives:**
- Improve first 90-day onboarding experience
- Accelerate support ticket resolution times
- Promote automatic payment methods
- Increase feature adoption through training

**Revenue Opportunities:**
- 2,800+ low-risk customers identified for upsell
- Focus on protection bundles and streaming services
- Target customers with 12+ months tenure for annual contracts

---

## Results & Business Impact

- **Revenue Protection:** Identified $5.47M at risk, enabling proactive retention
- **Precision Targeting:** 88.5% precision reduces wasted retention spend
- **Early Warning System:** Catches 82.6% of potential churners before they leave
- **Upsell Pipeline:** Generated 2,800+ qualified upsell opportunities

---

## Future Enhancements

- [ ] Deploy model as REST API with Flask/FastAPI
- [ ] Implement real-time prediction pipeline
- [ ] Add customer segmentation clustering (K-means, DBSCAN)
- [ ] Build recommendation engine for personalized retention offers
- [ ] Integrate with CRM systems (Salesforce, HubSpot)
- [ ] A/B testing framework for retention campaigns
- [ ] Time-series forecasting for revenue predictions

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Your Name**
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/manikanta-chavvakula-43b308189)
- Portfolio: [@yourusername](https://manikanta-portfolio-six.vercel.app/)

---

## Acknowledgments

- Scikit-learn documentation and community
- Power BI community for visualization inspiration
- Kaggle for dataset inspiration and learning resources

---

**⭐ If you found this project helpful, please consider giving it a star!**