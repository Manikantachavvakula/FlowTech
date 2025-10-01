"""
FlowTech Customer Intelligence Project
Step 4: Machine Learning Model Training

What this does (ELI5):
This trains several "robot brains" to predict which customers will leave.
We'll test different robots and pick the smartest one!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FLOWTECH CHURN PREDICTION - MODEL TRAINING")
print("=" * 80)
print("\n")

# -------------------------------------------------------------------
# STEP 1: Load Feature-Engineered Data
# -------------------------------------------------------------------
print("📂 STEP 1: Loading feature-engineered dataset...")
print("-" * 80)

df = pd.read_csv(PROCESSED_DATA_DIR / 'customers_features.csv')
print(f"✅ Loaded: {len(df):,} customers with {len(df.columns)} features")

# -------------------------------------------------------------------
# STEP 2: Prepare Features for Modeling
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔧 STEP 2: Preparing features for modeling...")
print("-" * 80)

# Features to exclude (IDs, original target, text descriptions)
exclude_features = [
    'customerID', 'Churn', 'Churn_Binary',  # Target and ID
    'pricing_tier', 'tenure_segment', 'risk_level',  # Categorical (will encode separately)
]

# Get numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in exclude_features]

print(f"\n📊 Feature Preparation:")
print(f"   Total features available: {len(df.columns)}")
print(f"   Numeric features selected: {len(numeric_features)}")

# Encode categorical features that we want to keep
print(f"\n🔄 Encoding categorical features...")
label_encoders = {}

categorical_to_encode = ['pricing_tier', 'tenure_segment', 'risk_level']
for col in categorical_to_encode:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    numeric_features.append(col + '_encoded')
    print(f"   ✅ Encoded: {col}")

# Create feature matrix and target
X = df[numeric_features]
y = df['Churn_Binary']

print(f"\n📐 Final Dataset Shape:")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")
print(f"   Churn Rate: {y.mean()*100:.2f}%")

# -------------------------------------------------------------------
# STEP 3: Train-Test Split
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("✂️  STEP 3: Splitting data into train and test sets...")
print("-" * 80)

# ELI5: Split into "learning set" (80%) and "test set" (20%)
# Like studying with 80% of practice problems, then taking a test with the other 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training set: {X_train.shape[0]:,} customers ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set: {X_test.shape[0]:,} customers ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Training churn rate: {y_train.mean()*100:.2f}%")
print(f"   Test churn rate: {y_test.mean()*100:.2f}%")

# -------------------------------------------------------------------
# STEP 4: Feature Scaling
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("⚖️  STEP 4: Scaling features...")
print("-" * 80)

# ELI5: Make all numbers the same size (like converting everything to the same unit)
# So "age: 30" and "salary: 50000" are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled using StandardScaler")
print(f"   Mean: ~0, Std: ~1 for all features")

# -------------------------------------------------------------------
# STEP 5: Handle Class Imbalance with SMOTE
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("⚖️  STEP 5: Handling class imbalance with SMOTE...")
print("-" * 80)

# ELI5: We have way more "stayed" customers than "left" customers
# SMOTE creates fake "left" examples so the robot learns both equally well
print(f"\n📊 Before SMOTE:")
print(f"   Not Churned: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"   Churned: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n📊 After SMOTE:")
print(f"   Not Churned: {(y_train_balanced==0).sum():,} ({(y_train_balanced==0).sum()/len(y_train_balanced)*100:.1f}%)")
print(f"   Churned: {(y_train_balanced==1).sum():,} ({(y_train_balanced==1).sum()/len(y_train_balanced)*100:.1f}%)")
print(f"\n✅ Classes balanced for training")

# -------------------------------------------------------------------
# STEP 6: Train Multiple Models
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🤖 STEP 6: Training multiple machine learning models...")
print("-" * 80)

# Dictionary to store models
models = {}
results = {}

# Model 1: Logistic Regression (ELI5: Simple, fast, easy to explain)
print(f"\n🔵 Training Model 1: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_balanced, y_train_balanced)
models['Logistic Regression'] = lr_model
print(f"   ✅ Trained successfully")

# Model 2: Random Forest (ELI5: Asks many "decision trees" and takes a vote)
print(f"\n🟢 Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
models['Random Forest'] = rf_model
print(f"   ✅ Trained successfully")

# Model 3: Gradient Boosting (ELI5: Learns from mistakes, gets smarter each round)
print(f"\n🟡 Training Model 3: Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_balanced, y_train_balanced)
models['Gradient Boosting'] = gb_model
print(f"   ✅ Trained successfully")

# -------------------------------------------------------------------
# STEP 7: Evaluate Models
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 STEP 7: Evaluating model performance...")
print("-" * 80)

for name, model in models.items():
    print(f"\n{'─' * 80}")
    print(f"📈 {name} Performance:")
    print(f"{'─' * 80}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print metrics
    print(f"\n   Accuracy:  {accuracy:.4f} (% correctly predicted)")
    print(f"   Precision: {precision:.4f} (% of predicted churners actually churned)")
    print(f"   Recall:    {recall:.4f} (% of actual churners we caught)")
    print(f"   F1 Score:  {f1:.4f} (balance of precision & recall)")
    print(f"   ROC-AUC:   {roc_auc:.4f} (overall model quality)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   ┌─────────────────────────────┐")
    print(f"   │ Predicted No  │ Predicted Yes│")
    print(f"   ├─────────────────────────────┤")
    print(f"   │ TN: {cm[0,0]:4d}      │ FP: {cm[0,1]:4d}     │ Actual No")
    print(f"   │ FN: {cm[1,0]:4d}      │ TP: {cm[1,1]:4d}     │ Actual Yes")
    print(f"   └─────────────────────────────┘")

# -------------------------------------------------------------------
# STEP 8: Compare Models
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🏆 STEP 8: Model comparison summary...")
print("-" * 80)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
comparison_df = comparison_df.round(4)

print("\n📊 Model Comparison:")
print(comparison_df.to_string())

# Find best model by ROC-AUC
best_model_name = comparison_df['roc_auc'].idxmax()
best_model = models[best_model_name]
print(f"\n🥇 Best Model: {best_model_name}")
print(f"   ROC-AUC Score: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")

# -------------------------------------------------------------------
# STEP 9: Feature Importance (for tree-based models)
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎯 STEP 9: Feature importance analysis...")
print("-" * 80)

# Get feature importance from Random Forest
if 'Random Forest' in models:
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': models['Random Forest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 15 Most Important Features (Random Forest):")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)

# -------------------------------------------------------------------
# STEP 10: Save Models and Artifacts
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("💾 STEP 10: Saving models and artifacts...")
print("-" * 80)

# Save scaler
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print(f"✅ Saved: scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, MODELS_DIR / 'label_encoders.pkl')
print(f"✅ Saved: label_encoders.pkl")

# Save all models
for name, model in models.items():
    filename = name.lower().replace(' ', '_') + '_model.pkl'
    joblib.dump(model, MODELS_DIR / filename)
    print(f"✅ Saved: {filename}")

# Save best model separately
joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
print(f"✅ Saved: best_model.pkl ({best_model_name})")

# Save feature names
with open(MODELS_DIR / 'feature_names.txt', 'w') as f:
    f.write('\n'.join(numeric_features))
print(f"✅ Saved: feature_names.txt")

# Save model comparison
comparison_df.to_csv(REPORTS_DIR / 'model_comparison.csv')
print(f"✅ Saved: model_comparison.csv")

# -------------------------------------------------------------------
# STEP 11: Create Visualizations
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 STEP 11: Creating visualizations...")
print("-" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    comparison_df[metric].plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#f39c12'])
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: model_comparison.png")
plt.close()

# 2. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#3498db', '#2ecc71', '#f39c12']

for (name, result), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})", 
            linewidth=2, color=color)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: roc_curves.png")
plt.close()

# 3. Feature Importance Plot
if 'Random Forest' in models:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(15)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='#2ecc71')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: feature_importance.png")
    plt.close()

# -------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n✅ Trained and evaluated 3 models:")
print(f"   • Logistic Regression")
print(f"   • Random Forest")
print(f"   • Gradient Boosting")

print(f"\n🥇 Best Model: {best_model_name}")
print(f"   • ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
print(f"   • Recall: {comparison_df.loc[best_model_name, 'recall']:.4f} (catches {comparison_df.loc[best_model_name, 'recall']*100:.1f}% of churners)")
print(f"   • Precision: {comparison_df.loc[best_model_name, 'precision']:.4f}")

print(f"\n💾 Saved artifacts:")
print(f"   • {len(models)} model files")
print(f"   • Scaler and encoders")
print(f"   • Feature importance report")
print(f"   • 3 visualization charts")

print("\n" + "=" * 80)
