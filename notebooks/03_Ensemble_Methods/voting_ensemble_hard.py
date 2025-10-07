# Cell 1: Imports and Variable Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from catboost import CatBoostClassifier

n_folds = 3
random_seed = 42
model_name = 'VotingEnsemble_Hard'
base_dir = '/content/drive/MyDrive/Extro_Intro'
model_dir = os.path.join(base_dir, 'models', model_name)
results_dir = os.path.join(model_dir, 'results')
figures_dir = os.path.join(model_dir, 'figures')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Cell 2: Data Loading
train = pd.read_csv(os.path.join(base_dir, 'data', 'train_selected.csv'))
test = pd.read_csv(os.path.join(base_dir, 'data', 'test_selected.csv'))
X = train.drop(['Personality'], axis=1)
y = train['Personality']

# Cell 3: Load Pre-trained Tuned Models
estimators = []

# Load Logistic Regression Tuned
try:
    lr_path = os.path.join(base_dir, 'models', 'LogisticRegression_Tuned', 'logistic_regression_tuned_model.pkl')
    lr_model = joblib.load(lr_path)
    estimators.append(('lr_tuned', lr_model))
    print("✓ Loaded Logistic Regression Tuned")
except Exception as e:
    print(f"✗ Failed to load Logistic Regression Tuned: {e}")

# Load Random Forest Tuned
try:
    rf_path = os.path.join(base_dir, 'models', 'RandomForest_Tuned', 'random_forest_tuned_model.pkl')
    rf_model = joblib.load(rf_path)
    estimators.append(('rf_tuned', rf_model))
    print("✓ Loaded Random Forest Tuned")
except Exception as e:
    print(f"✗ Failed to load Random Forest Tuned: {e}")

# Load XGBoost Tuned
try:
    xgb_path = os.path.join(base_dir, 'models', 'XGBoost_Tuned', 'xgboost_tuned_model.pkl')
    xgb_model = joblib.load(xgb_path)
    estimators.append(('xgb_tuned', xgb_model))
    print("✓ Loaded XGBoost Tuned")
except Exception as e:
    print(f"✗ Failed to load XGBoost Tuned: {e}")

# Load LightGBM Tuned
try:
    lgb_path = os.path.join(base_dir, 'models', 'LightGBM_Tuned', 'lightgbm_tuned_model.pkl')
    lgb_model = joblib.load(lgb_path)
    estimators.append(('lgb_tuned', lgb_model))
    print("✓ Loaded LightGBM Tuned")
except Exception as e:
    print(f"✗ Failed to load LightGBM Tuned: {e}")

# Load CatBoost Tuned
try:
    cat_path = os.path.join(base_dir, 'models', 'CatBoost_Tuned', 'catboost_tuned_model.cbm')
    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)
    estimators.append(('cat_tuned', cat_model))
    print("✓ Loaded CatBoost Tuned")
except Exception as e:
    print(f"✗ Failed to load CatBoost Tuned: {e}")

# Load AdaBoost Tuned
try:
    ada_path = os.path.join(base_dir, 'models', 'AdaBoost_Tuned', 'adaboost_tuned_model.pkl')
    ada_model = joblib.load(ada_path)
    estimators.append(('ada_tuned', ada_model))
    print("✓ Loaded AdaBoost Tuned")
except Exception as e:
    print(f"✗ Failed to load AdaBoost Tuned: {e}")

print(f"\nTotal models loaded: {len(estimators)}")
if len(estimators) == 0:
    raise ValueError("No models could be loaded! Please check the model paths.")

# Cell 4: Create Hard Voting Ensemble
voting_clf = VotingClassifier(estimators=estimators, voting='hard')
print(f"Created Hard Voting Ensemble with {len(estimators)} models using hard voting")

# Cell 5: Model Training and Cross-Validation
print("\nStarting cross-validation...")
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
y_pred = cross_val_predict(voting_clf, X, y, cv=skf, method='predict')
print("Cross-validation completed!")

# Note: Hard voting doesn't provide probabilities, so we can't calculate ROC-AUC
# We'll focus on classification metrics that don't require probabilities

# Cell 6: Evaluation Metrics
acc = accuracy_score(y, y_pred)
f1_macro = f1_score(y, y_pred, average='macro')
f1_class_0 = f1_score(y, y_pred, pos_label=0)
f1_class_1 = f1_score(y, y_pred, pos_label=1)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

cv_acc = cross_val_score(voting_clf, X, y, cv=skf, scoring='accuracy')
cv_f1 = cross_val_score(voting_clf, X, y, cv=skf, scoring='f1')
cv_f1_macro = cross_val_score(voting_clf, X, y, cv=skf, scoring='f1_macro')

metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'Precision', 'Recall'],
    'Value': [acc, f1_class_0, f1_class_1, f1_macro, precision, recall],
    'CV Mean': [cv_acc.mean(), np.nan, np.nan, cv_f1_macro.mean(), np.nan, np.nan],
    'CV Std': [cv_acc.std(), np.nan, np.nan, cv_f1_macro.std(), np.nan, np.nan]
})
print("\nModel Performance Metrics:")
print(metrics.to_string(index=False))

cm = confusion_matrix(y, y_pred)
print('\nConfusion Matrix:')
print(cm)

metrics.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

# Cell 7: Figure Generation
print("\nGenerating visualizations...")

# Metrics Bar Plot
plt.figure(figsize=(10, 6))
bar_metrics = ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1']
bar_values = metrics.loc[metrics['Metric'].isin(bar_metrics), 'Value']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = plt.bar(bar_metrics, bar_values, color=colors, alpha=0.8)
plt.ylabel('Score', fontsize=12)
plt.title('Hard Voting Ensemble Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, bar_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'metrics_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Hard Voting Ensemble', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Additional Analysis: Individual Model Predictions
print("\nAnalyzing individual model predictions...")
individual_predictions = {}
for name, model in estimators:
    try:
        pred = cross_val_predict(model, X, y, cv=skf, method='predict')
        individual_predictions[name] = pred
        print(f"✓ Generated predictions for {name}")
    except Exception as e:
        print(f"✗ Failed to generate predictions for {name}: {e}")

# Agreement Analysis
if len(individual_predictions) > 1:
    agreement_matrix = np.zeros((len(y), len(individual_predictions)))
    for i, (name, pred) in enumerate(individual_predictions.items()):
        agreement_matrix[:, i] = pred
    
    # Calculate agreement percentage
    agreement_counts = np.sum(agreement_matrix, axis=1)
    unanimous_agreement = np.sum(agreement_counts == 0) + np.sum(agreement_counts == len(individual_predictions))
    agreement_percentage = unanimous_agreement / len(y) * 100
    
    print(f"\nModel Agreement Analysis:")
    print(f"Unanimous agreement: {agreement_percentage:.2f}%")
    print(f"Models in ensemble: {len(individual_predictions)}")
    
    # Save agreement analysis
    agreement_data = {
        'unanimous_agreement_percentage': agreement_percentage,
        'total_samples': len(y),
        'unanimous_samples': unanimous_agreement,
        'models_count': len(individual_predictions)
    }
    pd.Series(agreement_data).to_csv(os.path.join(results_dir, 'agreement_analysis.csv'))

print("Visualizations saved!")

# Cell 8: Save Model and Predictions
print("\nSaving model and results...")
voting_clf.fit(X, y)
joblib.dump(voting_clf, os.path.join(model_dir, 'voting_ensemble_hard_model.pkl'))
pd.DataFrame({'y_true': y, 'y_pred': y_pred}).to_csv(os.path.join(results_dir, 'cv_predictions.csv'), index=False)

# Save ensemble configuration
ensemble_config = {
    'estimators': [name for name, _ in estimators],
    'voting_method': 'hard',
    'n_estimators': len(estimators),
    'models_used': [name for name, _ in estimators]
}
pd.Series(ensemble_config).to_csv(os.path.join(results_dir, 'ensemble_config.csv'))

print(f"\n✓ Hard Voting Ensemble completed successfully!")
print(f"✓ Model saved to: {model_dir}")
print(f"✓ Results saved to: {results_dir}")
print(f"✓ Figures saved to: {figures_dir}")
print(f"✓ Ensemble combines {len(estimators)} pre-trained tuned models using hard voting")
print(f"✓ Note: ROC-AUC not calculated as hard voting doesn't provide probabilities") 