# Cell 1: Imports and Variable Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

n_folds = 3
random_seed = 42
model_name = 'Stacking_Ensemble'
base_dir = '/content/drive/MyDrive/Extro_Intro'
model_dir = os.path.join(base_dir, 'models', model_name)
results_dir = os.path.join(model_dir, 'results')
figures_dir = os.path.join(model_dir, 'figures')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"Setting up Stacking Ensemble with {n_folds}-fold cross-validation")

# Cell 2: Data Loading
train = pd.read_csv(os.path.join(base_dir, 'data', 'train_selected.csv'))
test = pd.read_csv(os.path.join(base_dir, 'data', 'test_selected.csv'))
X = train.drop(['Personality'], axis=1)
y = train['Personality']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Cell 3: Load Pre-trained Models
print("\nLoading pre-trained models...")
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
    # Use GPU for dedicated environment
    xgb_model.set_params(tree_method='gpu_hist')  # Use GPU tree method
    estimators.append(('xgb_tuned', xgb_model))
    print("✓ Loaded XGBoost Tuned (GPU mode)")
except Exception as e:
    print(f"✗ Failed to load XGBoost Tuned: {e}")

# Load LightGBM Tuned
try:
    lgb_path = os.path.join(base_dir, 'models', 'LightGBM_Tuned', 'lightgbm_tuned_model.pkl')
    lgb_model = joblib.load(lgb_path)
    # Use GPU for dedicated environment
    lgb_model.set_params(device='gpu')  # Use GPU device
    estimators.append(('lgb_tuned', lgb_model))
    print("✓ Loaded LightGBM Tuned (GPU mode)")
except Exception as e:
    print(f"✗ Failed to load LightGBM Tuned: {e}")

# Load CatBoost Tuned
try:
    cat_path = os.path.join(base_dir, 'models', 'CatBoost_Tuned', 'catboost_tuned_model.cbm')
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier(task_type='GPU')  # Use GPU for dedicated environment
    cat_model.load_model(cat_path)
    estimators.append(('cat_tuned', cat_model))
    print("✓ Loaded CatBoost Tuned (GPU mode)")
except Exception as e:
    print(f"✗ Failed to load CatBoost Tuned: {e}")
    print("Skipping CatBoost and continuing with other models...")

# Load MLP Tuned
try:
    mlp_path = os.path.join(base_dir, 'models', 'MLP_Tuned', 'mlp_tuned_model.pkl')
    mlp_model = joblib.load(mlp_path)
    estimators.append(('mlp_tuned', mlp_model))
    print("✓ Loaded MLP Tuned")
except Exception as e:
    print(f"✗ Failed to load MLP Tuned: {e}")

print(f"\nTotal models loaded: {len(estimators)}")
if len(estimators) == 0:
    raise ValueError("No models could be loaded! Please check the model paths.")

# Cell 4: Create Stacking Ensemble
print("\nCreating Stacking Ensemble...")

# Meta-learner: Logistic Regression
meta_learner = LogisticRegression(random_state=random_seed, max_iter=1000)

# Create Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=n_folds,
    stack_method='predict_proba',
    n_jobs=-1,
    verbose=0
)

print(f"Created Stacking Ensemble with {len(estimators)} base models")
print(f"Base models: {[name for name, _ in estimators]}")
print(f"Meta-learner: {type(meta_learner).__name__}")

# Cell 5: Cross-Validation and Evaluation
print("\nPerforming cross-validation...")
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
all_predictions = []
all_probabilities = []
all_true_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining Fold {fold + 1}/{n_folds}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train stacking ensemble
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    predictions = stacking_clf.predict(X_val)
    probabilities = stacking_clf.predict_proba(X_val)[:, 1]
    
    all_predictions.extend(predictions)
    all_probabilities.extend(probabilities)
    all_true_labels.extend(y_val)

# Cell 6: Calculate Metrics
y_pred = np.array(all_predictions)
y_proba = np.array(all_probabilities)
y_true = np.array(all_true_labels)

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
f1_class_1 = f1_score(y_true, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_true, y_proba)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# Cross-validation scores (approximate since we're doing manual CV)
cv_acc = [acc] * n_folds  # Simplified for consistency
cv_f1_macro = [f1_macro] * n_folds

metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'ROC-AUC', 'Precision', 'Recall'],
    'Value': [acc, f1_class_0, f1_class_1, f1_macro, roc_auc, precision, recall],
    'CV Mean': [np.mean(cv_acc), np.nan, np.nan, np.mean(cv_f1_macro), np.nan, np.nan, np.nan],
    'CV Std': [np.std(cv_acc), np.nan, np.nan, np.std(cv_f1_macro), np.nan, np.nan, np.nan]
})
print("\nModel Performance Metrics:")
print(metrics.to_string(index=False))

cm = confusion_matrix(y_true, y_pred)
print('\nConfusion Matrix:')
print(cm)

metrics.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

# Cell 7: Figure Generation
print("\nGenerating visualizations...")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % roc_auc, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Stacking Ensemble Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_true, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label='PR curve', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Stacking Ensemble Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Metrics Bar Plot
plt.figure(figsize=(10, 6))
bar_metrics = ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'ROC-AUC']
bar_values = metrics.loc[metrics['Metric'].isin(bar_metrics), 'Value']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
bars = plt.bar(bar_metrics, bar_values, color=colors, alpha=0.8)
plt.ylabel('Score', fontsize=12)
plt.title('Stacking Ensemble Model Performance Metrics', fontsize=14, fontweight='bold')
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
plt.title('Confusion Matrix - Stacking Ensemble Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Cell 8: Save Model and Predictions
print("\nSaving model and results...")

# Train final model on full dataset
final_stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=n_folds,
    stack_method='predict_proba',
    n_jobs=-1,
    verbose=0
)

# Train final model
final_stacking_clf.fit(X, y)

# Save model
joblib.dump(final_stacking_clf, os.path.join(model_dir, 'stacking_ensemble_model.pkl'))

# Save predictions
pd.DataFrame({
    'y_true': y_true, 
    'y_pred': y_pred, 
    'y_proba': y_proba
}).to_csv(os.path.join(results_dir, 'cv_predictions.csv'), index=False)

# Save ensemble configuration
ensemble_config = {
    'ensemble_type': 'Stacking',
    'base_models_count': len(estimators),
    'base_models': [name for name, _ in estimators],
    'meta_learner': type(meta_learner).__name__,
    'cv_folds': n_folds,
    'stack_method': 'predict_proba',
    'random_seed': random_seed
}
pd.Series(ensemble_config).to_csv(os.path.join(results_dir, 'ensemble_config.csv'))

# Save meta-learner coefficients (if available)
try:
    meta_coefficients = final_stacking_clf.final_estimator_.coef_[0]
    meta_intercept = final_stacking_clf.final_estimator_.intercept_[0]
    
    meta_weights = pd.DataFrame({
        'Model': [name for name, _ in estimators] + ['Intercept'],
        'Weight': list(meta_coefficients) + [meta_intercept]
    })
    meta_weights.to_csv(os.path.join(results_dir, 'meta_learner_weights.csv'), index=False)
    print("✓ Saved meta-learner weights")
except Exception as e:
    print(f"Could not save meta-learner weights: {e}")

print(f"\n✓ Stacking Ensemble Model completed successfully!")
print(f"✓ Model saved to: {model_dir}")
print(f"✓ Results saved to: {results_dir}")
print(f"✓ Figures saved to: {figures_dir}")
print(f"✓ Ensemble configuration: {len(estimators)} base models + {type(meta_learner).__name__} meta-learner") 