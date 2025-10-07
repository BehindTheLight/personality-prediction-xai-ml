# XGBoost Hyperparameter Tuning with Optuna

# Cell 1: Imports and Variable Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
import joblib

n_folds = 3
random_seed = 42
base_dir = '/content/drive/MyDrive/Extro_Intro'
model_name = 'XGBoost_Tuned'
model_dir = f'{base_dir}/models/{model_name}/'
results_dir = f'{model_dir}/results/'
figures_dir = f'{model_dir}/figures/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Cell 2: Data Loading
train = pd.read_csv(f'{base_dir}/data/train_selected.csv')
test = pd.read_csv(f'{base_dir}/data/test_selected.csv')
X = train.drop('Personality', axis=1).values
y = train['Personality'].values

# Cell 3: Optuna Objective Function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 12)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    # Robust GPU/CPU fallback for XGBoost
    try:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_seed,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            n_jobs=-1
        )
        model.fit(np.array([[0,0],[1,1]]), np.array([0,1]))
    except Exception as e:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_seed,
            tree_method='hist',
            predictor='auto',
            n_jobs=-1
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    return scores.mean()

# Cell 4: Run Optuna Study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
study.optimize(objective, n_trials=20)

print('Best trial:')
print(study.best_trial)

# Cell 5: Train and Evaluate Best Model
best_params = study.best_params
# Robust GPU/CPU fallback for XGBoost
try:
    model = xgb.XGBClassifier(**best_params, random_state=random_seed, tree_method='gpu_hist', predictor='gpu_predictor', n_jobs=-1)
    model.fit(np.array([[0,0],[1,1]]), np.array([0,1]))
    print('Using GPU for XGBoost.')
except Exception as e:
    print('GPU not available, falling back to CPU.')
    model = xgb.XGBClassifier(**best_params, random_state=random_seed, tree_method='hist', predictor='auto', n_jobs=-1)
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')
y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]

acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
macro_f1 = f1_score(y, y_pred, average='macro')
f1_per_class = f1_score(y, y_pred, average=None)
f1_class_0 = f1_per_class[0]
f1_class_1 = f1_per_class[1]
roc_auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

cv_acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
cv_macro_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
cv_roc_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

metrics = pd.DataFrame({
    'Model': [model_name],
    'Accuracy': [acc],
    'F1-score': [f1],
    'F1-class-0': [f1_class_0],
    'F1-class-1': [f1_class_1],
    'Macro-F1': [macro_f1],
    'ROC-AUC': [roc_auc],
    'Precision': [precision],
    'Recall': [recall],
    'CV-Accuracy-mean': [cv_acc.mean()],
    'CV-Accuracy-std': [cv_acc.std()],
    'CV-F1-mean': [cv_f1.mean()],
    'CV-F1-std': [cv_f1.std()],
    'CV-Macro-F1-mean': [cv_macro_f1.mean()],
    'CV-Macro-F1-std': [cv_macro_f1.std()],
    'CV-ROC-AUC-mean': [cv_roc_auc.mean()],
    'CV-ROC-AUC-std': [cv_roc_auc.std()]
})
metrics.to_csv(f'{results_dir}/metrics.csv', index=False)

print('XGBoost Tuned Model Cross-Validation Results:')
print(metrics.to_string(index=False))

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(f'{figures_dir}/roc_curve.png')
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y, y_proba)
plt.figure()
plt.plot(rec, prec, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(f'{figures_dir}/pr_curve.png')
plt.close()

# Bar Plot of Metrics
plt.figure()
metrics.set_index('Model').iloc[0].plot(kind='bar')
plt.title('XGBoost Tuned Model Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{figures_dir}/metrics_bar.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{figures_dir}/confusion_matrix.png')
plt.close()

print('\nConfusion Matrix:')
print(cm)

# Cell 6: Save Model and Predictions
model.fit(X, y)
joblib.dump(model, f'{model_dir}/xgboost_tuned_model.pkl')
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(f'{results_dir}/cv_predictions.csv', index=False)
pd.Series(best_params).to_csv(f'{results_dir}/best_params.csv')

print('XGBoost Tuned Model training, evaluation, and saving complete.') 