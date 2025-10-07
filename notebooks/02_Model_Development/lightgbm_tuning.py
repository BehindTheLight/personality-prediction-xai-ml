# Cell 1: Imports and Variable Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
import optuna

n_folds = 3
random_seed = 42
n_jobs = -1
model_name = 'LightGBM_Tuned'
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

# Cell 3: Optuna Objective Function
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'random_state': random_seed,
        'n_jobs': n_jobs,
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 64),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    try:
        clf = lgb.LGBMClassifier(**param)
    except TypeError:
        param.pop('device')
        clf = lgb.LGBMClassifier(**param)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    score = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro').mean()
    return score

# Cell 4: Run Optuna Study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
print('Best params:', best_params)

# Cell 5: Train and Evaluate Best Model
params = best_params.copy()
params.update({'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'random_state': random_seed, 'n_jobs': n_jobs})
try:
    params['device'] = 'gpu'
    clf = lgb.LGBMClassifier(**params)
except TypeError:
    params.pop('device', None)
    clf = lgb.LGBMClassifier(**params)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]

acc = accuracy_score(y, y_pred)
f1_macro = f1_score(y, y_pred, average='macro')
f1_class_0 = f1_score(y, y_pred, pos_label=0)
f1_class_1 = f1_score(y, y_pred, pos_label=1)
roc_auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

cv_acc = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring='f1')
cv_f1_macro = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
cv_roc_auc = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')

metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'ROC-AUC', 'Precision', 'Recall'],
    'Value': [acc, f1_class_0, f1_class_1, f1_macro, roc_auc, precision, recall],
    'CV Mean': [cv_acc.mean(), np.nan, np.nan, cv_f1_macro.mean(), cv_roc_auc.mean(), np.nan, np.nan],
    'CV Std': [cv_acc.std(), np.nan, np.nan, cv_f1_macro.std(), cv_roc_auc.std(), np.nan, np.nan]
})
print(metrics.to_string(index=False))

cm = confusion_matrix(y, y_pred)
print('Confusion Matrix:')
print(cm)

metrics.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

# Cell 6: Figure Generation
from sklearn.metrics import roc_curve, precision_recall_curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(figures_dir, 'roc_curve.png'))
plt.close()

prec, rec, _ = precision_recall_curve(y, y_proba)
plt.figure()
plt.plot(rec, prec, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig(os.path.join(figures_dir, 'pr_curve.png'))
plt.close()

plt.figure()
bar_metrics = ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'ROC-AUC']
plt.bar(bar_metrics, metrics.loc[metrics['Metric'].isin(bar_metrics), 'Value'])
plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim(0, 1)
plt.savefig(os.path.join(figures_dir, 'metrics_bar.png'))
plt.close()

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
plt.close()

# Cell 7: Save Model and Predictions
clf.fit(X, y)
joblib.dump(clf, os.path.join(model_dir, 'lightgbm_tuned_model.pkl'))
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(os.path.join(results_dir, 'cv_predictions.csv'), index=False)
pd.Series(best_params).to_csv(os.path.join(results_dir, 'best_params.csv')) 