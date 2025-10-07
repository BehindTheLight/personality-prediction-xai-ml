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

# General settings
n_folds = 3
random_seed = 42
n_jobs = -1
model_name = 'LightGBM_Base'
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

# Cell 3: Model Training and Cross-Validation
try:
    clf = lgb.LGBMClassifier(device='gpu', random_state=random_seed, n_jobs=n_jobs)
except TypeError:
    clf = lgb.LGBMClassifier(random_state=random_seed, n_jobs=n_jobs)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]

# Cell 4: Evaluation Metrics
acc = accuracy_score(y, y_pred)
f1_macro = f1_score(y, y_pred, average='macro')
f1_class_0 = f1_score(y, y_pred, pos_label=0)
f1_class_1 = f1_score(y, y_pred, pos_label=1)
roc_auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# Cross-validation mean/std
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

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
print('Confusion Matrix:')
print(cm)

# Save metrics
metrics.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

# Cell 5: Figure Generation
# ROC Curve
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

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y, y_proba)
plt.figure()
plt.plot(rec, prec, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig(os.path.join(figures_dir, 'pr_curve.png'))
plt.close()

# Bar Plot of Metrics
plt.figure()
bar_metrics = ['Accuracy', 'F1-score (class 0)', 'F1-score (class 1)', 'Macro F1', 'ROC-AUC']
plt.bar(bar_metrics, metrics.loc[metrics['Metric'].isin(bar_metrics), 'Value'])
plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim(0, 1)
plt.savefig(os.path.join(figures_dir, 'metrics_bar.png'))
plt.close()

# Confusion Matrix Plot
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
plt.close()

# Cell 6: Save Model and Predictions
clf.fit(X, y)
joblib.dump(clf, os.path.join(model_dir, 'lightgbm_base_model.pkl'))
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(os.path.join(results_dir, 'cv_predictions.csv'), index=False) 