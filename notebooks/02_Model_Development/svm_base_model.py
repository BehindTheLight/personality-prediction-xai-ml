# SVM Base Model - Colab Notebook Cell Structure

# Cell 1: Imports and Variable Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from cuml.svm import SVC as cuSVC
    gpu_available = True
except ImportError:
    from sklearn.svm import SVC
    gpu_available = False
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
import joblib

# General settings (refer to model_layout.txt)
n_folds = 3
random_seed = 42
base_dir = '/content/drive/MyDrive/Extro_Intro'
model_name = 'SVM_Base'
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

# Cell 3: Model Training and Cross-Validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
if gpu_available:
    model = cuSVC(kernel='linear', probability=True, random_state=random_seed)
    # cuML does not support cross_val_predict, so do manual CV
    y_pred = np.zeros_like(y)
    y_proba = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
        y_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
else:
    model = SVC(kernel='linear', probability=True, random_state=random_seed)
    y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]

# Cell 4: Evaluation Metrics
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
macro_f1 = f1_score(y, y_pred, average='macro')
f1_per_class = f1_score(y, y_pred, average=None)
f1_class_0 = f1_per_class[0]
f1_class_1 = f1_per_class[1]
roc_auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# CV mean and std for main metrics
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

print('SVM Base Model Cross-Validation Results:')
print(metrics.to_string(index=False))

# Cell 5: Figure Generation
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
plt.title('SVM Base Model Metrics')
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
if gpu_available:
    # cuML SVC does not have save_model; use pickle
    import pickle
    with open(f'{model_dir}/svm_base_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    # For even faster training, consider using cuML's LinearSVC for linear kernel
else:
    joblib.dump(model, f'{model_dir}/svm_base_model.pkl')
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(f'{results_dir}/cv_predictions.csv', index=False)

print('SVM Base Model training, evaluation, and saving complete.') 