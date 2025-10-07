# Logistic Regression Model - Colab Notebook Cell Structure

# Cell 1: Imports and Variable Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
import joblib

# General settings (refer to model_layout.txt)
n_folds = 3
random_seed = 42
n_jobs = -1
base_dir = '/content/drive/MyDrive/Extro_Intro'
model_name = 'LogisticRegression'
model_dir = f'{base_dir}/models/{model_name}/'
results_dir = f'{model_dir}/results/'
figures_dir = f'{model_dir}/figures/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Cell 2: Data Loading
train = pd.read_csv(f'{base_dir}/data/train_selected.csv')
test = pd.read_csv(f'{base_dir}/data/test_selected.csv')
X = train.drop('Personality', axis=1)
y = train['Personality']

# Cell 3: Model Training and Cross-Validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
model = LogisticRegression(max_iter=100, random_state=random_seed, n_jobs=n_jobs)
y_pred = cross_val_predict(model, X, y, cv=skf, method='predict')
y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]

# Cell 4: Evaluation Metrics
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
metrics = pd.DataFrame({
    'Model': [model_name],
    'Accuracy': [acc],
    'F1-score': [f1],
    'ROC-AUC': [roc_auc],
    'Precision': [precision],
    'Recall': [recall]
})
metrics.to_csv(f'{results_dir}/metrics.csv', index=False)

# Print metrics to output cell
print('Logistic Regression Cross-Validation Results:')
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
plt.title('Logistic Regression Metrics')
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

# Print confusion matrix to output cell
print('\nConfusion Matrix:')
print(cm)

# Cell 6: Save Model and Predictions
model.fit(X, y)
joblib.dump(model, f'{model_dir}/logistic_regression_model.pkl')
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(f'{results_dir}/cv_predictions.csv', index=False)

print('Logistic Regression training, evaluation, and saving complete.') 