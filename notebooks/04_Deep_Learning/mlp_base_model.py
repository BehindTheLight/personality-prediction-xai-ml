# Cell 1: Imports and Variable Setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve

n_folds = 3
random_seed = 42
model_name = 'MLP_Base'
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
# Base MLP with default hyperparameters
clf = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',
    solver='adam',
    alpha=0.0001,  # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    shuffle=True,
    random_state=random_seed,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4,
    verbose=False
)

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

# Cell 5: Figure Generation
# ROC Curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % roc_auc, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - MLP Base Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, label='PR curve', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - MLP Base Model', fontsize=14, fontweight='bold')
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
plt.title('MLP Base Model Performance Metrics', fontsize=14, fontweight='bold')
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
plt.title('Confusion Matrix - MLP Base Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Cell 6: Save Model and Predictions
clf.fit(X, y)
joblib.dump(clf, os.path.join(model_dir, 'mlp_base_model.pkl'))
pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_proba': y_proba}).to_csv(os.path.join(results_dir, 'cv_predictions.csv'), index=False)

# Save model configuration
model_config = {
    'hidden_layer_sizes': clf.hidden_layer_sizes,
    'activation': clf.activation,
    'solver': clf.solver,
    'alpha': clf.alpha,
    'learning_rate': clf.learning_rate,
    'max_iter': clf.max_iter,
    'early_stopping': clf.early_stopping,
    'n_iter_no_change': clf.n_iter_no_change
}
pd.Series(model_config).to_csv(os.path.join(results_dir, 'model_config.csv'))

print(f"\n✓ MLP Base Model completed successfully!")
print(f"✓ Model saved to: {model_dir}")
print(f"✓ Results saved to: {results_dir}")
print(f"✓ Figures saved to: {figures_dir}") 