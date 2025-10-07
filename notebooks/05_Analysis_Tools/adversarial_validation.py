import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Output directory
output_dir = '/content/drive/MyDrive/Extro_Intro/Adversarial_Validation'
os.makedirs(output_dir, exist_ok=True)

# Load data
train = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/train_selected.csv')
test = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/test_selected.csv')

# Remove target from train
X_train = train.drop(columns=['Personality'])
X_test = test.copy()

# Add is_test label
X_train['is_test'] = 0
X_test['is_test'] = 1

# Combine
adv_data = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_adv = adv_data['is_test'].values
X_adv = adv_data.drop(columns=['is_test'])

# Train adversarial classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
auc_scores = cross_val_score(clf, X_adv, y_adv, cv=skf, scoring='roc_auc')
mean_auc = auc_scores.mean()
std_auc = auc_scores.std()

print(f"Adversarial Validation ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# Save ROC-AUC mean and std
with open(os.path.join(output_dir, 'adversarial_validation_auc.txt'), 'w') as f:
    f.write(f"Adversarial Validation ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
    f.write(f"CV Scores: {auc_scores.tolist()}\n")

# Save CV scores
pd.DataFrame({'cv_roc_auc': auc_scores}).to_csv(os.path.join(output_dir, 'adversarial_cv_scores.csv'), index=False)

# Fit on all data for feature importance
clf.fit(X_adv, y_adv)
importances = pd.Series(clf.feature_importances_, index=X_adv.columns).sort_values(ascending=False)
print("Top features distinguishing train/test:")
print(importances.head(10))

# Save top 10 feature importances
importances.head(10).to_csv(os.path.join(output_dir, 'adversarial_top10_features.csv'), header=['importance'])
# Save all feature importances
importances.to_csv(os.path.join(output_dir, 'adversarial_feature_importances.csv'), header=['importance'])

# Plot feature importances
plt.figure(figsize=(8,5))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title('Top 10 Features Distinguishing Train/Test (Adversarial Validation)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'adversarial_feature_importance.png'))
plt.show() 