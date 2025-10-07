# Cell: Feature Selection using Random Forest Feature Importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

# Assume train_final and test_final DataFrames are already loaded
# If not, uncomment and adjust:
# train_final = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/train_engineered.csv')
# test_final = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/test_engineered.csv')

# Separate features and target
y = train_final['Personality']
X = train_final.drop(['Personality'], axis=1)

# Fit Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Extro_Intro/figures/feature_importances.png')
plt.close()

# Select top N features
N = 8
top_features = feat_imp.index[:N].tolist()
print('Top features:', top_features)

# Reduce train and test sets to top features + target
train_selected = train_final[top_features + ['Personality']]
test_selected = test_final[top_features]

# Save reduced datasets
output_dir = '/content/drive/MyDrive/Extro_Intro/data'
train_selected.to_csv(os.path.join(output_dir, 'train_selected.csv'), index=False)
test_selected.to_csv(os.path.join(output_dir, 'test_selected.csv'), index=False)
print('Feature-selected train and test data saved.') 