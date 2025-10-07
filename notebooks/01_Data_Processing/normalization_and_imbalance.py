# Cell: Normalize/Standardize Numeric Features and Address Class Imbalance
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Assume train and test DataFrames are already loaded in the Colab environment
# If not, uncomment and adjust the following lines:
# train = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/train_cleaned.csv')
# test = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/test_cleaned.csv')

# Identify numeric columns (excluding id and target)
numeric_cols = [col for col in train.columns if train[col].dtype in [np.float64, np.int64] and col not in ['id', 'Personality']]

# Standardize numeric features
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

# Address class imbalance using SMOTE (only for training data)
X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Reconstruct the balanced train DataFrame
train_final = pd.DataFrame(X_res, columns=X.columns)
train_final['Personality'] = y_res
# Optionally, add back the id column if needed (here, omitted as SMOTE generates synthetic samples)

# Save processed data
output_dir = '/content/drive/MyDrive/Extro_Intro/data'
train_final.to_csv(os.path.join(output_dir, 'train_final.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test_final.csv'), index=False)
print('Final processed train and test data saved.') 