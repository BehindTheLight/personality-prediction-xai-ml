# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Set up paths and import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

project_dir = '/content/drive/MyDrive/Extro_Intro'
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')

# Cell 3: Load the datasets
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Cell 4: Handle missing values
# Numeric: impute with median (fit on train, transform both)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
imputer_num = SimpleImputer(strategy='median')
train[numeric_cols] = imputer_num.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer_num.transform(test[numeric_cols])

# Categorical: impute with most frequent (fit on train, transform both)
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['id', 'Personality']]
imputer_cat = SimpleImputer(strategy='most_frequent')
train[cat_cols] = imputer_cat.fit_transform(train[cat_cols])
test[cat_cols] = imputer_cat.transform(test[cat_cols])

# Cell 5: Encode categorical variables
# Encode Personality as binary (Extrovert=1, Introvert=0) for train only
train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})
# Encode other categoricals with LabelEncoder (fit on train, transform both)
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Cell 6: Save cleaned data
cleaned_train_path = os.path.join(data_dir, 'train_cleaned.csv')
cleaned_test_path = os.path.join(data_dir, 'test_cleaned.csv')
train.to_csv(cleaned_train_path, index=False)
test.to_csv(cleaned_test_path, index=False)
print('Cleaned train data saved to:', cleaned_train_path)
print('Cleaned test data saved to:', cleaned_test_path) 