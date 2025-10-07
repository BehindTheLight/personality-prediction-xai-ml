# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Set up paths and import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set your project directory in Google Drive
project_dir = '/content/drive/MyDrive/Extro_Intro'
os.makedirs(project_dir, exist_ok=True)
data_dir = os.path.join(project_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
figures_dir = os.path.join(project_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Cell 3: Load the dataset
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')

# If you haven't already, copy your train.csv and test.csv to the data_dir in Google Drive
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Cell 4: Initial Data Inspection
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print(train.head())

# Cell 5: Check for missing values
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
print("Missing values in train:\n", missing_train)
print("Missing values in test:\n", missing_test)

# Save missing value report
missing_train.to_csv(os.path.join(results_dir, 'missing_train.csv'))
missing_test.to_csv(os.path.join(results_dir, 'missing_test.csv'))

# Cell 6: Basic statistics
desc_train = train.describe(include='all')
desc_test = test.describe(include='all')
desc_train.to_csv(os.path.join(results_dir, 'desc_train.csv'))
desc_test.to_csv(os.path.join(results_dir, 'desc_test.csv'))

# Cell 7: Visualize distributions
for col in train.select_dtypes(include=[np.number]).columns:
    plt.figure()
    sns.histplot(train[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(figures_dir, f'{col}_hist.png'))
    plt.close()

# Cell 8: Visualize target variable
plt.figure()
sns.countplot(x='Personality', data=train)
plt.title('Class Distribution')
plt.savefig(os.path.join(figures_dir, 'class_distribution.png'))
plt.close() 