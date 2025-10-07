# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Set up paths and import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

project_dir = '/content/drive/MyDrive/Extro_Intro'
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
figures_dir = os.path.join(project_dir, 'figures')

# Cell 3: Load the dataset
train_path = os.path.join(data_dir, 'train.csv')
train = pd.read_csv(train_path)

# Cell 4: Boxplots for numeric features (outlier visualization)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(os.path.join(figures_dir, f'{col}_boxplot.png'))
    plt.close()

# Cell 5: Z-score outlier detection
outlier_summary = {}
for col in numeric_cols:
    if train[col].isnull().all():
        outlier_summary[col] = {'num_outliers': 0, 'percent_outliers': 0}
        continue
    z_scores = np.abs(zscore(train[col].dropna()))
    num_outliers = (z_scores > 3).sum()
    percent_outliers = 100 * num_outliers / train[col].dropna().shape[0]
    outlier_summary[col] = {'num_outliers': int(num_outliers), 'percent_outliers': percent_outliers}

# Save outlier summary to CSV
outlier_df = pd.DataFrame(outlier_summary).T
outlier_df.to_csv(os.path.join(results_dir, 'outlier_summary.csv'))
print(outlier_df) 