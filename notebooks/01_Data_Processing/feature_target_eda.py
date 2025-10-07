# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Set up paths and import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = '/content/drive/MyDrive/Extro_Intro'
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
figures_dir = os.path.join(project_dir, 'figures')

# Cell 3: Load the dataset
train_path = os.path.join(data_dir, 'train.csv')
train = pd.read_csv(train_path)

# Cell 4: Groupby statistics for numeric features by Personality
gb = train.groupby('Personality')
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
gb_stats = gb[numeric_cols].mean().T
print(gb_stats)
gb_stats.to_csv(os.path.join(results_dir, 'groupby_personality_means.csv'))

# Cell 5: Boxplots for numeric features by Personality
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} by Personality')
    plt.savefig(os.path.join(figures_dir, f'{col}_by_personality_box.png'))
    plt.close()

# Cell 6: Bar plots for categorical features by Personality
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['Personality', 'id']]
for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, hue='Personality', data=train)
    plt.title(f'{col} by Personality')
    plt.savefig(os.path.join(figures_dir, f'{col}_by_personality_bar.png'))
    plt.close()

# Cell 7: Correlation heatmap (numeric features)
corr = train[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'))
plt.close() 