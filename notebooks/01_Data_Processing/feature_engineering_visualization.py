# Cell: Visualize Engineered Features
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Assume train_final DataFrame is already loaded and includes new features
# If not, uncomment and adjust:
# train_final = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/train_engineered.csv')

features_to_plot = ['socializing_score', 'social_activity_score', 'introversion_score']
figures_dir = '/content/drive/MyDrive/Extro_Intro/figures'
os.makedirs(figures_dir, exist_ok=True)

for feat in features_to_plot:
    # Histogram
    plt.figure()
    sns.histplot(train_final[feat], kde=True)
    plt.title(f'Distribution of {feat}')
    plt.savefig(os.path.join(figures_dir, f'{feat}_hist.png'))
    plt.close()
    # Boxplot by Personality
    plt.figure()
    sns.boxplot(x='Personality', y=feat, data=train_final)
    plt.title(f'{feat} by Personality')
    plt.savefig(os.path.join(figures_dir, f'{feat}_by_personality_box.png'))
    plt.close()
print('Visualizations for engineered features saved.') 