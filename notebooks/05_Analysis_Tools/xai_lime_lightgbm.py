# If running in Colab, uncomment the next line:
# !pip install lime

import os
import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Paths
model_path = '/content/drive/MyDrive/Extro_Intro/models/LightGBM_Tuned/lightgbm_tuned_model.pkl'
data_path = '/content/drive/MyDrive/Extro_Intro/data/train_selected.csv'
xai_dir = '/content/drive/MyDrive/Extro_Intro/XAI'
os.makedirs(xai_dir, exist_ok=True)

# Load model and data
model = joblib.load(model_path)
data = pd.read_csv(data_path)
X = data.drop(columns=['Personality'])
y = data['Personality']

# LIME explainer
explainer = LimeTabularExplainer(
    X.values,
    feature_names=X.columns.tolist(),
    class_names=['Introvert', 'Extrovert'],
    mode='classification',
    discretize_continuous=True
)

# Explain the first sample
sample_idx = 0
exp = explainer.explain_instance(
    X.iloc[sample_idx].values,
    model.predict_proba,
    num_features=8
)

# Save explanation as HTML
lime_html_path = os.path.join(xai_dir, 'lime_explanation_lightgbm_sample0.html')
exp.save_to_file(lime_html_path)

# Print top features for this explanation
print(f'LIME explanation for sample {sample_idx}:')
for feature, weight in exp.as_list():
    print(f'{feature}: {weight}') 