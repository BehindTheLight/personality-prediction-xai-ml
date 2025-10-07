import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Paths
model_path = '/content/drive/MyDrive/Extro_Intro/models/LightGBM_Tuned/lightgbm_tuned_model.pkl'
data_path = '/content/drive/MyDrive/Extro_Intro/data/train_selected.csv'
xai_dir = '/content/drive/MyDrive/Extro_Intro/XAI'
os.makedirs(xai_dir, exist_ok=True)

# Load model and data
model = joblib.load(model_path)
data = pd.read_csv(data_path)
X = data.drop(columns=['Personality'])

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP summary plot (global feature importance)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_summary_plot_lightgbm.png'))
plt.close()

# Top 10 features by mean(|SHAP|)
shap_abs = pd.DataFrame(abs(shap_values), columns=X.columns)
mean_abs_shap = shap_abs.mean().sort_values(ascending=False)
top10 = mean_abs_shap.head(10)
top10.to_csv(os.path.join(xai_dir, 'shap_top10_features_lightgbm.csv'), header=['mean_abs_shap'])
print('Top 10 features by mean(|SHAP|):')
print(top10)

# Save SHAP values for first 5 samples
shap_first5 = pd.DataFrame(shap_values[:5], columns=X.columns)
shap_first5.to_csv(os.path.join(xai_dir, 'shap_values_lightgbm_first5.csv'), index=False) 