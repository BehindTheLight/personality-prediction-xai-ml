import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
model_path = '/content/drive/MyDrive/Extro_Intro/models/LightGBM_Tuned/lightgbm_tuned_model.pkl'
data_path = '/content/drive/MyDrive/Extro_Intro/data/train_selected.csv'
xai_dir = '/content/drive/MyDrive/Extro_Intro/XAI'
os.makedirs(xai_dir, exist_ok=True)

# Load model and data
print("Loading model and data...")
model = joblib.load(model_path)
data = pd.read_csv(data_path)
X = data.drop(columns=['Personality'])
y = data['Personality']

# SHAP analysis
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 1. SHAP Dependence Plots for Top Features
print("Creating SHAP dependence plots...")
top_features = ['social_activity_score', 'socializing_score', 'Time_spent_Alone', 'introversion_score']

for feature in top_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, 
        shap_values, 
        X, 
        interaction_index=None,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, f'shap_dependence_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 2. SHAP Interaction Plots
print("Creating SHAP interaction plots...")
# Most important interaction: social_activity_score vs Time_spent_Alone
plt.figure(figsize=(12, 8))
shap.dependence_plot(
    'social_activity_score', 
    shap_values, 
    X, 
    interaction_index='Time_spent_Alone',
    show=False
)
plt.title('SHAP Interaction: social_activity_score vs Time_spent_Alone')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_interaction_social_activity_vs_time_alone.png'), dpi=300, bbox_inches='tight')
plt.close()

# Another important interaction: socializing_score vs introversion_score
plt.figure(figsize=(12, 8))
shap.dependence_plot(
    'socializing_score', 
    shap_values, 
    X, 
    interaction_index='introversion_score',
    show=False
)
plt.title('SHAP Interaction: socializing_score vs introversion_score')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_interaction_socializing_vs_introversion.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. SHAP Waterfall Plots for Specific Samples
print("Creating SHAP waterfall plots...")

# Sample 1: Clear introvert (low social_activity_score, high Time_spent_Alone)
introvert_sample_idx = X[X['social_activity_score'] < X['social_activity_score'].quantile(0.1)].index[0]
introvert_sample = X.iloc[introvert_sample_idx]

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[introvert_sample_idx],
        base_values=explainer.expected_value,
        data=introvert_sample.values,
        feature_names=X.columns
    ),
    show=False
)
plt.title(f'SHAP Waterfall Plot: Introvert Sample (Index {introvert_sample_idx})')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_waterfall_introvert_sample.png'), dpi=300, bbox_inches='tight')
plt.close()

# Sample 2: Clear extrovert (high social_activity_score, low Time_spent_Alone)
extrovert_sample_idx = X[X['social_activity_score'] > X['social_activity_score'].quantile(0.9)].index[0]
extrovert_sample = X.iloc[extrovert_sample_idx]

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[extrovert_sample_idx],
        base_values=explainer.expected_value,
        data=extrovert_sample.values,
        feature_names=X.columns
    ),
    show=False
)
plt.title(f'SHAP Waterfall Plot: Extrovert Sample (Index {extrovert_sample_idx})')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_waterfall_extrovert_sample.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. SHAP Force Plot for Overview
print("Creating SHAP force plot...")
plt.figure(figsize=(16, 8))
shap.force_plot(
    explainer.expected_value,
    shap_values[:100],  # First 100 samples for clarity
    X.iloc[:100],
    show=False
)
plt.title('SHAP Force Plot: First 100 Samples')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_force_plot_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Save sample data for waterfall plots
sample_data = pd.DataFrame({
    'sample_idx': [introvert_sample_idx, extrovert_sample_idx],
    'personality': ['introvert', 'extrovert'],
    'social_activity_score': [introvert_sample['social_activity_score'], extrovert_sample['social_activity_score']],
    'time_spent_alone': [introvert_sample['Time_spent_Alone'], extrovert_sample['Time_spent_Alone']],
    'prediction': [model.predict_proba([introvert_sample])[0][1], model.predict_proba([extrovert_sample])[0][1]]
})
sample_data.to_csv(os.path.join(xai_dir, 'shap_waterfall_sample_data.csv'), index=False)

# 6. Feature Effect Summary
print("Creating feature effect summary...")
feature_effects = {}
for feature in X.columns:
    # Calculate mean absolute SHAP value for this feature
    mean_abs_shap = np.abs(shap_values[:, X.columns.get_loc(feature)]).mean()
    feature_effects[feature] = mean_abs_shap

feature_effects_df = pd.DataFrame(list(feature_effects.items()), columns=['Feature', 'Mean_Abs_SHAP'])
feature_effects_df = feature_effects_df.sort_values('Mean_Abs_SHAP', ascending=False)
feature_effects_df.to_csv(os.path.join(xai_dir, 'shap_feature_effects_summary.csv'), index=False)

print("SHAP Dependence Analysis Complete!")
print(f"Files saved to: {xai_dir}")
print("\nTop 5 features by mean absolute SHAP:")
print(feature_effects_df.head()) 