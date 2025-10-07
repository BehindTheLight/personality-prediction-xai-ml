import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay

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

# 1. SHAP Interaction Values Analysis
print("Computing SHAP interaction values...")
explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X)

# Calculate interaction importance matrix
n_features = X.shape[1]
interaction_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        if i != j:
            interaction_matrix[i, j] = np.abs(shap_interaction_values[:, i, j]).mean()

# Create interaction heatmap
plt.figure(figsize=(12, 10))
interaction_df = pd.DataFrame(
    interaction_matrix, 
    index=X.columns, 
    columns=X.columns
)

# Create heatmap
sns.heatmap(
    interaction_df, 
    annot=True, 
    fmt='.3f', 
    cmap='viridis', 
    square=True,
    cbar_kws={'label': 'Mean |SHAP Interaction|'}
)
plt.title('SHAP Feature Interaction Matrix')
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_interaction_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save interaction matrix
interaction_df.to_csv(os.path.join(xai_dir, 'shap_interaction_matrix.csv'))

# 2. Top Feature Interactions
print("Analyzing top feature interactions...")
# Flatten interaction matrix and get top interactions
interactions = []
for i in range(n_features):
    for j in range(i+1, n_features):
        interactions.append({
            'feature1': X.columns[i],
            'feature2': X.columns[j],
            'interaction_strength': interaction_matrix[i, j]
        })

interactions_df = pd.DataFrame(interactions)
interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
interactions_df.to_csv(os.path.join(xai_dir, 'top_feature_interactions.csv'), index=False)

print("Top 5 feature interactions:")
print(interactions_df.head())

# 3. Partial Dependence Plots (PDP)
print("Creating Partial Dependence Plots...")
top_features = ['social_activity_score', 'socializing_score', 'Time_spent_Alone', 'introversion_score']

for feature in top_features:
    plt.figure(figsize=(10, 6))
    
    # Create PDP
    pdp = PartialDependenceDisplay.from_estimator(
        model, 
        X, 
        [feature],
        percentiles=(0.05, 0.95),
        line_kw={'color': 'red', 'linewidth': 2}
    )
    
    plt.title(f'Partial Dependence Plot: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Partial dependence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, f'pdp_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 4. ICE Plots (Individual Conditional Expectation)
print("Creating ICE Plots...")
for feature in top_features[:2]:  # Top 2 features for ICE plots
    plt.figure(figsize=(12, 8))
    
    # Sample 50 individuals for ICE plot
    sample_indices = np.random.choice(X.index, size=50, replace=False)
    X_sample = X.loc[sample_indices]
    
    # Create ICE plot
    ice = PartialDependenceDisplay.from_estimator(
        model, 
        X_sample, 
        [feature],
        percentiles=(0.05, 0.95),
        line_kw={'alpha': 0.3, 'color': 'blue'},
        ice_lines_kw={'alpha': 0.1, 'color': 'blue'}
    )
    
    plt.title(f'ICE Plot: {feature} (50 samples)')
    plt.xlabel(feature)
    plt.ylabel('Predicted probability')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, f'ice_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 5. SHAP vs Traditional Feature Importance Comparison
print("Comparing SHAP vs Traditional Feature Importance...")

# Traditional Random Forest feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf_model.feature_importances_
})

# SHAP importance (from previous analysis)
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'SHAP_Importance': np.abs(shap_interaction_values).mean(axis=(0, 2))
})

# Merge and compare
importance_comparison = rf_importance.merge(shap_importance, on='Feature')
importance_comparison = importance_comparison.sort_values('SHAP_Importance', ascending=False)
importance_comparison.to_csv(os.path.join(xai_dir, 'feature_importance_comparison.csv'), index=False)

# Create comparison plot
plt.figure(figsize=(12, 8))
top_10_features = importance_comparison.head(10)

x = np.arange(len(top_10_features))
width = 0.35

plt.bar(x - width/2, top_10_features['RF_Importance'], width, label='Random Forest', alpha=0.8)
plt.bar(x + width/2, top_10_features['SHAP_Importance'], width, label='SHAP', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Comparison: Random Forest vs SHAP')
plt.xticks(x, top_10_features['Feature'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Correlation vs SHAP Importance Analysis
print("Analyzing correlation vs SHAP importance...")
# Calculate feature correlations with target
correlations = []
for feature in X.columns:
    corr = np.corrcoef(X[feature], y)[0, 1]
    correlations.append(abs(corr))

correlation_df = pd.DataFrame({
    'Feature': X.columns,
    'Abs_Correlation': correlations
})

# Merge with SHAP importance
correlation_shap = correlation_df.merge(
    importance_comparison[['Feature', 'SHAP_Importance']], 
    on='Feature'
)

# Create correlation vs SHAP scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(
    correlation_shap['Abs_Correlation'], 
    correlation_shap['SHAP_Importance'],
    alpha=0.7,
    s=100
)

# Add feature labels
for idx, row in correlation_shap.iterrows():
    plt.annotate(
        row['Feature'], 
        (row['Abs_Correlation'], row['SHAP_Importance']),
        xytext=(5, 5), 
        textcoords='offset points',
        fontsize=8
    )

plt.xlabel('Absolute Correlation with Target')
plt.ylabel('SHAP Importance')
plt.title('Feature Correlation vs SHAP Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'correlation_vs_shap_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save correlation analysis
correlation_shap.to_csv(os.path.join(xai_dir, 'correlation_vs_shap_analysis.csv'), index=False)

print("Feature Interaction Analysis Complete!")
print(f"Files saved to: {xai_dir}")
print("\nTop 3 feature interactions:")
print(interactions_df.head(3))
print("\nCorrelation between RF and SHAP importance:")
corr_coef = np.corrcoef(importance_comparison['RF_Importance'], importance_comparison['SHAP_Importance'])[0, 1]
print(f"Correlation coefficient: {corr_coef:.3f}") 