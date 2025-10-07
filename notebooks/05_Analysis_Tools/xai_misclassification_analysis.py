import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

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

# Fix GPU issue by forcing CPU mode for cross-validation
print("Fixing GPU compatibility for cross-validation...")
original_device = model.get_params().get('device', 'auto')
model.set_params(device='cpu')  # Force CPU mode

# 1. Identify Misclassified Samples
print("Identifying misclassified samples...")
# Get predictions using cross-validation to avoid data leakage
y_pred_cv = cross_val_predict(model, X, y, cv=3, method='predict')
y_proba_cv = cross_val_predict(model, X, y, cv=3, method='predict_proba')[:, 1]

# Restore original device setting
model.set_params(device=original_device)

# Create results dataframe
results_df = pd.DataFrame({
    'true_label': y,
    'predicted_label': y_pred_cv,
    'predicted_probability': y_proba_cv,
    'is_correct': y == y_pred_cv,
    'confidence': np.maximum(y_proba_cv, 1 - y_proba_cv)  # Distance from 0.5
})

# Add original features
for col in X.columns:
    results_df[col] = X[col].values

# Identify misclassified samples
misclassified = results_df[~results_df['is_correct']].copy()
correctly_classified = results_df[results_df['is_correct']].copy()

print(f"Total samples: {len(results_df)}")
print(f"Correctly classified: {len(correctly_classified)} ({len(correctly_classified)/len(results_df)*100:.1f}%)")
print(f"Misclassified: {len(misclassified)} ({len(misclassified)/len(results_df)*100:.1f}%)")

# 2. Analyze Misclassification Patterns
print("Analyzing misclassification patterns...")

# Confusion matrix analysis
cm = confusion_matrix(y, y_pred_cv)
print("\nConfusion Matrix:")
print(cm)

# False Positives (predicted extrovert, actually introvert)
false_positives = misclassified[misclassified['true_label'] == 0]
# False Negatives (predicted introvert, actually extrovert)
false_negatives = misclassified[misclassified['true_label'] == 1]

print(f"\nFalse Positives (predicted extrovert, actually introvert): {len(false_positives)}")
print(f"False Negatives (predicted introvert, actually extrovert): {len(false_negatives)}")

# 3. SHAP Analysis for Misclassified Samples
print("Performing SHAP analysis for misclassified samples...")
# Ensure model is in CPU mode for SHAP analysis
model.set_params(device='cpu')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Get SHAP values for misclassified samples
misclassified_shap = shap_values[misclassified.index]
false_pos_shap = shap_values[false_positives.index]
false_neg_shap = shap_values[false_negatives.index]

# Compare SHAP values between correctly and incorrectly classified
correct_shap = shap_values[correctly_classified.index]

# Calculate mean absolute SHAP values for each group
def get_mean_abs_shap(shap_vals):
    return np.abs(shap_vals).mean(axis=0)

correct_mean_shap = get_mean_abs_shap(correct_shap)
misclassified_mean_shap = get_mean_abs_shap(misclassified_shap)
fp_mean_shap = get_mean_abs_shap(false_pos_shap)
fn_mean_shap = get_mean_abs_shap(false_neg_shap)

# Create comparison plot
plt.figure(figsize=(14, 8))
x = np.arange(len(X.columns))
width = 0.2

plt.bar(x - 1.5*width, correct_mean_shap, width, label='Correctly Classified', alpha=0.8)
plt.bar(x - 0.5*width, misclassified_mean_shap, width, label='All Misclassified', alpha=0.8)
plt.bar(x + 0.5*width, fp_mean_shap, width, label='False Positives', alpha=0.8)
plt.bar(x + 1.5*width, fn_mean_shap, width, label='False Negatives', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Mean |SHAP Value|')
plt.title('SHAP Values Comparison: Correct vs Misclassified Samples')
plt.xticks(x, X.columns, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'shap_misclassification_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Feature Value Analysis for Misclassified Samples
print("Analyzing feature values for misclassified samples...")

# Compare feature distributions
feature_comparison = []
for feature in X.columns:
    correct_mean = correctly_classified[feature].mean()
    misclassified_mean = misclassified[feature].mean()
    fp_mean = false_positives[feature].mean()
    fn_mean = false_negatives[feature].mean()
    
    feature_comparison.append({
        'Feature': feature,
        'Correct_Mean': correct_mean,
        'Misclassified_Mean': misclassified_mean,
        'False_Positive_Mean': fp_mean,
        'False_Negative_Mean': fn_mean,
        'Correct_vs_Misclassified_Diff': correct_mean - misclassified_mean
    })

feature_comparison_df = pd.DataFrame(feature_comparison)
feature_comparison_df = feature_comparison_df.sort_values('Correct_vs_Misclassified_Diff', key=abs, ascending=False)
feature_comparison_df.to_csv(os.path.join(xai_dir, 'misclassification_feature_analysis.csv'), index=False)

# Create feature comparison plot
plt.figure(figsize=(14, 8))
top_features = feature_comparison_df.head(8)['Feature'].values

x = np.arange(len(top_features))
width = 0.2

correct_means = [feature_comparison_df[feature_comparison_df['Feature'] == f]['Correct_Mean'].iloc[0] for f in top_features]
misclassified_means = [feature_comparison_df[feature_comparison_df['Feature'] == f]['Misclassified_Mean'].iloc[0] for f in top_features]
fp_means = [feature_comparison_df[feature_comparison_df['Feature'] == f]['False_Positive_Mean'].iloc[0] for f in top_features]
fn_means = [feature_comparison_df[feature_comparison_df['Feature'] == f]['False_Negative_Mean'].iloc[0] for f in top_features]

plt.bar(x - 1.5*width, correct_means, width, label='Correctly Classified', alpha=0.8)
plt.bar(x - 0.5*width, misclassified_means, width, label='All Misclassified', alpha=0.8)
plt.bar(x + 0.5*width, fp_means, width, label='False Positives', alpha=0.8)
plt.bar(x + 1.5*width, fn_means, width, label='False Negatives', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Feature Value')
plt.title('Feature Values Comparison: Correct vs Misclassified Samples')
plt.xticks(x, top_features, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'misclassification_feature_values.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Confidence Analysis
print("Analyzing prediction confidence...")

# Confidence distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(correctly_classified['confidence'], bins=30, alpha=0.7, label='Correct', density=True)
plt.hist(misclassified['confidence'], bins=30, alpha=0.7, label='Misclassified', density=True)
plt.xlabel('Prediction Confidence')
plt.ylabel('Density')
plt.title('Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(correctly_classified['predicted_probability'], correctly_classified['confidence'], 
           alpha=0.5, label='Correct', s=20)
plt.scatter(misclassified['predicted_probability'], misclassified['confidence'], 
           alpha=0.7, label='Misclassified', s=30, color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Confidence')
plt.title('Probability vs Confidence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(xai_dir, 'misclassification_confidence_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. LIME Analysis for Edge Cases
print("Performing LIME analysis for edge cases...")

# Create LIME explainer
lime_explainer = LimeTabularExplainer(
    X.values,
    feature_names=X.columns.tolist(),
    class_names=['Introvert', 'Extrovert'],
    mode='classification',
    discretize_continuous=True
)

# Analyze a few representative misclassified samples
edge_cases = []

# Case 1: High confidence false positive
if len(false_positives) > 0:
    high_conf_fp = false_positives.loc[false_positives['confidence'].idxmax()]
    edge_cases.append(('High_Confidence_False_Positive', high_conf_fp))

# Case 2: High confidence false negative
if len(false_negatives) > 0:
    high_conf_fn = false_negatives.loc[false_negatives['confidence'].idxmax()]
    edge_cases.append(('High_Confidence_False_Negative', high_conf_fn))

# Case 3: Low confidence correct prediction (near boundary)
low_conf_correct = correctly_classified.loc[correctly_classified['confidence'].idxmin()]
edge_cases.append(('Low_Confidence_Correct', low_conf_correct))

# Generate LIME explanations for edge cases
for case_name, sample in edge_cases:
    sample_idx = sample.name
    sample_features = X.iloc[sample_idx]
    
    # Generate LIME explanation
    exp = lime_explainer.explain_instance(
        sample_features.values,
        model.predict_proba,
        num_features=8
    )
    
    # Save explanation as HTML
    lime_html_path = os.path.join(xai_dir, f'lime_{case_name.lower()}.html')
    exp.save_to_file(lime_html_path)
    
    # Save explanation as PNG
    fig = exp.as_pyplot_figure()
    plt.title(f'LIME Explanation: {case_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, f'lime_{case_name.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print explanation
    print(f"\nLIME explanation for {case_name}:")
    print(f"True label: {sample['true_label']}, Predicted: {sample['predicted_label']}")
    print(f"Confidence: {sample['confidence']:.3f}")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight}")

# 7. Save comprehensive misclassification summary
print("Saving misclassification summary...")

summary_stats = {
    'Total_Samples': len(results_df),
    'Correctly_Classified': len(correctly_classified),
    'Misclassified': len(misclassified),
    'Accuracy': len(correctly_classified) / len(results_df),
    'False_Positives': len(false_positives),
    'False_Negatives': len(false_negatives),
    'Avg_Confidence_Correct': correctly_classified['confidence'].mean(),
    'Avg_Confidence_Misclassified': misclassified['confidence'].mean(),
    'Min_Confidence_Correct': correctly_classified['confidence'].min(),
    'Max_Confidence_Misclassified': misclassified['confidence'].max()
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv(os.path.join(xai_dir, 'misclassification_summary.csv'), index=False)

# Save detailed misclassified samples
misclassified.to_csv(os.path.join(xai_dir, 'misclassified_samples_detailed.csv'), index=False)

print("Misclassification Analysis Complete!")
print(f"Files saved to: {xai_dir}")
print(f"\nMisclassification Summary:")
print(f"Accuracy: {summary_stats['Accuracy']:.3f}")
print(f"False Positives: {summary_stats['False_Positives']}")
print(f"False Negatives: {summary_stats['False_Negatives']}")
print(f"Average confidence (correct): {summary_stats['Avg_Confidence_Correct']:.3f}")
print(f"Average confidence (misclassified): {summary_stats['Avg_Confidence_Misclassified']:.3f}") 