import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
output_dir = '/content/drive/MyDrive/Extro_Intro/Model_Comparison_Plots'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
data_path = '/content/drive/MyDrive/Extro_Intro/data/train_selected.csv'
data = pd.read_csv(data_path)
X = data.drop(columns=['Personality'])
y = data['Personality']

# Define model paths and names
model_configs = {
    'Logistic Regression': '/content/drive/MyDrive/Extro_Intro/models/LogisticRegression_Tuned/logistic_regression_tuned_model.pkl',
    'SVM': '/content/drive/MyDrive/Extro_Intro/models/SVM_Tuned/svm_tuned_model.pkl',
    'Random Forest': '/content/drive/MyDrive/Extro_Intro/models/RandomForest_Tuned/random_forest_tuned_model.pkl',
    'KNN': '/content/drive/MyDrive/Extro_Intro/models/KNN_Tuned/knn_tuned_model.pkl',
    'XGBoost': '/content/drive/MyDrive/Extro_Intro/models/XGBoost_Tuned/xgboost_tuned_model.pkl',
    'LightGBM': '/content/drive/MyDrive/Extro_Intro/models/LightGBM_Tuned/lightgbm_tuned_model.pkl',
    'CatBoost': '/content/drive/MyDrive/Extro_Intro/models/CatBoost_Tuned/catboost_tuned_model.cbm',
    'Extra Trees': '/content/drive/MyDrive/Extro_Intro/models/ExtraTrees_Tuned/extra_trees_tuned_model.pkl',
    'AdaBoost': '/content/drive/MyDrive/Extro_Intro/models/AdaBoost_Tuned/adaboost_tuned_model.pkl',
    'MLP (sklearn)': '/content/drive/MyDrive/Extro_Intro/models/MLP_Tuned/mlp_tuned_model.pkl',
    'MLP (PyTorch)': '/content/drive/MyDrive/Extro_Intro/models/MLP_PyTorch_Tuned/mlp_pytorch_tuned_model.pth',
    'TabNet': '/content/drive/MyDrive/Extro_Intro/models/TabNet_Tuned/tabnet_tuned_model.zip',
    'Voting Ensemble (Soft)': '/content/drive/MyDrive/Extro_Intro/models/VotingEnsemble_Optimized/voting_ensemble_optimized_model.pkl',
    'Voting Ensemble (Hard)': '/content/drive/MyDrive/Extro_Intro/models/VotingEnsemble_Hard/voting_ensemble_hard_model.pkl',
    'Stacking Ensemble': '/content/drive/MyDrive/Extro_Intro/models/Stacking_Ensemble/stacking_ensemble_model.pkl'
}

# Load models and calculate metrics
print("Loading models and calculating metrics...")
results = []

for model_name, model_path in model_configs.items():
    try:
        # Load model
        if model_path.endswith('.cbm'):
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(model_path)
        elif model_path.endswith('.pth'):
            import torch
            from torch import nn
            # Load PyTorch model (simplified loading)
            model = joblib.load(model_path.replace('.pth', '_wrapper.pkl'))
        elif model_path.endswith('.zip'):
            from pytorch_tabnet.tab_model import TabNetClassifier
            model = TabNetClassifier()
            model.load_model(model_path)
        else:
            model = joblib.load(model_path)
        
        # Calculate cross-validation scores
        cv_accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        cv_f1 = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
        cv_roc_auc = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
        cv_precision = cross_val_score(model, X, y, cv=3, scoring='precision').mean()
        cv_recall = cross_val_score(model, X, y, cv=3, scoring='recall').mean()
        
        results.append({
            'Model': model_name,
            'Accuracy': cv_accuracy,
            'F1-Score': cv_f1,
            'ROC-AUC': cv_roc_auc,
            'Precision': cv_precision,
            'Recall': cv_recall
        })
        print(f"Loaded: {model_name}")
        
    except Exception as e:
        print(f"Failed: {model_name}: {e}")
        continue

# Create DataFrame
df_results = pd.DataFrame(results)
print(f"\nSuccessfully loaded {len(df_results)} models")

# 1. TOP 5 MODELS BY ROC-AUC
print("\n1. Creating Top 5 Models by ROC-AUC plot...")
top5_roc = df_results.nlargest(5, 'ROC-AUC')

plt.figure(figsize=(12, 8))
bars = plt.barh(top5_roc['Model'], top5_roc['ROC-AUC'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('Top 5 Models by ROC-AUC Score', fontsize=16, fontweight='bold', pad=20)
plt.xlim(0.8, 1.0)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top5_roc['ROC-AUC'])):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{value:.4f}', ha='left', va='center', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top5_models_roc_auc.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. TOP 5 MODELS BY ACCURACY
print("2. Creating Top 5 Models by Accuracy plot...")
top5_acc = df_results.nlargest(5, 'Accuracy')

plt.figure(figsize=(12, 8))
bars = plt.barh(top5_acc['Model'], top5_acc['Accuracy'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.xlabel('Accuracy Score', fontsize=12, fontweight='bold')
plt.title('Top 5 Models by Accuracy Score', fontsize=16, fontweight='bold', pad=20)
plt.xlim(0.8, 1.0)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top5_acc['Accuracy'])):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{value:.4f}', ha='left', va='center', fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top5_models_accuracy.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. COMPREHENSIVE MODEL COMPARISON HEATMAP
print("3. Creating comprehensive model comparison heatmap...")
metrics = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
comparison_df = df_results.set_index('Model')[metrics]

plt.figure(figsize=(14, 10))
sns.heatmap(comparison_df, annot=True, fmt='.4f', cmap='RdYlBu_r', 
            cbar_kws={'label': 'Score'}, linewidths=0.5)
plt.title('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. RADAR CHART FOR TOP 5 MODELS
print("4. Creating radar chart for top 5 models...")
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(theta, *args, closed=closed, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(theta, *args, **kwargs)
            return lines
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
    register_projection(RadarAxes)
    return theta

# Prepare data for radar chart
top5_models = df_results.nlargest(5, 'ROC-AUC')
theta = radar_factory(len(metrics), frame='polygon')

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
for i, (idx, row) in enumerate(top5_models.iterrows()):
    values = [row[metric] for metric in metrics]
    ax.plot(theta, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
    ax.fill(theta, values, alpha=0.25, color=colors[i])

ax.set_varlabels(metrics)
ax.set_ylim(0.7, 1.0)
plt.title('Top 5 Models Performance Radar Chart', size=16, y=1.1, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top5_models_radar.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. MODEL CATEGORY COMPARISON
print("5. Creating model category comparison...")
# Categorize models
df_results['Category'] = df_results['Model'].apply(lambda x: 
    'Ensemble' if 'Ensemble' in x or 'Stacking' in x
    else 'Deep Learning' if 'MLP' in x or 'TabNet' in x
    else 'Tree-based' if any(tree in x for tree in ['Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Extra Trees', 'AdaBoost'])
    else 'Linear' if 'Logistic' in x or 'SVM' in x
    else 'Distance-based' if 'KNN' in x
    else 'Other'
)

category_avg = df_results.groupby('Category')[metrics].mean().round(4)

plt.figure(figsize=(12, 8))
category_avg.plot(kind='bar', figsize=(12, 8), width=0.8)
plt.title('Average Performance by Model Category', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model Category', fontsize=12, fontweight='bold')
plt.ylabel('Average Score', fontsize=12, fontweight='bold')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_category_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. PERFORMANCE RANKING TABLE
print("6. Creating performance ranking table...")
# Create ranking table
ranking_df = df_results.copy()
ranking_df['ROC-AUC Rank'] = ranking_df['ROC-AUC'].rank(ascending=False).astype(int)
ranking_df['Accuracy Rank'] = ranking_df['Accuracy'].rank(ascending=False).astype(int)
ranking_df['F1-Score Rank'] = ranking_df['F1-Score'].rank(ascending=False).astype(int)
ranking_df['Overall Rank'] = (ranking_df['ROC-AUC Rank'] + ranking_df['Accuracy Rank'] + ranking_df['F1-Score Rank']) / 3
ranking_df = ranking_df.sort_values('Overall Rank')

# Save ranking table
ranking_df.to_csv(os.path.join(output_dir, 'model_ranking_table.csv'), index=False)

# Create ranking visualization
plt.figure(figsize=(14, 10))
ranking_data = ranking_df[['Model', 'ROC-AUC Rank', 'Accuracy Rank', 'F1-Score Rank']].set_index('Model')
ranking_data = ranking_data.sort_values('ROC-AUC Rank')

sns.heatmap(ranking_data, annot=True, fmt='d', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Rank (Lower is Better)'}, linewidths=0.5)
plt.title('Model Performance Rankings (Lower Numbers = Better Performance)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Models', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_ranking_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. SCATTER PLOT: ACCURACY vs ROC-AUC
print("7. Creating scatter plot: Accuracy vs ROC-AUC...")
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_results['ROC-AUC'], df_results['Accuracy'], 
                     s=100, alpha=0.7, c=range(len(df_results)), cmap='viridis')

# Add model labels
for i, row in df_results.iterrows():
    plt.annotate(row['Model'], (row['ROC-AUC'], row['Accuracy']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
plt.title('Model Performance: Accuracy vs ROC-AUC', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Model Index')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_vs_roc_auc_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

# 8. SUMMARY STATISTICS
print("8. Creating summary statistics...")
summary_stats = df_results[metrics].describe()
print("\nSummary Statistics:")
print(summary_stats)

# Save summary statistics
summary_stats.to_csv(os.path.join(output_dir, 'model_performance_summary.csv'))

# Create summary visualization
plt.figure(figsize=(12, 8))
summary_stats.loc[['mean', 'std']].T.plot(kind='bar', yerr=summary_stats.loc['std'], 
                                         capsize=5, figsize=(12, 8))
plt.title('Model Performance Summary: Mean ± Standard Deviation', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.legend(['Mean', 'Standard Deviation'])
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll comparison plots saved to: {output_dir}")
print(f"Generated {8} different comparison visualizations:")
print("1. Top 5 Models by ROC-AUC")
print("2. Top 5 Models by Accuracy") 
print("3. Comprehensive Model Comparison Heatmap")
print("4. Top 5 Models Radar Chart")
print("5. Model Category Comparison")
print("6. Model Performance Rankings")
print("7. Accuracy vs ROC-AUC Scatter Plot")
print("8. Performance Summary Statistics")

# Print top 5 models
print(f"\nTOP 5 MODELS BY ROC-AUC:")
print(df_results.nlargest(5, 'ROC-AUC')[['Model', 'ROC-AUC', 'Accuracy', 'F1-Score']].to_string(index=False)) 