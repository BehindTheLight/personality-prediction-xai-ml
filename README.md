# Personality Prediction: Machine Learning Approaches for Introvert-Extrovert Classification

## Overview
This project presents a comprehensive comparison of 14 machine learning algorithms for personality prediction using behavioral data. We implement classical methods, advanced tree-based models, deep learning approaches, and ensemble techniques to classify individuals as introverts or extroverts based on social behavior patterns.

## Key Results
- **Best Model**: Stacking Ensemble with ROC-AUC of 0.9850
- **Highest Accuracy**: CatBoost Base with 97.62% accuracy
- **Dataset**: 18,524 training samples, 6,176 test samples
- **Features**: 8 selected features including engineered composite features

## Methodology
- **Data Preprocessing**: Missing value imputation, SMOTE resampling, adversarial validation
- **Feature Engineering**: Composite features (Social Activity Score, Socializing Score, Introversion Score)
- **Model Categories**: Classical ML, Advanced Tree-based, Deep Learning, Ensemble Methods
- **Hyperparameter Optimization**: Optuna framework with 3-fold stratified cross-validation
- **Explainable AI**: SHAP and LIME analysis for model interpretability

## Project Structure
```
├── data/                           # Processed dataset files
├── notebooks/                      # Complete implementation (89 files)
│   ├── 01_Data_Processing/         # Data exploration, cleaning, feature engineering
│   ├── 02_Model_Development/       # 14 ML models (base + tuned versions)
│   ├── 03_Ensemble_Methods/        # Voting and stacking ensembles
│   ├── 04_Deep_Learning/           # MLP and TabNet implementations
│   ├── 05_Analysis_Tools/          # XAI and validation tools
│   ├── 06_XAI_Analysis/            # SHAP and LIME analysis results
│   └── 07_Adversarial_Validation/  # Distribution validation
├── models/                         # Trained model files only
│   ├── Classical Models: LogisticRegression, SVM, RandomForest, KNN, ExtraTrees, AdaBoost
│   ├── Advanced Models: XGBoost, LightGBM, CatBoost (Base + Tuned versions)
│   ├── Deep Learning: MLP, TabNet (Base + Tuned versions)
│   ├── Ensemble Methods: VotingEnsemble, Stacking_Ensemble
│   └── Each model includes: trained model file (.pkl/.cbm)
├── results/                        # Model performance data
│   ├── model_results/              # Individual model results
│   ├── model_performance_summary.csv
│   ├── feature_importance_analysis.csv
│   └── ensemble_configuration.csv
├── figures/                        # All visualizations and plots
│   ├── model_plots/                # Individual model performance plots
│   └── Key project visualizations
├── documentation/                  # Report paper and project documentation
└── requirements.txt                # Python dependencies
```

## Key Findings
1. **Ensemble Methods Dominate**: Stacking ensemble achieves highest performance
2. **Feature Engineering Impact**: Engineered composite features rank highest in importance
3. **Social Behavior Dominance**: Social activity patterns are strongest predictors
4. **Model Interpretability**: SHAP analysis reveals clear feature importance hierarchy

## Technologies Used
- **Python Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch
- **Hyperparameter Optimization**: Optuna
- **Explainable AI**: SHAP, LIME
- **Visualization**: Matplotlib, Seaborn
- **Documentation**: LaTeX

## Implementation Details
This repository contains the complete implementation with:
- **Complete model implementations** for all 14 algorithms with hyperparameter tuning
- **Complete trained models** (ALL 24 models) including saved model files
- **Comprehensive results** with detailed performance metrics for every model
- **Comprehensive XAI analysis** with SHAP and LIME implementations
- **Adversarial validation** to ensure reliable model evaluation
- **Research report** (`Personality_Prediction_Research_Paper.pdf`) with detailed methodology and results

## Research Report
The complete report on the project (`Personality_Prediction_Report.pdf`) is available in the documentation folder, presenting detailed methodology, results, and analysis.


## Dataset
Kaggle Competition: "Predict the Introverts from the Extroverts" (Playground Series - Season 5, Episode 7)

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order:
   - Start with `01_Data_Processing/` for data exploration and preprocessing
   - Continue with `02_Model_Development/` for model training
   - Explore `05_Analysis_Tools/` and `06_XAI_Analysis/` for interpretability analysis
4. **ALL pre-trained models** (24 models) are available in the `models/` folder for immediate use
5. **Complete results and visualizations** are included for every model

## License
This project is for academic and research purposes.
