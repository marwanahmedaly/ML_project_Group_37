# Student Outcome Prediction (Dropout, Enrolled, Graduate)

A systematic machine-learning study that predicts whether a university student will drop out, stay enrolled or graduate. The notebook focuses on handling class imbalance, comparing model families fairly, and explaining the final model.

## Data

- `data.csv`: 4,424 student records, 36 predictors (demographics, admission data, parental background, first- and second-semester curricular units, and economic indicators) plus a three-class target: Dropout, Enrolled, Graduate.
- Semicolon-separated file. The structure matches the public "Predict Students' Dropout and Academic Success" dataset; add the original citation and license here before redistributing.

## What the notebook does

`ml_finall_project.ipynb` is organized in phases:

1. **Data audit and multicollinearity analysis.** Four highly correlated variables are replaced by three engineered performance measures (35 predictors) instead of being deleted.
2. **Stratified preprocessing.** Stratified 80/20 split (3,539 train, 885 test) and scaling fitted on the training data only.
3. **Baseline benchmarking** of six model families: logistic regression, decision tree, random forest, gradient boosting, XGBoost and an MLP, with balanced class weights. A three-way comparison tests original features, dropping correlated features and the engineered features.
4. **Hyperparameter optimization** of XGBoost with randomized search, **SMOTE versus class weighting**, and learning curves to check over- and under-fitting.
5. **Visualization and interpretation:** t-SNE of the data manifold and SHAP values with feature-importance cross-checks.
6. **Final evaluation:** confusion matrices, ROC and precision-recall curves, error analysis, and statistical significance tests between models.

## Key results (saved notebook outputs)

| Model / setting | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Tuned XGBoost (selected by macro-F1) | 73.79% | 0.7004 | Dropout recall 70.07% |
| Random forest baseline | 77.51% | 0.6988 | Highest accuracy; dropout recall 76.76% |
| Feature strategy: original / dropped / engineered | | 0.6955 / 0.6570 / 0.6988 | Dropping correlated features hurt |
| Class weighting vs SMOTE | | 0.7004 vs about 0.6969 | SMOTE changed class trade-offs, not every metric |

Cross-validation macro-F1 for the tuned model is 0.7018 with a standard deviation of 0.0104.

## Limitations

- Results are exploratory. Models were compared on the test split, and some cross-validation happens after scaling, so this is not an untouched final hold-out.
- The "Enrolled" class is hard to separate from the other two.
- Differences among the compared ensembles were not statistically significant.
- No claim is made that the model reduces real dropout; no intervention was tested.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ml_finall_project.ipynb
```

`requirements.txt` lists pandas, numpy, matplotlib, seaborn, shap, scikit-learn, xgboost and imbalanced-learn.

## Authors

Group 37. Add the names and course details here.
