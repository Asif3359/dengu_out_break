Model performance assessment
Strengths
High R² on validation (0.9556): explains ~95.6% of variance.
Strong directional accuracy (99.46%): good for trend prediction.
Test performance: MAE 34.38, RMSE 200.17 — better than validation, suggesting reasonable generalization.
Useful for outbreak prediction: captures patterns well.
Concerns
Overfitting:
Training R²: 0.9994 vs Validation R²: 0.9556
Training MAE: 2.86 vs Validation MAE: 47.27
Large gap indicates overfitting.
High RMSE relative to MAE:
RMSE (200.17) >> MAE (34.38) suggests some large errors.
Recommendations
Reduce overfitting:
Increase min_samples_split and min_samples_leaf
Reduce max_depth (currently 20)
Increase max_features
Add more regularization
Evaluate error distribution:
Check if large errors are concentrated in specific regions/time periods
Consider region-specific models if needed
Compare with other models:
XGBoost showed similar performance (MAE: 53.89)
An ensemble might help
Verdict
The model is relevant and useful for dengue outbreak prediction, especially given:
High directional accuracy (99.46%)
Good R² (0.9556)
Reasonable test performance
The overfitting gap should be addressed to improve robustness. Should I:
Add hyperparameter tuning to reduce overfitting?
Create an ensemble of Random Forest and XGBoost?
Analyze where the model makes the largest errors?