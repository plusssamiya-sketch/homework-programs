# Machine Learning Assignments

This folder contains my machine learning homework scripts. The examples use generated data so the files can be rerun without extra datasets.

## Programs

- `linear_model_pipeline.py`: end-to-end regression pipeline with preprocessing, cross-validation, hyperparameter tuning, coefficient interpretation, and residual diagnostics.
- `classification_model_selection.py`: classification model comparison with stratified validation, ROC-AUC scoring, calibration-aware probability output, and a concise model card.
- `double_machine_learning.py`: cross-fitted causal machine learning assignment for estimating an average treatment effect in a partially linear model.

## Learning Objectives

- Build train/test workflows without data leakage.
- Compare baseline and regularized models using cross-validation.
- Interpret fitted models with metrics and diagnostic summaries.
- Keep experiments reproducible through fixed random seeds and structured output.
- Distinguish prediction performance from causal identification.

## Questions I Am Tracking

1. When does regularization improve out-of-sample prediction without destroying interpretability?
2. How should classification models be compared when class balance and ranking quality both matter?
3. How can flexible prediction models be used as nuisance estimators in causal inference?

## Things To Add Later

- Replace synthetic datasets with a real applied microdata or finance dataset.
- Add model calibration curves and threshold-sensitive cost functions.
- Write a short memo explaining the identifying assumptions behind double machine learning.
- Add robustness checks for random seeds, sample size, and nuisance model choice.

## Run

```bash
python linear_model_pipeline.py
python classification_model_selection.py
python double_machine_learning.py
```
