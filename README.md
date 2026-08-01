# Homework Programs: Machine Learning, Deep Learning, and Econometrics

This repository organizes professional homework-level programs for three quantitative courses:

- `Machine Learning`
- `Deep Learning`
- `计量经济学`

Each folder includes a clear course-level README and self-contained Python programs. The assignments emphasize reproducible experiments, structured modeling workflows, and interpretable results.

## Repository Structure

```text
.
|-- Machine Learning
|   |-- README.md
|   |-- linear_model_pipeline.py
|   `-- classification_model_selection.py
|-- Deep Learning
|   |-- README.md
|   |-- mlp_tabular_classification.py
|   `-- cnn_image_classification.py
|-- 计量经济学
|   |-- README.md
|   |-- ols_diagnostics.py
|   `-- panel_did_analysis.py
`-- requirements.txt
```

## Course Coverage

### Machine Learning

The machine learning assignments cover supervised learning workflows:

- data preprocessing with leakage-safe pipelines;
- regression and classification model selection;
- cross-validation and hyperparameter tuning;
- model evaluation through RMSE, MAE, R-squared, ROC-AUC, accuracy, and confusion matrices.

### Deep Learning

The deep learning assignments cover PyTorch training templates:

- dataset and dataloader construction;
- MLP and CNN model definitions;
- optimization with AdamW;
- validation tracking, early stopping, dropout, and weight decay.

### 计量经济学

计量经济学作业覆盖常见实证研究方法：

- OLS 回归估计与稳健标准误；
- 异方差检验与多重共线性诊断；
- 面板数据双重差分 DID；
- 固定效应与聚类稳健标准误。

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Examples

```bash
python "Machine Learning\linear_model_pipeline.py"
python "Machine Learning\classification_model_selection.py"
python "Deep Learning\mlp_tabular_classification.py"
python "Deep Learning\cnn_image_classification.py"
python "计量经济学\ols_diagnostics.py"
python "计量经济学\panel_did_analysis.py"
```

## Notes

All programs use synthetic or built-in datasets so the repository can run without private course data. For a real homework submission, replace the data-generation functions with course datasets while keeping the same modeling and evaluation structure.
