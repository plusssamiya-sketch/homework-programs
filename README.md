# Doctoral Study Portfolio: Machine Learning, Deep Learning, and Econometrics

This repository organizes doctoral-level study assignments for three quantitative courses:

- `Machine Learning`
- `Deep Learning`
- `计量经济学`

Each folder includes course-level notes and self-contained Python programs. The assignments emphasize reproducible experiments, structured modeling workflows, interpretable results, and research-oriented extensions.

The goal is not only to store code, but also to document how a graduate student studies methods: define a question, state assumptions, implement an estimator, diagnose the result, and explain what can be improved.

## Repository Structure

```text
.
|-- Machine Learning
|   |-- README.md
|   |-- linear_model_pipeline.py
|   |-- classification_model_selection.py
|   `-- double_machine_learning.py
|-- Deep Learning
|   |-- README.md
|   |-- mlp_tabular_classification.py
|   |-- cnn_image_classification.py
|   `-- transformer_sequence_model.py
|-- 计量经济学
|   |-- README.md
|   |-- ols_diagnostics.py
|   |-- panel_did_analysis.py
|   `-- iv_gmm_simulation.py
|-- docs
|   |-- research_training_plan.md
|   `-- reproducibility_checklist.md
`-- requirements.txt
```

## Study Standard

Every assignment is organized around four questions:

1. What is the statistical or computational problem?
2. What assumptions make the estimator meaningful?
3. How is the method implemented and validated?
4. What diagnostics reveal failure modes or limitations?

## Course Coverage

### Machine Learning

The machine learning assignments cover supervised learning workflows:

- data preprocessing with leakage-safe pipelines;
- regression and classification model selection;
- cross-validation and hyperparameter tuning;
- model evaluation through RMSE, MAE, R-squared, ROC-AUC, accuracy, and confusion matrices.
- causal prediction with double machine learning and cross-fitting.

### Deep Learning

The deep learning assignments cover PyTorch training templates:

- dataset and dataloader construction;
- MLP and CNN model definitions;
- optimization with AdamW;
- validation tracking, early stopping, dropout, and weight decay.
- transformer-based sequence modeling and attention-mask handling.

### 计量经济学

计量经济学作业覆盖常见实证研究方法：

- OLS 回归估计与稳健标准误；
- 异方差检验与多重共线性诊断；
- 面板数据双重差分 DID；
- 固定效应与聚类稳健标准误；
- 工具变量、弱工具变量诊断与 GMM 矩条件。

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
python "Machine Learning\double_machine_learning.py"
python "Deep Learning\mlp_tabular_classification.py"
python "Deep Learning\cnn_image_classification.py"
python "Deep Learning\transformer_sequence_model.py"
python "计量经济学\ols_diagnostics.py"
python "计量经济学\panel_did_analysis.py"
python "计量经济学\iv_gmm_simulation.py"
```

## Notes

All programs use synthetic or built-in datasets so the repository can run without private course data. For a real homework submission, replace the data-generation functions with course datasets while keeping the same modeling and evaluation structure.

See `docs/research_training_plan.md` for a suggested semester workflow and `docs/reproducibility_checklist.md` for submission standards.
