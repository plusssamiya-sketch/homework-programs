# Coursework Notes: Machine Learning, Deep Learning, and Econometrics

This repository keeps my course assignments and practice scripts for three quantitative courses:

- `Machine Learning`
- `Deep Learning`
- `计量经济学`

Each folder has short notes and Python scripts. I keep the examples self-contained so I can rerun them later without depending on private class data.

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

## How I Use This Repository

For each topic, I try to keep four things clear:

1. What problem the script is solving.
2. What assumptions the method needs.
3. How the result is checked.
4. What I would change if using real data.

## Course Notes

### Machine Learning

Machine learning scripts currently cover:

- data preprocessing with leakage-safe pipelines;
- regression and classification model selection;
- cross-validation and hyperparameter tuning;
- model evaluation through RMSE, MAE, R-squared, ROC-AUC, accuracy, and confusion matrices.
- double machine learning with cross-fitting.

### Deep Learning

Deep learning scripts currently cover:

- dataset and dataloader construction;
- MLP and CNN model definitions;
- optimization with AdamW;
- validation tracking, early stopping, dropout, and weight decay.
- a small transformer sequence classifier.

### 计量经济学

计量经济学部分目前包括：

- OLS 回归估计与稳健标准误；
- 异方差检验与多重共线性诊断；
- 面板数据双重差分 DID；
- 固定效应与聚类稳健标准误；
- 工具变量、弱工具变量诊断与矩条件。

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

See `docs/research_training_plan.md` for my study schedule and `docs/reproducibility_checklist.md` for the checks I want to do before submitting work.
