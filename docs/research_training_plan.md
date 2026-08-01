# Research Training Plan

This plan turns the repository from a code collection into a graduate study record. It can be used as a semester-long roadmap for doctoral coursework or independent study.

## Core Principle

Each assignment should end with a short research memo:

- research question;
- identification or modeling assumptions;
- estimator and implementation details;
- diagnostics and robustness checks;
- limitations and next steps.

## Twelve-Week Study Map

| Week | Theme | Output |
| --- | --- | --- |
| 1 | Reproducible Python workflows | environment, fixed seeds, project notes |
| 2 | Linear models and regularization | regression pipeline and residual diagnostics |
| 3 | Classification and model selection | ROC-AUC, calibration, error analysis |
| 4 | Causal ML foundations | double machine learning and cross-fitting |
| 5 | Neural network training loops | MLP with early stopping |
| 6 | Computer vision template | CNN with validation tracking |
| 7 | Sequence modeling | transformer encoder assignment |
| 8 | OLS and robust inference | HC3 standard errors, heteroskedasticity tests |
| 9 | Panel data and DID | fixed effects and clustered standard errors |
| 10 | IV and GMM | weak instrument diagnostics and moments |
| 11 | Robustness workshop | sensitivity checks and alternative specifications |
| 12 | Portfolio polish | README, code review, replication checklist |

## What Makes the Work Doctoral-Level

Doctoral coursework should show judgment, not only implementation. A strong submission explains why a method is appropriate, what would make it fail, and how the empirical evidence supports or weakens the conclusion.

For each script, the recommended extension is to add a companion memo with:

- mathematical formulation;
- assumptions;
- pseudo-code;
- interpretation of coefficients or metrics;
- discussion of threats to validity.

## Evaluation Rubric

| Criterion | Excellent Work |
| --- | --- |
| Conceptual clarity | States the research problem and assumptions before coding |
| Technical implementation | Uses modular, reproducible, tested code |
| Diagnostics | Reports both primary metrics and failure-mode checks |
| Interpretation | Connects estimates to substantive meaning |
| Extensibility | Makes it easy to replace synthetic data with real data |
