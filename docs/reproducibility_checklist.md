# Reproducibility Checklist

Use this checklist before treating any assignment as complete.

## Data

- Data source is documented.
- Synthetic data generation uses fixed random seeds.
- Train/test splits are reproducible.
- Data leakage risks are discussed.

## Code

- The script runs from the repository root or from its course folder.
- Dependencies are listed in `requirements.txt`.
- Functions are named according to their statistical role.
- Random seeds are defined in one visible location.
- The output includes enough metrics to evaluate the claim.

## Modeling

- Baseline model is included or clearly discussed.
- Hyperparameters are either justified or selected through validation.
- Assumptions are stated in comments, README text, or a companion memo.
- Diagnostics are reported, not hidden.

## Econometrics-Specific Checks

- Standard errors match the research design.
- Fixed effects are included when the design requires them.
- Instrument relevance is checked before IV interpretation.
- DID results are connected to pre/post and treated/control logic.

## Deep Learning-Specific Checks

- Training and validation metrics are both reported.
- The script can run on CPU.
- Device selection is explicit.
- Regularization choices are visible.

## Submission

- No `__pycache__`, virtual environments, or local outputs are committed.
- README explains what each file is for.
- File names are descriptive and stable.
- The repository can be understood by someone who did not write it.
