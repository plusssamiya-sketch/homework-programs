"""
计量经济学作业 1：OLS 回归与诊断

任务：
    构造收入决定模型，估计普通最小二乘回归，并进行稳健标准误、
    异方差检验、多重共线性检查和残差诊断。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


RANDOM_STATE = 123


def simulate_cross_section(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    education = rng.normal(14, 2.2, n).clip(8, 22)
    experience = rng.normal(10, 5, n).clip(0, 35)
    urban = rng.binomial(1, 0.58, n)
    ability = rng.normal(0, 1, n)
    heteroskedastic_error = rng.normal(0, 0.15 + 0.02 * experience, n)

    log_income = (
        8.1
        + 0.082 * education
        + 0.036 * experience
        - 0.0007 * experience**2
        + 0.115 * urban
        + 0.18 * ability
        + heteroskedastic_error
    )
    return pd.DataFrame(
        {
            "log_income": log_income,
            "education": education,
            "experience": experience,
            "experience_sq": experience**2,
            "urban": urban,
            "ability_proxy": ability + rng.normal(0, 0.4, n),
        }
    )


def compute_vif(design_matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": design_matrix.columns,
            "vif": [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])],
        }
    )


def main() -> None:
    data = simulate_cross_section()
    x = data[["education", "experience", "experience_sq", "urban", "ability_proxy"]]
    x = sm.add_constant(x)
    y = data["log_income"]

    ols_model = sm.OLS(y, x).fit()
    robust_model = ols_model.get_robustcov_results(cov_type="HC3")
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(ols_model.resid, x)

    print("\nOLS estimates")
    print(ols_model.summary().tables[1])
    print("\nHC3 robust standard errors")
    print(robust_model.summary().tables[1])
    print("\nBreusch-Pagan heteroskedasticity test")
    print(f"LM statistic={bp_stat:.3f}, p-value={bp_pvalue:.4f}")
    print("\nVariance inflation factor")
    print(compute_vif(x.drop(columns=["const"])).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nResidual diagnostics")
    print(pd.Series(ols_model.resid, name="residual").describe().to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
