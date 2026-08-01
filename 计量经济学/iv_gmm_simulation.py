"""
计量经济学作业 3：工具变量与 GMM 矩条件

任务：
    模拟内生解释变量、外生工具变量和结构方程，比较 OLS 与 2SLS，
    并报告弱工具变量诊断和矩条件解释。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


SEED = 909


@dataclass(frozen=True)
class IVReport:
    ols_beta: float
    tsls_beta: float
    first_stage_f: float
    structural_residual_mean_moment: float


def simulate_iv_data(n: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    control = rng.normal(size=n)
    omitted_ability = rng.normal(size=n)
    endog_error = 0.7 * omitted_ability + rng.normal(size=n)
    treatment = 0.9 * z1 + 0.55 * z2 + 0.4 * control + endog_error
    outcome_error = 0.8 * omitted_ability + rng.normal(size=n)
    outcome = 1.6 * treatment + 0.5 * control + outcome_error
    return pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
            "control": control,
            "z1": z1,
            "z2": z2,
        }
    )


def estimate_2sls(data: pd.DataFrame) -> IVReport:
    y = data["outcome"]
    x_ols = sm.add_constant(data[["treatment", "control"]])
    ols = sm.OLS(y, x_ols).fit()

    first_stage_x = sm.add_constant(data[["z1", "z2", "control"]])
    first_stage = sm.OLS(data["treatment"], first_stage_x).fit()
    data = data.copy()
    data["treatment_hat"] = first_stage.fittedvalues

    second_stage_x = sm.add_constant(data[["treatment_hat", "control"]])
    second_stage = sm.OLS(y, second_stage_x).fit()
    f_test = first_stage.f_test("z1 = 0, z2 = 0")

    structural_residual = y - second_stage.params["const"] - second_stage.params["treatment_hat"] * data["treatment"] - second_stage.params["control"] * data["control"]
    instrument_moment = float(np.mean(data["z1"] * structural_residual))

    return IVReport(
        ols_beta=float(ols.params["treatment"]),
        tsls_beta=float(second_stage.params["treatment_hat"]),
        first_stage_f=float(f_test.fvalue),
        structural_residual_mean_moment=instrument_moment,
    )


def main() -> None:
    data = simulate_iv_data()
    report = estimate_2sls(data)

    print("\nInstrumental variables simulation")
    print(f"OLS treatment coefficient: {report.ols_beta:.3f}")
    print(f"2SLS treatment coefficient: {report.tsls_beta:.3f}")
    print(f"First-stage excluded-instrument F statistic: {report.first_stage_f:.3f}")
    print(f"Sample moment E[z1 * structural residual]: {report.structural_residual_mean_moment:.5f}")
    print("\nInterpretation")
    print("OLS is biased upward because treatment is correlated with omitted ability.")
    print("2SLS uses z1 and z2 to isolate exogenous treatment variation.")


if __name__ == "__main__":
    main()
