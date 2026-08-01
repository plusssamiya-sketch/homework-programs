"""
计量经济学作业 2：面板数据双重差分 DID

任务：
    模拟城市政策评估数据，估计处理组在政策后的平均处理效应。
    模型包括个体固定效应和时间固定效应，并使用城市层面聚类稳健标准误。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


RANDOM_STATE = 202


def simulate_panel(n_cities: int = 60, n_periods: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    cities = np.arange(n_cities)
    periods = np.arange(n_periods)
    treated_cities = set(rng.choice(cities, size=n_cities // 2, replace=False))

    rows = []
    for city in cities:
        city_effect = rng.normal(0, 0.7)
        treated = int(city in treated_cities)
        for period in periods:
            post = int(period >= 4)
            time_effect = 0.08 * period + 0.05 * np.sin(period)
            policy_effect = 0.35 * treated * post
            unemployment = rng.normal(5.5 - 0.08 * period - 0.25 * treated, 0.4)
            error = rng.normal(0, 0.35)
            outcome = 2.0 + city_effect + time_effect + policy_effect - 0.06 * unemployment + error
            rows.append(
                {
                    "city": city,
                    "period": period,
                    "treated": treated,
                    "post": post,
                    "did": treated * post,
                    "unemployment": unemployment,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data = simulate_panel()
    model = smf.ols(
        "outcome ~ did + unemployment + C(city) + C(period)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["city"]})

    did_effect = model.params["did"]
    did_se = model.bse["did"]
    ci_low, ci_high = model.conf_int().loc["did"]

    print("\nDifference-in-differences estimate")
    print(f"ATT estimate={did_effect:.3f}")
    print(f"Cluster-robust SE={did_se:.3f}")
    print(f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]")
    print("\nKey coefficient table")
    print(model.summary().tables[1])

    grouped = data.groupby(["treated", "post"], as_index=False)["outcome"].mean()
    print("\nTwo-by-two group means")
    print(grouped.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
