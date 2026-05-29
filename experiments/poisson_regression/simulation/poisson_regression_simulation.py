import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from scipy.stats import norm
import statsmodels.api as sm


# =========================
# 0. Utilities
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def is_finite(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def summarize_ci(values, alpha: float = 0.05):
    """
    Percentile CI (alpha=0.05 -> 95% CI).
    Returns (mean, lo, hi, n).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    lo = np.quantile(v, alpha / 2)
    hi = np.quantile(v, 1 - alpha / 2)
    return float(np.mean(v)), float(lo), float(hi), int(v.size)


def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


# =========================
# 1. Data generation
# =========================

@dataclass
class SimPoissonConfig:
    N: int = 2000
    d_num: int = 40
    theta_loc: float = 0.0
    theta_sigma: float = 0.1
    bias: float = 0.5
    var_choices: Tuple[int, ...] = (1, 2, 3, 4, 5)


def sample_cov_diag_poisson(
    N: int,
    d: int,
    var_choices: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """
    Generate diagonal covariance entries for covariates.

    The scaling cov_x[j] = 1 / (choice * N) makes the design stable
    as the sample size increases.
    """
    choices = np.random.choice(var_choices, size=d)
    cov_diag = 1.0 / (choices * N)
    return cov_diag


def generate_simulated_poisson(
    cfg: SimPoissonConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Generate simulated Poisson regression data.

    Returns
    -------
    X : ndarray, shape (N, d)
        Covariates.
    y : ndarray, shape (N,)
        Poisson responses.
    true_theta : ndarray, shape (d,)
        True regression coefficients.
    bias : float
        Intercept-like bias used in the data-generating process.
    cov_x_diag : ndarray, shape (d,)
        Diagonal entries of the covariate covariance matrix.
    """
    cov_x_diag = sample_cov_diag_poisson(cfg.N, cfg.d_num, cfg.var_choices)
    X = np.random.multivariate_normal(
        mean=np.zeros(cfg.d_num),
        cov=np.diag(cov_x_diag),
        size=cfg.N,
    )

    true_theta = np.random.normal(
        loc=cfg.theta_loc,
        scale=cfg.theta_sigma,
        size=cfg.d_num,
    )

    eta = X @ true_theta + cfg.bias
    mu = np.exp(eta)
    y = np.random.poisson(lam=mu)

    return X, y, true_theta, cfg.bias, cov_x_diag


def generate_poisson_test_set(
    N_test: int,
    cov_x_diag: np.ndarray,
    true_theta: np.ndarray,
    bias: float,
):
    """
    Generate a fixed test set from the same data-generating process.

    This is kept for reproducibility, although the public table below
    only reports calibration and covariance errors.
    """
    d = len(cov_x_diag)
    X = np.random.multivariate_normal(
        mean=np.zeros(d),
        cov=np.diag(cov_x_diag),
        size=N_test,
    )

    eta = X @ true_theta + bias
    mu = np.exp(eta)
    y = np.random.poisson(lam=mu)

    return X, y, mu


# =========================
# 2. MLE + sandwich target
# =========================

def fit_poisson_mle_statsmodels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit the Poisson GLM MLE using statsmodels.
    """
    model = sm.GLM(y, X, family=sm.families.Poisson())
    res = model.fit()
    return res.params


def compute_J_V_at_mle(
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical curvature and gradient-noise matrices at the MLE.

    J = E[ exp(x^T theta_hat) x x^T ]
    V = E[ (y - exp(x^T theta_hat))^2 x x^T ]

    Both are sample averages.
    """
    N, _ = X.shape

    eta = X @ mle
    mu = np.exp(eta)

    J = (X.T * mu) @ X / N

    r2 = (y - mu) ** 2
    V = (X.T * r2) @ X / N

    return J, V


def sandwich_covariance_scaled(
    J: np.ndarray,
    V: np.ndarray,
    N: int,
    eps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sandwich covariance and its N-scaled version.

    sandwich = J^{-1} V J^{-1}
    scaled_sandwich = sandwich / N
    """
    d = J.shape[0]

    if eps > 0:
        J = J + eps * np.eye(d)

    Jinv = np.linalg.inv(J)
    sandwich = Jinv @ V @ Jinv
    scaled_sandwich = sandwich / N

    return sandwich, scaled_sandwich


# =========================
# 3. Preconditioners
# =========================

def precond_CT(Jinv: np.ndarray) -> np.ndarray:
    """
    Continuous-time baseline preconditioner.
    """
    return Jinv


def precond_improved_quadratic_constant_noise(
    J: np.ndarray,
    V: np.ndarray,
    Jinv: np.ndarray,
    N: int,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    DQ+const preconditioner.

    Lambda = (1/N) (V J^{-1} + J^{-1} V) (C + V/N)^{-1},
    with C = J in this implementation.
    """
    d = J.shape[0]

    C = J
    M = C + (1.0 / N) * V + eps * np.eye(d)

    Lambda = (1.0 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(M)

    return Lambda


def compute_taylor_quantities(
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    scaled_sandwich: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute quantities used by the DQ+Exact preconditioner.
    """
    N, d = X.shape

    eta = X @ mle
    mu = np.exp(eta)

    J = (X.T * mu) @ X / N

    errors = y - mu
    Ex = (errors @ X) / N
    E = Ex.reshape(d, 1) @ Ex.reshape(1, d)

    C2 = (X.T * (errors ** 2)) @ X / N - E

    C1 = np.zeros((d, d))
    for i in range(N):
        xxT = X[i].reshape(d, 1) @ X[i].reshape(1, d)
        C1 += (mu[i] ** 2) * (xxT @ scaled_sandwich @ xxT)

    C1 /= N
    C1 -= J @ scaled_sandwich @ J

    P = scaled_sandwich @ J

    return C1, C2, P


def precond_taylor(
    C1: np.ndarray,
    C2: np.ndarray,
    P: np.ndarray,
    J: np.ndarray,
    scaled_sandwich: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    DQ+Exact preconditioner.

    C = (C1 + C2) / B
    Lambda = (P + P^T) (C + J cov J)^{-1}
    """
    C = (C1 + C2) / float(batch_size)
    denom = C + J @ scaled_sandwich @ J

    Lambda = (P + P.T) @ np.linalg.inv(denom)

    return Lambda


# =========================
# 4. SGD sampler
# =========================

def sgd_poisson(
    n_steps: int,
    batch_size: int,
    theta0: np.ndarray,
    lr: float,
    pre_mat: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    fixed_lr: bool = True,
    approximate_loss: bool = False,
) -> np.ndarray:
    """
    Run preconditioned stochastic-gradient updates for Poisson regression.
    """
    N, d = X.shape

    path = np.zeros((n_steps + 1, d))
    path[0] = theta0

    for t in range(n_steps):
        gamma = lr if fixed_lr else lr / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        yb = y[idx]

        theta = path[t]

        if not approximate_loss:
            eta = Xb @ theta
            mu = np.exp(eta)
            grad = ((yb - mu)[:, None] * Xb).mean(axis=0)
        else:
            mu_mle = np.exp(Xb @ mle)
            grad = np.zeros(d)

            for j in range(batch_size):
                xj = Xb[j].reshape(d, 1)
                grad += (
                    mu_mle[j]
                    * (xj @ xj.T)
                    @ (theta - mle)
                ).reshape(-1) / batch_size

        delta = gamma * (pre_mat @ grad)
        path[t + 1] = theta + delta

    return path


# =========================
# 5. Metrics
# =========================

def post_burnin_samples(
    path: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> np.ndarray:
    T = path.shape[0]
    start = int(burnin_frac * T)
    return path[start::thin].copy()


def cov_frob_error(
    samples: np.ndarray,
    cov_target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Relative Frobenius covariance error.
    """
    cov_hat = np.cov(samples.T, bias=True)

    return float(
        np.linalg.norm(cov_hat - cov_target, ord="fro")
        / (np.linalg.norm(cov_target, ord="fro") + eps)
    )


def quantile_calib_rmse(
    samples: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    q_list=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    jitter: float = 1e-10,
) -> float:
    """
    Quantile-calibration RMSE after whitening by the target Gaussian.
    """
    d = cov.shape[0]

    cov_stable = cov + jitter * np.eye(d)
    L = np.linalg.cholesky(cov_stable)

    z = np.linalg.solve(L, (samples - mean).T).T

    q = np.array(q_list)
    zq_target = norm.ppf(q)
    zq_emp = np.quantile(z, q, axis=0)

    diff = zq_emp - zq_target.reshape(-1, 1)
    rmse_dim = np.sqrt(np.mean(diff ** 2, axis=0))

    return float(np.mean(rmse_dim))


def metrics_from_path(
    path: np.ndarray,
    mle: np.ndarray,
    cov_target: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Optional[Dict[str, float]]:
    """
    Compute only the public-facing metrics:
      1. calibration error
      2. covariance error
    """
    samples = post_burnin_samples(path, burnin_frac, thin)

    if samples.shape[0] < 5 or not is_finite(samples):
        return None

    return {
        "quantile_calib_error": quantile_calib_rmse(samples, mle, cov_target),
        "cov_frob_error": cov_frob_error(samples, cov_target),
    }


# =========================
# 6. Repeated runs + CI table
# =========================

METHODS_ORDER = [
    "CT",
    "DQ+const",
    "DQ+Exact",
]

METRICS = [
    "quantile_calib_error",
    "cov_frob_error",
]


def run_with_cis(
    n_reps: int,
    seed0: int,
    batch_sizes: List[int],
    num_epochs: int,
    X: np.ndarray,
    y: np.ndarray,
    initial_state: np.ndarray,
    lr_list_ct: List[float],
    pre_mat_ct: np.ndarray,
    Lambda_improved_by_idx: Dict[int, np.ndarray],
    Lambda_taylor_by_idx: Dict[int, np.ndarray],
    mle: np.ndarray,
    cov_target: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run repeated experiments and summarize mean + 95% CI.
    """
    N = X.shape[0]
    records = []

    for rep in range(n_reps):
        set_seed(seed0 + rep)

        for i, B in enumerate(batch_sizes):
            n_steps = num_epochs * int(N / B)

            # ---- CT
            path_ct = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=lr_list_ct[i],
                pre_mat=pre_mat_ct,
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_ct,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "CT",
                    "batch_size": int(B),
                    **m,
                })

            # ---- DQ+const
            path_im = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=1.0,
                pre_mat=Lambda_improved_by_idx[i],
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_im,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "DQ+const",
                    "batch_size": int(B),
                    **m,
                })

            # ---- DQ+Exact
            path_ta = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=1.0,
                pre_mat=Lambda_taylor_by_idx[i],
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_ta,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "DQ+Exact",
                    "batch_size": int(B),
                    **m,
                })

    df_raw = pd.DataFrame(records)

    rows = []
    for (method, B), sub in df_raw.groupby(["method", "batch_size"]):
        row = {
            "method": method,
            "batch_size": int(B),
        }

        for metric in METRICS:
            mean, lo, hi, n = summarize_ci(sub[metric].values, alpha=0.05)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
            row[f"{metric}_n"] = n

        rows.append(row)

    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}

    df_ci = pd.DataFrame(rows)
    df_ci["method_rank"] = df_ci["method"].map(method_rank)

    df_ci = (
        df_ci
        .sort_values(["batch_size", "method_rank"])
        .drop(columns=["method_rank"])
        .reset_index(drop=True)
    )

    return df_raw, df_ci


def make_table_with_cis(
    df_ci: pd.DataFrame,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Make a clean public-facing table with only calibration and covariance errors.
    """
    metric_display_names = {
        "quantile_calib_error": "calibration_error",
        "cov_frob_error": "covariance_error",
    }

    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}
    df_show = df_ci.copy()
    df_show["method_rank"] = df_show["method"].map(method_rank)

    out = []
    for _, r in df_show.sort_values(["batch_size", "method_rank"]).iterrows():
        row = {
            "batch_size": int(r["batch_size"]),
            "method": r["method"],
        }

        for metric in METRICS:
            row[metric_display_names[metric]] = format_ci(
                r[f"{metric}_mean"],
                r[f"{metric}_lo"],
                r[f"{metric}_hi"],
                digits,
            )

        out.append(row)

    return pd.DataFrame(out)


def make_table_means_only(
    df_ci: pd.DataFrame,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Optional: mean-only version for main-text tables.
    """
    metric_display_names = {
        "quantile_calib_error": "calibration_error",
        "cov_frob_error": "covariance_error",
    }

    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}
    df_show = df_ci.copy()
    df_show["method_rank"] = df_show["method"].map(method_rank)

    out = []
    for _, r in df_show.sort_values(["batch_size", "method_rank"]).iterrows():
        row = {
            "batch_size": int(r["batch_size"]),
            "method": r["method"],
        }

        for metric in METRICS:
            val = r[f"{metric}_mean"]
            row[metric_display_names[metric]] = (
                f"{val:.{digits}f}" if np.isfinite(val) else "NA"
            )

        out.append(row)

    return pd.DataFrame(out)


# =========================
# 7. Plotting
# =========================

def plot_metrics_with_cis(
    df_ci: pd.DataFrame,
    savepath: Optional[str] = None,
):
    """
    Plot calibration error and covariance error with 95% CIs.
    """
    metrics_info = [
        ("quantile_calib_error", "Calibration error"),
        ("cov_frob_error", "Covariance error"),
    ]

    plt.figure(figsize=(11, 4.5))
    plt.rcParams.update({"font.size": 13})

    for k, (metric_name, ylabel) in enumerate(metrics_info, start=1):
        plt.subplot(1, 2, k)

        for method in METHODS_ORDER:
            sub = df_ci[df_ci["method"] == method].sort_values("batch_size")

            x = sub["batch_size"].values.astype(float)
            y = sub[f"{metric_name}_mean"].values.astype(float)
            lo = sub[f"{metric_name}_lo"].values.astype(float)
            hi = sub[f"{metric_name}_hi"].values.astype(float)

            yerr = np.vstack([y - lo, hi - y])

            plt.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=4,
                label=method,
            )

        plt.xscale("log")
        plt.xlabel("Batch size $B$")
        plt.ylabel(ylabel)
        plt.title(ylabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()


def plot_cov_error_with_cis(
    df_ci: pd.DataFrame,
    savepath: Optional[str] = None,
):
    """
    Plot covariance error only.
    """
    metric_name = "cov_frob_error"

    plt.figure(figsize=(6.5, 4.8))
    plt.rcParams.update({"font.size": 13})

    for method in METHODS_ORDER:
        sub = df_ci[df_ci["method"] == method].sort_values("batch_size")

        x = sub["batch_size"].values.astype(float)
        y = sub[f"{metric_name}_mean"].values.astype(float)
        lo = sub[f"{metric_name}_lo"].values.astype(float)
        hi = sub[f"{metric_name}_hi"].values.astype(float)

        yerr = np.vstack([y - lo, hi - y])

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xscale("log")
    plt.xlabel("Batch size $B$")
    plt.ylabel("Covariance error")
    plt.title("Covariance error")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()


# =========================
# 8. Main experiment
# =========================

if __name__ == "__main__":

    # ---- Config
    cfg = SimPoissonConfig(
        N=5000,
        d_num=50,
        theta_sigma=0.1,
        bias=1,
    )

    num_epochs = 500
    batch_sizes = [16, int(0.1 * cfg.N)]
    n_reps = 30
    seed0 = 123
    burnin_frac = 0.1
    thin = 1

    # ---- Generate data
    set_seed(100)

    X, y, true_theta, bias, cov_x_diag = generate_simulated_poisson(cfg)
    N, d = X.shape

    # ---- Fit MLE + sandwich target
    mle = fit_poisson_mle_statsmodels(X, y)

    J, V = compute_J_V_at_mle(X, y, mle)

    _, scaled_sandwich = sandwich_covariance_scaled(
        J,
        V,
        N,
        eps=0.0,
    )

    # ---- Fixed test set
    # Kept for reproducibility, although the public-facing outputs below
    # only report calibration and covariance errors.
    X_test, y_test, mu_true_test = generate_poisson_test_set(
        N_test=2000,
        cov_x_diag=cov_x_diag,
        true_theta=true_theta,
        bias=bias,
    )

    # ---- Preconditioners
    Jinv = np.linalg.inv(J)

    pre_mat_ct = precond_CT(Jinv)

    Lambda_improved_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_improved_by_idx[i] = precond_improved_quadratic_constant_noise(
            J,
            V,
            Jinv,
            N,
        )

    C1, C2, P = compute_taylor_quantities(
        X,
        y,
        mle,
        scaled_sandwich,
    )

    Lambda_taylor_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_taylor_by_idx[i] = precond_taylor(
            C1,
            C2,
            P,
            J,
            scaled_sandwich,
            batch_size=B,
        )

    # ---- Learning rates for CT baseline
    # Rule: lr = 2B/N
    lr_list_ct = [2.0 * B / N for B in batch_sizes]

    # ---- Initial state
    initial_state = mle.copy()

    # ---- Run repeated experiments
    df_raw, df_ci = run_with_cis(
        n_reps=n_reps,
        seed0=seed0,
        batch_sizes=batch_sizes,
        num_epochs=num_epochs,
        X=X,
        y=y,
        initial_state=initial_state,
        lr_list_ct=lr_list_ct,
        pre_mat_ct=pre_mat_ct,
        Lambda_improved_by_idx=Lambda_improved_by_idx,
        Lambda_taylor_by_idx=Lambda_taylor_by_idx,
        mle=mle,
        cov_target=scaled_sandwich,
        burnin_frac=burnin_frac,
        thin=thin,
    )

    # ---- Print public-facing tables
    print("\n=== Mean + 95% CI table ===")
    table_df = make_table_with_cis(df_ci, digits=3)
    print(table_df.to_string(index=False))

    print("\n=== Mean-only table ===")
    table_mean_df = make_table_means_only(df_ci, digits=3)
    print(table_mean_df.to_string(index=False))

    # ---- Save CSV files
    df_raw.to_csv("poisson_raw_results.csv", index=False)
    df_ci.to_csv("poisson_summary_with_cis.csv", index=False)
    table_df.to_csv("poisson_public_table_with_cis.csv", index=False)
    table_mean_df.to_csv("poisson_public_table_means_only.csv", index=False)

    # ---- Plots
    plot_metrics_with_cis(
        df_ci,
        savepath="poisson_calibration_covariance_errors_CIs.pdf",
    )

    plot_cov_error_with_cis(
        df_ci,
        savepath="poisson_covariance_error_CIs.pdf",
    )

if __name__ == "__main__":
    main()
