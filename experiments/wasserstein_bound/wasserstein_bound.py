import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import linear_sum_assignment


# ============================================================
# 0. Global settings
# ============================================================
GLOBAL_SEED = 123


# ============================================================
# 1. Matrix utilities
# ============================================================
def symmetrize(A):
    return 0.5 * (A + A.T)


def psd_matrix_sqrt(A, eps=1e-12):
    A = symmetrize(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


# ============================================================
# 2. Gaussian W2 distance
# ============================================================
def gaussian_w2_from_stats(m1, S1, m2, S2, eps=1e-10):
    S1 = symmetrize(S1) + eps * np.eye(S1.shape[0])
    S2 = symmetrize(S2) + eps * np.eye(S2.shape[0])

    diff_term = np.sum((m1 - m2) ** 2)

    S2_sqrt = psd_matrix_sqrt(S2, eps=eps)
    middle = S2_sqrt @ S1 @ S2_sqrt
    middle_sqrt = psd_matrix_sqrt(middle, eps=eps)

    cov_term = np.trace(S1 + S2 - 2.0 * middle_sqrt)
    cov_term = max(cov_term, 0.0)

    return np.sqrt(diff_term + cov_term)


def empirical_mean_cov(samples):
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T, bias=False)

    if cov.ndim == 0:
        cov = np.array([[cov]])

    return mean, cov


def gaussian_w2_from_samples(samples1, samples2, eps=1e-10):
    m1, S1 = empirical_mean_cov(samples1)
    m2, S2 = empirical_mean_cov(samples2)
    return gaussian_w2_from_stats(m1, S1, m2, S2, eps=eps)


def empirical_w2_matching(samples1, samples2):
    n1, d1 = samples1.shape
    n2, d2 = samples2.shape

    if n1 != n2:
        raise ValueError("samples1 and samples2 must have the same number of samples.")
    if d1 != d2:
        raise ValueError("Dimension mismatch.")

    cost = np.sum((samples1[:, None, :] - samples2[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    return np.sqrt(cost[row_ind, col_ind].mean())


# ============================================================
# 3. Synthetic data generators
# ============================================================
def generate_poisson_data(
    sample_size=2000,
    dimension=2,
    theta_loc=0.0,
    theta_scale=1.0,
    bias=1.0,
    variance_list=(1, 2, 3, 4, 5),
    seed=123,
):
    rng = np.random.default_rng(seed)

    cov_x = np.zeros(dimension)
    for i in range(dimension):
        cov_x[i] = 1.0 / (rng.choice(variance_list) * sample_size)

    X = rng.multivariate_normal(
        mean=np.zeros(dimension),
        cov=np.diag(cov_x),
        size=sample_size,
    )

    theta_star = rng.normal(
        loc=theta_loc,
        scale=theta_scale,
        size=dimension,
    )

    eta = X @ theta_star + bias
    mu = np.exp(eta)
    y = rng.poisson(mu)

    return X, y.astype(float), theta_star


def generate_negative_binomial_data(
    sample_size=2000,
    dimension=2,
    theta_loc=0.5,
    theta_scale=1.0,
    bias=1.0,
    r_dispersion=5.0,
    variance_list=(1, 2, 3, 4, 5),
    seed=321,
):
    """
    Negative-binomial data with mean

        mu = exp(x^T theta + bias)

    and variance

        Var(Y | X) = mu + mu^2 / r.

    We fit this data using Poisson loss, so this is the
    misspecified synthetic setting.
    """
    rng = np.random.default_rng(seed)

    cov_x = np.zeros(dimension)
    for i in range(dimension):
        cov_x[i] = 1.0 / (rng.choice(variance_list) * sample_size)

    X = rng.multivariate_normal(
        mean=np.zeros(dimension),
        cov=np.diag(cov_x),
        size=sample_size,
    )

    theta_star = rng.normal(
        loc=theta_loc,
        scale=theta_scale,
        size=dimension,
    )

    eta = X @ theta_star + bias
    mu = np.exp(eta)

    # Gamma-Poisson mixture representation
    shape = r_dispersion
    scale = mu / r_dispersion
    lam = rng.gamma(shape=shape, scale=scale)
    y = rng.poisson(lam)

    return X, y.astype(float), theta_star


# ============================================================
# 4. Poisson regression model
# ============================================================
class PoissonRegressionModel:
    name = "poisson"

    @staticmethod
    def _clip_eta(eta, clip=30.0):
        return np.clip(eta, -clip, clip)

    @staticmethod
    def loss_full(theta, X, y):
        eta = PoissonRegressionModel._clip_eta(X @ theta)
        mu = np.exp(eta)

        # Ignore constant log(y!)
        return np.mean(mu - y * eta)

    @staticmethod
    def grad_full(theta, X, y):
        eta = PoissonRegressionModel._clip_eta(X @ theta)
        mu = np.exp(eta)
        return X.T @ (mu - y) / X.shape[0]

    @staticmethod
    def hess_full(theta, X, y):
        eta = PoissonRegressionModel._clip_eta(X @ theta)
        mu = np.exp(eta)
        return (X.T * mu) @ X / X.shape[0]

    @staticmethod
    def grad_i(theta, x_i, y_i):
        eta = np.clip(x_i @ theta, -30.0, 30.0)
        mu = np.exp(eta)
        return (mu - y_i) * x_i

    @staticmethod
    def hess_i(theta, x_i, y_i):
        eta = np.clip(x_i @ theta, -30.0, 30.0)
        mu = np.exp(eta)
        return mu * np.outer(x_i, x_i)

    @staticmethod
    def fit_theta_hat(X, y, theta0=None):
        N, d = X.shape

        if theta0 is None:
            theta0 = np.zeros(d)

        res = optimize.minimize(
            fun=lambda th: PoissonRegressionModel.loss_full(th, X, y),
            x0=theta0,
            jac=lambda th: PoissonRegressionModel.grad_full(th, X, y),
            method="BFGS",
            options={"maxiter": 2000},
        )

        if not res.success:
            print("Warning: optimization did not fully converge.")
            print(res.message)

        return res.x


# ============================================================
# 5. Quadratic surrogate around theta_hat
# ============================================================
def precompute_surrogate_terms(model, X, y, theta_hat):
    """
    Precompute g_i(theta_hat) and H_i(theta_hat).

    The quadratic surrogate gradient is

        grad ell_i^quad(theta)
        = g_i(theta_hat) + H_i(theta_hat)(theta - theta_hat).
    """
    N, d = X.shape

    g_list = np.zeros((N, d))
    H_list = np.zeros((N, d, d))

    for i in range(N):
        g_list[i] = model.grad_i(theta_hat, X[i], y[i])
        H_list[i] = model.hess_i(theta_hat, X[i], y[i])

    return g_list, H_list


# ============================================================
# 6. SGD paths
# ============================================================
def sgd_path_original(
    model,
    X,
    y,
    theta0,
    batch_size,
    lr,
    n_iters,
    seed=123,
):
    rng = np.random.default_rng(seed)

    N, d = X.shape
    theta = theta0.copy()

    path = np.zeros((n_iters + 1, d))
    path[0] = theta.copy()

    for t in range(n_iters):
        idx = rng.choice(N, size=batch_size, replace=False)

        grad_batch = np.mean(
            [model.grad_i(theta, X[i], y[i]) for i in idx],
            axis=0,
        )

        theta = theta - lr * grad_batch
        path[t + 1] = theta.copy()

    return path


def sgd_path_surrogate(
    X,
    theta0,
    theta_hat,
    g_list,
    H_list,
    batch_size,
    lr,
    n_iters,
    seed=123,
):
    rng = np.random.default_rng(seed)

    N, d = X.shape
    theta = theta0.copy()

    path = np.zeros((n_iters + 1, d))
    path[0] = theta.copy()

    for t in range(n_iters):
        idx = rng.choice(N, size=batch_size, replace=False)
        delta = theta - theta_hat

        grad_batch = np.mean(
            [g_list[i] + H_list[i] @ delta for i in idx],
            axis=0,
        )

        theta = theta - lr * grad_batch
        path[t + 1] = theta.copy()

    return path


# ============================================================
# 7. Stationary samples
# ============================================================
def get_stationary_samples(path, burnin_frac=0.5, thin=10, max_keep=None):
    n = path.shape[0]
    start = int(np.floor(burnin_frac * n))

    samples = path[start::thin]

    if max_keep is not None and samples.shape[0] > max_keep:
        idx = np.linspace(0, samples.shape[0] - 1, max_keep).astype(int)
        samples = samples[idx]

    return samples


# ============================================================
# 8. One configuration
# ============================================================
def run_one_configuration(
    model,
    X,
    y,
    theta_hat,
    g_list,
    H_list,
    batch_size,
    lr,
    n_iters,
    burnin_frac=0.5,
    thin=10,
    max_keep=500,
    seed=123,
    w2_mode="gaussian",
):
    path_true = sgd_path_original(
        model=model,
        X=X,
        y=y,
        theta0=theta_hat,
        batch_size=batch_size,
        lr=lr,
        n_iters=n_iters,
        seed=seed,
    )

    path_surr = sgd_path_surrogate(
        X=X,
        theta0=theta_hat,
        theta_hat=theta_hat,
        g_list=g_list,
        H_list=H_list,
        batch_size=batch_size,
        lr=lr,
        n_iters=n_iters,
        seed=seed + 100000,
    )

    samples_true = get_stationary_samples(
        path_true,
        burnin_frac=burnin_frac,
        thin=thin,
        max_keep=max_keep,
    )

    samples_surr = get_stationary_samples(
        path_surr,
        burnin_frac=burnin_frac,
        thin=thin,
        max_keep=max_keep,
    )

    if w2_mode == "gaussian":
        w2 = gaussian_w2_from_samples(samples_true, samples_surr)
    elif w2_mode == "matching":
        m = min(len(samples_true), len(samples_surr))
        w2 = empirical_w2_matching(samples_true[:m], samples_surr[:m])
    else:
        raise ValueError("w2_mode must be 'gaussian' or 'matching'.")

    return {
        "w2": w2,
        "samples_true": samples_true,
        "samples_surr": samples_surr,
        "path_true": path_true,
        "path_surr": path_surr,
    }


# ============================================================
# 9. Scaling experiment
# ============================================================
def run_scaling_experiment(
    model,
    X,
    y,
    dataset_name,
    batch_sizes=(16, 32, 256),
    ratio_grid=None,
    n_epochs=300,
    burnin_frac=0.5,
    thin=10,
    max_keep=500,
    n_reps=5,
    fit_seed=123,
    w2_mode="gaussian",
):
    """
    ratio_grid contains values of lambda / B.

    For each ratio,

        lr = ratio * B.
    """
    N, d = X.shape

    if ratio_grid is None:
        ratio_grid = [
            0.25 / N,
            0.50 / N,
            1.00 / N,
            1.50 / N,
            2.00 / N,
        ]

    theta0 = np.zeros(d)

    theta_hat = model.fit_theta_hat(
        X=X,
        y=y,
        theta0=theta0,
    )

    g_list, H_list = precompute_surrogate_terms(
        model=model,
        X=X,
        y=y,
        theta_hat=theta_hat,
    )

    all_rows = []

    for B in batch_sizes:
        n_iters = int(np.ceil(n_epochs * N / B))

        for ratio in ratio_grid:
            lr = ratio * B
            w2_list = []

            for rep in range(n_reps):
                out = run_one_configuration(
                    model=model,
                    X=X,
                    y=y,
                    theta_hat=theta_hat,
                    g_list=g_list,
                    H_list=H_list,
                    batch_size=B,
                    lr=lr,
                    n_iters=n_iters,
                    burnin_frac=burnin_frac,
                    thin=thin,
                    max_keep=max_keep,
                    seed=fit_seed + 1000 * rep + 17 * B,
                    w2_mode=w2_mode,
                )

                w2_list.append(out["w2"])

            row = {
                "dataset": dataset_name,
                "model": model.name,
                "N": N,
                "d": d,
                "B": B,
                "lambda_over_B": ratio,
                "lr": lr,
                "n_iters": n_iters,
                "w2_mean": float(np.mean(w2_list)),
                "w2_std": float(np.std(w2_list, ddof=1)) if len(w2_list) > 1 else 0.0,
                "w2_q025": float(np.quantile(w2_list, 0.025)),
                "w2_q975": float(np.quantile(w2_list, 0.975)),
            }

            all_rows.append(row)

            print(
                f"[{dataset_name}] "
                f"B={B:>3d}, "
                f"lambda/B={ratio:.6e}, "
                f"lr={lr:.6e}, "
                f"W2={row['w2_mean']:.6e} ± {row['w2_std']:.6e}"
            )

    results = pd.DataFrame(all_rows)

    return results, theta_hat


# ============================================================
# 10. Plug-in SGD upper bound constant
# ============================================================
def compute_poisson_sgd_upper_bound_constant(X, y, theta_hat, clip=30.0):
    """
    Plug-in proxy for the SGD bound

        W2(pi_theta, pi_psi) <= A * (lambda / B)

    using the Poisson model structure.
    """
    N, d = X.shape

    eta = np.clip(X @ theta_hat, -clip, clip)
    mu_vec = np.exp(eta)

    # empirical Hessian at theta_hat
    H_hat = (X.T * mu_vec) @ X / N
    H_hat = 0.5 * (H_hat + H_hat.T)

    eigvals = np.linalg.eigvalsh(H_hat)
    mu = float(np.min(eigvals))
    L = float(np.max(eigvals))
    hat_mu = mu

    # numerical safeguard
    eps = 1e-12
    mu = max(mu, eps)
    L = max(L, eps)
    hat_mu = max(hat_mu, eps)

    # third-derivative proxy for Poisson
    x_norm = np.linalg.norm(X, axis=1)
    M_i = mu_vec * (x_norm ** 3)
    M_bar = float(np.mean(M_i))
    M2_bar = float(np.mean(M_i ** 2))

    # fourth moment of sample gradient noise at theta_hat
    g = ((mu_vec - y)[:, None]) * X
    g_bar = np.mean(g, axis=0)
    tau4_4 = float(np.mean(np.linalg.norm(g - g_bar, axis=1) ** 4))

    C0 = (2.0 / mu) * (
        M2_bar / (8.0 * L)
        + (M_bar ** 2) / (4.0 * mu)
    )

    A0 = 96.0 * C0 * tau4_4 / (hat_mu ** 2)
    A = float(np.sqrt(max(A0, 0.0)))

    return {
        "A": A,
        "A0": A0,
        "C0": C0,
        "mu": mu,
        "L": L,
        "hat_mu": hat_mu,
        "M_bar": M_bar,
        "M2_bar": M2_bar,
        "tau4_4": tau4_4,
    }


def add_sgd_upper_bound_column(results_df, A):
    out = results_df.copy()
    out["upper_bound"] = A * out["lambda_over_B"]
    return out


# ============================================================
# 11. Side-by-side plot
# ============================================================
def plot_two_w2_with_upper_bounds(
    results_left,
    A_left,
    title_left,
    results_right,
    A_right,
    title_right,
    save_path=None,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
    )

    # ----------------------------
    # Left panel
    # ----------------------------
    ax = axes[0]

    for B in sorted(results_left["B"].unique()):
        sub = results_left[results_left["B"] == B].sort_values("lambda_over_B")

        ax.plot(
            sub["lambda_over_B"].to_numpy(),
            sub["w2_mean"].to_numpy(),
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=f"Empirical Wasserstein-2 B={B}",
        )

    x_all_left = np.sort(results_left["lambda_over_B"].unique())

    ax.plot(
        x_all_left,
        A_left * x_all_left,
        linestyle="--",
        linewidth=2,
        color="red",
        label="Upper bound",
    )

    ax.set_xlabel(r"$\lambda/B$")
    ax.set_ylabel(r"$W_2(\pi_\theta,\pi_\psi)$")
    ax.set_title(title_left)
    ax.grid(alpha=0.3)
    ax.legend()

    # ----------------------------
    # Right panel
    # ----------------------------
    ax = axes[1]

    for B in sorted(results_right["B"].unique()):
        sub = results_right[results_right["B"] == B].sort_values("lambda_over_B")

        ax.plot(
            sub["lambda_over_B"].to_numpy(),
            sub["w2_mean"].to_numpy(),
            marker="o",
            linewidth=1.5,
            markersize=5,
            label=f"Empirical Wasserstein-2 B={B}",
        )

    x_all_right = np.sort(results_right["lambda_over_B"].unique())

    ax.plot(
        x_all_right,
        A_right * x_all_right,
        linestyle="--",
        linewidth=2,
        color="red",
        label="Upper bound",
    )

    ax.set_xlabel(r"$\lambda/B$")
    ax.set_title(title_right)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    # --------------------------------------------------------
    # Common settings
    # --------------------------------------------------------
    sample_size = 2000
    dimension = 2

    batch_sizes = [16, 32, 256]
    n_reps = 5
    n_epochs = 300

    burnin_frac = 0.5
    thin = 10
    max_keep = 500

    model_poisson = PoissonRegressionModel()

    ratio_grid = [
        0.25 / sample_size,
        0.50 / sample_size,
        1.00 / sample_size,
        1.50 / sample_size,
        2.00 / sample_size,
    ]

    # ========================================================
    # 1) Synthetic Poisson data, fit with Poisson loss
    # ========================================================
    X_syn_pois, y_syn_pois, theta_star_pois = generate_poisson_data(
        sample_size=sample_size,
        dimension=dimension,
        theta_loc=0.0,
        theta_scale=1.0,
        bias=1.0,
        seed=123,
    )

    results_syn_pois, theta_hat_syn_pois = run_scaling_experiment(
        model=model_poisson,
        X=X_syn_pois,
        y=y_syn_pois,
        dataset_name="synthetic_poisson",
        batch_sizes=batch_sizes,
        ratio_grid=ratio_grid,
        n_epochs=n_epochs,
        burnin_frac=burnin_frac,
        thin=thin,
        max_keep=max_keep,
        n_reps=n_reps,
        fit_seed=123,
        w2_mode="gaussian",
    )

    bound_info_syn_pois = compute_poisson_sgd_upper_bound_constant(
        X_syn_pois,
        y_syn_pois,
        theta_hat_syn_pois,
    )

    results_syn_pois_with_bd = add_sgd_upper_bound_column(
        results_syn_pois,
        bound_info_syn_pois["A"],
    )

    print("\nSynthetic Poisson plug-in bound info:")
    for k, v in bound_info_syn_pois.items():
        print(f"{k}: {v}")

    print("\nSynthetic Poisson results with upper bound:")
    print(
        results_syn_pois_with_bd[
            ["B", "lambda_over_B", "w2_mean", "upper_bound"]
        ]
    )

    # ========================================================
    # 2) Synthetic negative-binomial data, fit with Poisson loss
    # ========================================================
    X_syn_nb, y_syn_nb, theta_star_nb = generate_negative_binomial_data(
        sample_size=sample_size,
        dimension=dimension,
        theta_loc=0.5,
        theta_scale=1.0,
        bias=1.0,
        r_dispersion=5.0,
        seed=321,
    )

    results_syn_nb, theta_hat_syn_nb = run_scaling_experiment(
        model=model_poisson,
        X=X_syn_nb,
        y=y_syn_nb,
        dataset_name="synthetic_negative_binomial_fit_poisson",
        batch_sizes=batch_sizes,
        ratio_grid=ratio_grid,
        n_epochs=n_epochs,
        burnin_frac=burnin_frac,
        thin=thin,
        max_keep=max_keep,
        n_reps=n_reps,
        fit_seed=321,
        w2_mode="gaussian",
    )

    bound_info_syn_nb = compute_poisson_sgd_upper_bound_constant(
        X_syn_nb,
        y_syn_nb,
        theta_hat_syn_nb,
    )

    results_syn_nb_with_bd = add_sgd_upper_bound_column(
        results_syn_nb,
        bound_info_syn_nb["A"],
    )

    print("\nSynthetic NB plug-in bound info:")
    for k, v in bound_info_syn_nb.items():
        print(f"{k}: {v}")

    print("\nSynthetic NB results with upper bound:")
    print(
        results_syn_nb_with_bd[
            ["B", "lambda_over_B", "w2_mean", "upper_bound"]
        ]
    )

    # ========================================================
    # 3) Save results
    # ========================================================
    results_syn_pois_with_bd.to_csv(
        "w2_scaling_synthetic_poisson_with_bound.csv",
        index=False,
    )

    results_syn_nb_with_bd.to_csv(
        "w2_scaling_synthetic_nb_with_bound.csv",
        index=False,
    )

    all_results = pd.concat(
        [
            results_syn_pois_with_bd,
            results_syn_nb_with_bd,
        ],
        axis=0,
        ignore_index=True,
    )

    all_results.to_csv(
        "w2_scaling_two_synthetic_datasets_with_bound.csv",
        index=False,
    )

    # ========================================================
    # 4) Final side-by-side plot
    # ========================================================
    plot_two_w2_with_upper_bounds(
        results_left=results_syn_pois,
        A_left=bound_info_syn_pois["A"],
        title_left="Synthetic Poisson data",
        results_right=results_syn_nb,
        A_right=bound_info_syn_nb["A"],
        title_right="Synthetic NB data (fit with Poisson loss)",
        save_path="synthetic_w2_upper_bound_side_by_side.pdf",
    )

    print("\nSaved:")
    print("  w2_scaling_synthetic_poisson_with_bound.csv")
    print("  w2_scaling_synthetic_nb_with_bound.csv")
    print("  w2_scaling_two_synthetic_datasets_with_bound.csv")
    print("  synthetic_w2_upper_bound_side_by_side.pdf")