"""Unified SWOT Rating Curve calibration script.

Supports three equation types:
  - classic:    Q = alpha * (H - z0)^beta
  - lowfroude:  Q = k * (A0 + dA)^(5/3) * W^(-2/3) * S^(1/2)   
  - sfd:        Q = alpha * (H - z0)^beta * S^delta   

Usage:
  python swot_rc.py --equation classic  data_files... -o output_dir
  python swot_rc.py --equation sfd --score --score_csv_file results.csv data_files...
"""

import argparse
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import xarray as xr

import pymc as pm
import arviz as az

try:
    from H2iVDI.core.geometry import curve_fit_lin2
    _HAS_H2IVDI = True
except ImportError:
    _HAS_H2IVDI = False

EQUATION_CHOICES = ["classic", "lowfroude", "sfd"]
MIN_VALID = {"classic": 4, "lowfroude": 5, "sfd": 4}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_observations_netcdf(rootdir, reach_id, folder="rc_hydroDL"):
    try:
        obs_file = os.path.join(rootdir, folder, "%i_SWOT.nc" % reach_id)
        dataset = xr.open_dataset(obs_file, group="reach", decode_times=False)
        Q = dataset.Q_hydroDL.values
        H = dataset.wse.values
        W = dataset.width.values
        S = dataset.slope2.values
        return H, W, S, Q
    except:
        pass

    try:
        obs_file = os.path.join(rootdir, folder, "%i_SWOT.nc" % reach_id)
        dataset = xr.open_dataset(obs_file, group="reach", decode_times=False)
        Q = dataset.Q_hydroDL.values
        H = dataset.wse.values
        W = dataset.width.values
        S = dataset.slope2.values
        return H, W, S, Q
    except:
        pass

    # if folder == 'Garonne':
    try:
        obs_file = os.path.join(rootdir, folder, "%i_DF1D.nc" % reach_id)
        dataset = xr.open_dataset(obs_file, group='reach', decode_times=False)
        Q_dassflow = dataset.Q.values
        H_dassflow = dataset.wse.values
        W_dassflow = dataset.width.values
        S_dassflow = dataset.slope2.values
        return H_dassflow, W_dassflow, S_dassflow, Q_dassflow
    except:
        print(f"Error loading observations for reach {reach_id}")
        return None, None, None, None

    # Aggregrated data
    if 'H' not in locals() or H is None:
        try:
            Q = dataset.Q_sos.values
            swot_obs_file = os.path.join(rootdir, "rc_hydroDL", "%i_SWOT.nc" % reach_id)
            swot_dataset = xr.open_dataset(swot_obs_file, group="reach", decode_times=False)
            H = swot_dataset.wse.values
            W = swot_dataset.width.values
            S = swot_dataset.slope2.values
            if len(Q) != len(H):
                print(f"Error: Data mismatch between Q hydro_sos ({len(Q)}) and H ({len(H)})")
                return None, None, None, None
            else:
                return H, W, S, Q
        except:
            print(f"Error loading observations for reach {reach_id}")
            return None, None, None, None

def load_observations_csv(reach_id, dataset):
    reach_data = dataset[dataset['reach_id_v16'] == reach_id]
    reach_data = reach_data.dropna(subset=['wse', 'width', 'slope', 'slope2', 'gage_flow_m3s'])
    H = reach_data['wse'].values
    W = reach_data['width'].values
    S = reach_data['slope'].values
    S2 = reach_data['slope2'].values
    Q = reach_data['gage_flow_m3s'].values
    return H, W, S, Q



def filter_section(H, W, plot=False):
    """Smooth width profile using piecewise linear fit (requires H2iVDI)."""
    if not _HAS_H2IVDI:
        raise ImportError("H2iVDI is required for lowfroude equation type")

    if len(H) < 5:
        print(f"[WARNING] Only {len(H)} data points, skipping curve fitting")
        return H, W

    isort = np.argsort(H)
    Hs = H[isort]
    Ws = W[isort]

    try:
        Hi, Wi, _ = curve_fit_lin2(Hs, Ws)
    except Exception as e:
        print(f"[WARNING] Curve fitting failed: {e}")
        return H, W

    if plot:
        plt.plot(Hs, Ws, "b-")
        plt.plot(Hs, Ws, "b.")
        plt.plot(Hi, Wi, "r--")
        plt.plot(Hi, Wi, "r-+")
        plt.show()

    Wsi = np.interp(Hs, Hi, Wi)
    Wf = np.zeros(W.size)
    Wf[isort] = Wsi
    return H, Wf


# ═══════════════════════════════════════════════════════════════════════════════
#  Score Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_nash_sutcliffe(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)


def calculate_r2(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    correlation_matrix = np.corrcoef(observed, simulated)
    return correlation_matrix[0, 1] ** 2


def rmse(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    return np.sqrt(np.mean((observed - simulated) ** 2))


def mae(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    return np.mean(np.abs(observed - simulated))


def bias(observed, simulated):
    observed = np.array(observed)
    simulated = np.array(simulated)
    return np.mean(observed - simulated)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_results(
    posteriors, Qrc, Hrc, Wrc, Src, z0_hat,
    qm_post, qm_kct_post, qci_post, param_label,
    r2_stat, plot_file=None,
    font_size_label=18, font_size_title=20, font_size_legend=16,
    font_size_tick=15, font_size_suptitle=12, use_default_fontsize=False
):
    """
    Create diagnostic plot for calibration results.

    posteriors: list of (name, data_2d) where data_2d shape is (n_chains, n_draws)
    Font size arguments let you customize label, title, legend, tick, and suptitle font sizes.
    If use_default_fontsize is True, all font sizes revert to matplotlib defaults.
    """
    import matplotlib as mpl

    # If user asks to use default font sizes, override
    if use_default_fontsize:
        font_size_label = mpl.rcParamsDefault.get("axes.labelsize", 12)
        font_size_title = mpl.rcParamsDefault.get("axes.titlesize", 12)
        font_size_legend = mpl.rcParamsDefault.get("legend.fontsize", 12)
        font_size_tick = mpl.rcParamsDefault.get("xtick.labelsize", 10)
        font_size_suptitle = mpl.rcParamsDefault.get("figure.titlesize", 16)

    n_params = len(posteriors)
    color_chains = ["red", "blue", "orange", "green"]
    ls_chains = ["-", "--", "-.", "--"]

    fig = plt.figure(figsize=(9, 9), layout="constrained")

    if n_params <= 2:
        gs = GridSpec(5, 6, figure=fig)
        param_axes = [fig.add_subplot(gs[0, i * 3:(i + 1) * 3]) for i in range(n_params)]
        ax_qi = fig.add_subplot(gs[1:3, 0:3])
        ax_qh = fig.add_subplot(gs[1:3, 3:6])
        ax_si = fig.add_subplot(gs[3:4, 0:3])
        ax_sh = fig.add_subplot(gs[3:4, 3:6])
        ax_wi = fig.add_subplot(gs[4:5, 0:3])
        ax_wh = fig.add_subplot(gs[4:5, 3:6])
    elif n_params == 3:
        gs = GridSpec(5, 6, figure=fig)
        param_axes = [fig.add_subplot(gs[0, i * 2:(i + 1) * 2]) for i in range(3)]
        ax_qi = fig.add_subplot(gs[1:3, 0:3])
        ax_qh = fig.add_subplot(gs[1:3, 3:6])
        ax_si = fig.add_subplot(gs[3:4, 0:3])
        ax_sh = fig.add_subplot(gs[3:4, 3:6])
        ax_wi = fig.add_subplot(gs[4:5, 0:3])
        ax_wh = fig.add_subplot(gs[4:5, 3:6])
    else:
        gs = GridSpec(5, 8, figure=fig)
        param_axes = [fig.add_subplot(gs[0, i * 2:(i + 1) * 2]) for i in range(n_params)]
        ax_qi = fig.add_subplot(gs[1:3, 0:4])
        ax_qh = fig.add_subplot(gs[1:3, 4:8])
        ax_si = fig.add_subplot(gs[3:4, 0:4])
        ax_sh = fig.add_subplot(gs[3:4, 4:8])
        ax_wi = fig.add_subplot(gs[4:5, 0:4])
        ax_wh = fig.add_subplot(gs[4:5, 4:8])

    name_dict = {
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "z0": r"$b$",
        "delta": r"$\delta$",
        "k": r"$k$",
        "A0": r"$A_0$",
    }
    for ax, (name, post_data) in zip(param_axes, posteriors):
        for chain in range(post_data.shape[0]):
            vals = post_data[chain, :]
            if hasattr(vals, 'values'):
                vals = vals.values
            az.plot_dist(vals, ax=ax,
                         plot_kwargs={"lw": 1, "ls": ls_chains[chain],
                                      "color": color_chains[chain]})
        ax.set_xlabel(name_dict[name], fontsize=font_size_label)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')
        ax.tick_params(axis='both', which='major', labelsize=font_size_tick)

    # Q vs index
    ax_qi.fill_between(np.arange(Hrc.size), qci_post[0], qci_post[1],
                        color="gray", alpha=0.7)
    ax_qi.plot(qm_post, "bo", label="Mean Posterior")
    ax_qi.plot(Qrc, "r.", label="Observed")
    ax_qi.plot(qm_kct_post, "g+", label="Inferred Rating Curve") # label=param_label)
    leg = ax_qi.legend(fontsize=font_size_legend)
    ax_qi.set_xlabel("index", fontsize=font_size_label)
    ax_qi.set_ylabel(r"$Q$ ($m^3/s$)", fontsize=font_size_label)
    ax_qi.tick_params(axis='both', which='major', labelsize=font_size_tick)
    for label in ax_qi.get_yticklabels():
        label.set_rotation(60)
        label.set_ha('right')

    # Q vs h
    isort = np.argsort(Hrc)
    ax_qh.fill_between(Hrc[isort] - z0_hat,
                        qci_post[0][isort], qci_post[1][isort],
                        color="gray", alpha=0.7)
    ax_qh.plot(Hrc[isort] - z0_hat, Qrc[isort], "r.")
    ax_qh.plot(Hrc[isort] - z0_hat, qm_post[isort], "b-")
    ax_qh.plot(Hrc[isort] - z0_hat, qm_kct_post[isort], "g+")
    ax_qh.set_xlabel(r"$Z - \hat{b}$ (m)", fontsize=font_size_label)
    ax_qh.set_ylabel(r"$Q$ ($m^3/s$)", fontsize=font_size_label)
    ax_qh.tick_params(axis='both', which='major', labelsize=font_size_tick)
    for label in ax_qh.get_yticklabels():
        label.set_rotation(60)
        label.set_ha('right')

    # S panels
    for ax_s in [ax_si, ax_sh]:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        ax_s.yaxis.set_major_formatter(fmt)
        ax_s.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        ax_s.tick_params(axis='both', which='major', labelsize=font_size_tick)

    ax_si.plot(Src, "c.")
    ax_si.set_xlabel("index", fontsize=font_size_label)
    ax_si.set_ylabel("Slope S", fontsize=font_size_label)

    ax_sh.plot(Hrc[isort] - z0_hat, Src[isort], "c.")
    ax_sh.set_xlabel(r"$Z - \hat{b}$ (m)", fontsize=font_size_label)
    ax_sh.set_ylabel("Slope S", fontsize=font_size_label)

    # W panels
    ax_wi.plot(Wrc, "k.")
    ax_wi.set_xlabel("index", fontsize=font_size_label)
    ax_wi.set_ylabel("Width W", fontsize=font_size_label)
    ax_wi.tick_params(axis='both', which='major', labelsize=font_size_tick)

    ax_wh.plot(Hrc[isort] - z0_hat, Wrc[isort], "k.")
    ax_wh.set_xlabel(r"$Z - \hat{b}$ (m)", fontsize=font_size_label)
    ax_wh.set_ylabel("Width W", fontsize=font_size_label)
    ax_wh.tick_params(axis='both', which='major', labelsize=font_size_tick)

    fig.suptitle(
        "valid count:%i, R2=%.3f" % (Qrc.size, r2_stat),
        fontsize=font_size_suptitle
    )
    plt.tight_layout()
    if plot_file is not None:
        # plt.savefig(plot_file) # use this line for large amount of data, save figure with lower resolution
        plt.savefig(plot_file, dpi=300) # ONLY for small amount of data,save figure with higher resolution
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration (Bayesian MCMC)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(equation_type, Q, H, W, S, plot_file=None, saved_samples_file=None):
    """
    Calibrate a rating curve model using Bayesian inference (MCMC).

    Returns a result dict with calibrated parameters and diagnostics, or None
    if there is insufficient data.
    """
    # ── Preprocessing ─────────────────────────────────────────────────────
    valid = np.logical_and(Q >= 0.1, S > 1e-9)
    valid = np.logical_and(valid, W >= 10.0)
    if np.sum(valid) < 5:
        print("Not enough data: %i %i %i"
              % (np.sum(Q >= 0.1), np.sum(S >= 1e-9), np.sum(W >= 10.0)))
        return None

    isort = np.argsort(H)
    dA = np.zeros(H.size)
    for it in range(1, H.size):
        dH = H[isort[it]] - H[isort[it - 1]]
        Wm = 0.5 * (W[isort[it]] + W[isort[it - 1]])
        dA[isort[it]] = dA[isort[it - 1]] + dH * Wm

    valid = np.logical_and(valid, dA >= 0.0)
    Qrc = Q[valid]
    Hrc = H[valid]
    Wrc = W[valid]
    Src = S[valid]
    dArc = dA[valid]
    n_data = Qrc.size

    # Statistics on S and W
    std_S = np.std(Src)
    std_W = np.std(Wrc)
    rel_std_S = std_S / np.mean(Src)
    rel_std_W = std_W / np.mean(Wrc)
    print(f"Mean of S: {np.mean(Src):.3f}")
    print(f"Mean of W: {np.mean(Wrc):.3f}")
    print(f"Standard deviation of S: {std_S:.3f}")
    print(f"Standard deviation of W: {std_W:.3f}")
    print(f"Relative standard variation of S: {rel_std_S:.3f}")
    print(f"Relative standard variation of W: {rel_std_W:.3f}")

    hmin = np.min(Hrc)

    # ── Build MCMC model ──────────────────────────────────────────────────
    model = pm.Model()
    with model:

        if equation_type == "classic":
            alpha = pm.Uniform('alpha', lower=2.0, upper=150.0)
            beta = pm.Normal('beta', mu=2.0, sigma=1.0)
            z0 = pm.TruncatedNormal('z0', mu=hmin - 7., sigma=10.,
                                     upper=hmin - 0.05)
            mu = alpha * (Hrc - z0) ** beta
            var_names = ["alpha", "beta", "z0", "offset", "factor"]

        elif equation_type == "lowfroude":
            z0 = pm.TruncatedNormal('z0', mu=hmin - 3., sigma=10.,
                                     upper=hmin - 0.05)
            h0 = hmin - z0
            A0_model = h0 * np.min(Wrc)
            # k = pm.Uniform('k', lower=2.0, upper=100.0)
            k = pm.Uniform('k', lower=0.1, upper=70.0)
            mu = (k * (A0_model + dArc) ** (5. / 3.)
                  * Wrc ** (-2. / 3.) * Src ** 0.5)
            var_names = ["z0", "k", "offset", "factor"]

        elif equation_type == "sfd":
            # alpha = pm.Uniform('alpha', lower=10.0, upper=2000.0)
            alpha = pm.Uniform('alpha', lower=2.0, upper=500.0)
            beta = pm.Normal('beta', mu=2.0, sigma=1.0)
            z0 = pm.TruncatedNormal('z0', mu=hmin - 7., sigma=10.,
                                     upper=hmin - 0.05)
            delta = pm.TruncatedNormal('delta', mu=0.5, sigma=0.05,
                                        lower=0.0, upper=1.0)
            mu = alpha * (Hrc - z0) ** beta * Src ** delta
            var_names = ["alpha", "beta", "z0", "delta", "offset", "factor"]

        # Shared uncertainty model
        offset = pm.Normal('offset', mu=0.0, sigma=5.0)
        factor = pm.TruncatedNormal('factor', mu=0.3, sigma=0.2, lower=0.0)
        q_sd = ((0.2 * Qrc) ** 2 + (offset + factor * mu) ** 2) ** 0.5
        qest = pm.Normal('q', mu=mu, sigma=q_sd, observed=Qrc)

        step = pm.Metropolis()
        try:
            trace = pm.sample(10000, step=step, tune=3000,
                              progressbar=True, random_seed=0)
        except:
            print("Hrc=", Hrc)
            print("Qrc=", Qrc)
            print("Wrc=", Wrc)
            print("Src=", Src)
            raise

        summary = az.summary(trace, var_names=var_names)
        print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]])

    # ── Extract diagnostics ───────────────────────────────────────────────
    diagnostics = {}
    for var in var_names:
        diagnostics[f'r_hat_{var}'] = summary['r_hat'][var]
        diagnostics[f'ess_{var}'] = summary['ess_bulk'][var]

    # ── Posterior processing ──────────────────────────────────────────────
    z0_post = trace["posterior"]["z0"]
    offset_post = trace["posterior"]["offset"]
    factor_post = trace["posterior"]["factor"]
    n_chains, n_draws = z0_post.shape

    q_post = np.zeros((n_chains, n_draws, n_data))
    qm_post = np.zeros(n_data)
    qci_post = np.zeros((2, n_data))
    phi_post = np.zeros(n_data)

    if equation_type == "classic":
        alpha_post = trace["posterior"]["alpha"]
        beta_post = trace["posterior"]["beta"]
        beta_m = np.mean(np.ravel(beta_post))
        z0m = float(np.mean(z0_post))

        for it in range(n_data):
            h = Hrc[it] - z0_post.values
            phi_post[it] = np.mean(np.ravel(Hrc[it] - z0m) ** beta_m)
            q_post[:, :, it] = alpha_post * h ** beta_post
            qm_post[it] = np.mean(np.ravel(q_post[:, :, it]))
            qci_post[:, it] = az.hdi(np.ravel(q_post[:, :, it]),
                                      hdi_prob=0.95)

    elif equation_type == "lowfroude":
        k_post = trace["posterior"]["k"]
        A0_post = (hmin - z0_post) * np.min(Wrc)

        for it in range(n_data):
            phi_post[it] = np.mean(np.ravel(
                (A0_post + dArc[it]) ** (5. / 3.)
                * Wrc[it] ** (-2. / 3.) * Src[it] ** 0.5))
            q_post[:, :, it] = (k_post
                                * (A0_post + dArc[it]) ** (5. / 3.)
                                * Wrc[it] ** (-2. / 3.) * Src[it] ** 0.5)
            qm_post[it] = np.mean(np.ravel(q_post[:, :, it]))
            qci_post[:, it] = az.hdi(np.ravel(q_post[:, :, it]),
                                      hdi_prob=0.95)

    elif equation_type == "sfd":
        alpha_post = trace["posterior"]["alpha"]
        beta_post = trace["posterior"]["beta"]
        delta_post = trace["posterior"]["delta"]
        beta_m = np.mean(np.ravel(beta_post))
        delta_m = np.mean(np.ravel(delta_post))
        z0m = float(np.mean(z0_post))

        for it in range(n_data):
            h = Hrc[it] - z0_post.values
            phi_post[it] = np.mean(
                np.ravel(Hrc[it] - z0m) ** beta_m * Src[it] ** delta_m)
            q_post[:, :, it] = (alpha_post * h ** beta_post
                                * Src[it] ** delta_post)
            qm_post[it] = np.mean(np.ravel(q_post[:, :, it]))
            qci_post[:, it] = az.hdi(np.ravel(q_post[:, :, it]),
                                      hdi_prob=0.95)

    # ── Parameter estimates ───────────────────────────────────────────────
    z0_hat = float(np.median(z0_post))
    offset_hat = np.median(np.ravel(offset_post))
    factor_hat = np.median(np.ravel(factor_post))
    h_hat = np.mean(Hrc - z0_hat)

    if equation_type == "classic":
        alpha_hat = np.median(np.ravel(alpha_post))
        alpha_tilde = np.mean(qm_post) / np.mean(phi_post)
        beta_hat = np.median(np.ravel(beta_post))

        qm_kct = np.array([
            alpha_hat * (Hrc[i] - z0_hat) ** beta_hat
            for i in range(n_data)])
        qm_kct_2 = np.array([
            alpha_tilde * (Hrc[i] - z0_hat) ** beta_hat
            for i in range(n_data)])

        param_label = (r"$\alpha=%.2f, \beta=%.2f$"
                       % (alpha_hat, beta_hat))
        posteriors_plot = [("z0", z0_post), ("alpha", alpha_post),
                           ("beta", beta_post)]
        result = {
            "h_hat": h_hat, "z0_hat": z0_hat,
            "alpha_hat": alpha_hat, "alpha_tilde": alpha_tilde,
            "beta_hat": beta_hat,
        }
        print("alpha_hat: %.3f" % alpha_hat)
        print("z0_hat: %.3f" % z0_hat)
        print("beta_hat: %.3f" % beta_hat)

    elif equation_type == "lowfroude":
        k_hat = np.median(np.ravel(k_post))
        k_tilde = np.mean(qm_post) / np.mean(phi_post)
        A0_hat = np.median(np.ravel(A0_post))

        qm_kct = np.array([
            k_hat * (A0_hat + dArc[i]) ** (5. / 3.)
            * Wrc[i] ** (-2. / 3.) * Src[i] ** 0.5
            for i in range(n_data)])
        qm_kct_2 = np.array([
            k_tilde * (A0_hat + dArc[i]) ** (5. / 3.)
            * Wrc[i] ** (-2. / 3.) * Src[i] ** 0.5
            for i in range(n_data)])

        param_label = r"$K=%.2f$" % k_hat
        posteriors_plot = [("z0", z0_post), ("k", k_post)]
        result = {
            "h_hat": h_hat, "z0_hat": z0_hat,
            "k_hat": k_hat, "k_tilde": k_tilde, "A0_hat": A0_hat,
        }
        print("k_hat: %.3f" % k_hat)
        print("k_tilde: %.3f" % k_tilde)
        print("z0_hat: %.3f" % z0_hat)
        print("A0_hat: %.3f" % A0_hat)

    elif equation_type == "sfd":
        alpha_hat = np.median(np.ravel(alpha_post))
        alpha_tilde = np.mean(qm_post) / np.mean(phi_post)
        beta_hat = np.median(np.ravel(beta_post))
        delta_hat = np.median(np.ravel(delta_post))

        qm_kct = np.array([
            alpha_hat * (Hrc[i] - z0_hat) ** beta_hat
            * Src[i] ** delta_hat
            for i in range(n_data)])
        qm_kct_2 = np.array([
            alpha_tilde * (Hrc[i] - z0_hat) ** beta_hat
            * Src[i] ** delta_hat
            for i in range(n_data)])

        param_label = (r"$\alpha=%.2f, \beta=%.2f, \delta=%.2f$"
                       % (alpha_hat, beta_hat, delta_hat))
        posteriors_plot = [("z0", z0_post), ("alpha", alpha_post),
                           ("beta", beta_post), ("delta", delta_post)]
        result = {
            "h_hat": h_hat, "z0_hat": z0_hat,
            "alpha_hat": alpha_hat, "alpha_tilde": alpha_tilde,
            "beta_hat": beta_hat, "delta_hat": delta_hat,
        }
        print("alpha_hat: %.3f" % alpha_hat)
        print("z0_hat: %.3f" % z0_hat)
        print("beta_hat: %.3f" % beta_hat)
        print("delta_hat: %.3f" % delta_hat)

    print("offset_hat: %.3f" % offset_hat)
    print("factor_hat: %.3f" % factor_hat)

    # ── Spearman R2 ──────────────────────────────────────────────────────
    r2 = sps.spearmanr(Qrc, qm_post)
    r2_2 = sps.spearmanr(Qrc, qm_kct_2)

    # ── Save posterior samples ────────────────────────────────────────────
    if saved_samples_file is not None:
        saved_samples = trace['posterior'].isel(
            draw=slice(0, trace['posterior'].draw.size, 10))
        saved_samples.to_netcdf(saved_samples_file)

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_results(posteriors_plot, Qrc, Hrc, Wrc, Src, z0_hat,
                  qm_post, qm_kct, qci_post, param_label,
                  r2.statistic, plot_file)

    # ── Build result dict ─────────────────────────────────────────────────
    result.update({
        "offset_hat": offset_hat,
        "factor_hat": factor_hat,
        "dAbar": np.mean(dArc),
        "Wbar": np.mean(Wrc),
        "Sbar": np.mean(Src),
        "Qbar": np.mean(Qrc),
        "std_S": std_S,
        "std_W": std_W,
        "rel_std_S": rel_std_S,
        "rel_std_W": rel_std_W,
        "R2": r2.statistic,
        "R2_2": r2_2.statistic,
    })
    result.update(diagnostics)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Forward Models (for scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_q(equation_type, row, Hrc, Wrc, Src, dArc, use_tilde=False):
    """Compute Q from calibrated parameters using the specified equation."""
    if equation_type == "classic":
        a = row['alpha_tilde'] if use_tilde else row['alpha_hat']
        return a * (Hrc - row['z0_hat']) ** row['beta_hat']

    elif equation_type == "lowfroude":
        kval = row['k_tilde'] if use_tilde else row['k_hat']
        return (kval * (row['A0_hat'] + dArc) ** (5. / 3.)
                * Wrc ** (-2. / 3.) * Src ** 0.5)

    elif equation_type == "sfd":
        a = row['alpha_tilde'] if use_tilde else row['alpha_hat']
        return (a * (Hrc - row['z0_hat']) ** row['beta_hat']
                * Src ** row['delta_hat'])


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration Loop
# ═══════════════════════════════════════════════════════════════════════════════

def calibration(args, rootdir, equation_type, data_file_format):
    print(f"Running calibration loop ({equation_type} equation)")
    results = []

    if data_file_format == "csv":
        dataset = pd.read_csv(args.data_files[0], sep=",")
        reach_ids = dataset['reach_id_v16'].unique().tolist() 
    elif data_file_format == "netcdf":
        reach_ids = []
        for index, file in enumerate(args.data_files):
            reach_id = int(os.path.basename(file).split("_")[0])
            reach_ids += [reach_id]


    for index, reach_id in enumerate(reach_ids):
        if args.output_dir is not None:
            reach_csv_file = os.path.join(args.output_dir,
                                          "reach_%i.csv" % reach_id)

            if os.path.isfile(reach_csv_file):
                print(f"Load already computed results for reach {reach_id}"
                      f" ({index + 1}/{len(args.data_files)})")
                df_reach = pd.read_csv(reach_csv_file, sep=";")
                data_reach = df_reach.iloc[0].to_dict()
            else:
                print(f"Load observation data for reach {reach_id}"
                      f" ({index + 1}/{len(args.data_files)})")
                folder = os.path.basename(
                    os.path.dirname(args.data_files[0]))

                if data_file_format == "netcdf":
                    H, W, S, Q = load_observations_netcdf(rootdir, reach_id, folder)
                elif data_file_format == "csv":
                    H, W, S, Q = load_observations_csv(reach_id, dataset)

                if Q is None or H is None or W is None or S is None:
                    print(f"[WARNING] Q is None for reach {reach_id}"
                          f" ({index + 1}/{len(args.data_files)})")
                    continue

                valid = np.logical_and(Q >= 0.1, S > 1e-9)
                valid = np.logical_and(valid, W >= 10.0)
                valid = np.logical_and(valid, H > 1e-9)
                if np.sum(valid) < MIN_VALID[equation_type]:
                    print(f"[WARNING] Not enough data for reach {reach_id}"
                          f" ({index + 1}/{len(args.data_files)})")
                    continue
                H = H[valid]
                W = W[valid]
                S = S[valid]
                Q = Q[valid]

                if equation_type == "lowfroude":
                    H, W = filter_section(H, W)

                plot_file = os.path.join(args.output_dir, "plot",
                                         "reach_%i.png" % reach_id)
                saved_samples_file = os.path.join(
                    args.output_dir, "saved_samples",
                    "reach_%i.nc" % reach_id)

                try:
                    res = calibrate(equation_type, Q, H, W, S,
                                    plot_file=plot_file,
                                    saved_samples_file=saved_samples_file)
                except Exception as e:
                    print(f"[WARNING] Error in calibration for reach"
                          f" {reach_id}"
                          f" ({index + 1}/{len(args.data_files)}): {e}")
                    continue

                if res is None:
                    print(f"[WARNING] calibration failed for reach"
                          f" {reach_id}"
                          f" ({index + 1}/{len(args.data_files)})")
                    continue

                data_reach = res
                data_reach["valid"] = Q.size
                df_reach = pd.DataFrame([data_reach])
                df_reach.to_csv(reach_csv_file, sep=";", index=False)

            data_reach["reach_id"] = reach_id
            results.append(data_reach)

    if results:
        data_df = pd.DataFrame(results)
        data_df.to_csv(os.path.join(args.output_dir, "swot_rc.csv"),
                       sep=";", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(args, rootdir, equation_type, data_file_format):
    csv_file = args.score_csv_file
    df = pd.read_csv(csv_file, sep=";")
    folder = os.path.basename(os.path.dirname(args.data_files[0]))

    if data_file_format == "csv":
        dataset = pd.read_csv(args.data_files[0], sep=",")

    for index, row in df.iterrows():
        reach_id = row['reach_id']
        print(f"Compute score for reach {int(reach_id):d}")
        if data_file_format == "netcdf":
            H, W, S, Q = load_observations_netcdf(rootdir, reach_id, folder=folder)
        elif data_file_format == "csv":
            H, W, S, Q = load_observations_csv(reach_id, dataset)

        valid = np.logical_and(Q >= 0.1, S > 1e-9)
        valid = np.logical_and(valid, W >= 10.0)

        isort = np.argsort(H)
        dA = np.zeros(H.size)
        for it in range(1, H.size):
            dH = H[isort[it]] - H[isort[it - 1]]
            Wm = 0.5 * (W[isort[it]] + W[isort[it - 1]])
            dA[isort[it]] = dA[isort[it - 1]] + dH * Wm

        valid = np.logical_and(valid, dA >= 0.0)
        Qrc = Q[valid]
        Hrc = H[valid]
        Wrc = W[valid]
        Src = S[valid]
        dArc = dA[valid]

        q_hat = predict_q(equation_type, row, Hrc, Wrc, Src, dArc,
                          use_tilde=False)
        q_tilde = predict_q(equation_type, row, Hrc, Wrc, Src, dArc,
                            use_tilde=True)

        df.loc[index, 'r2_param_hat'] = calculate_r2(Qrc, q_hat)
        df.loc[index, 'nse_param_hat'] = calculate_nash_sutcliffe(
            Qrc, q_hat)
        df.loc[index, 'rmse_param_hat'] = rmse(Qrc, q_hat)
        df.loc[index, 'nrmse_param_hat'] = rmse(Qrc, q_hat) / np.mean(Qrc)
        df.loc[index, 'mae_param_hat'] = mae(Qrc, q_hat)
        df.loc[index, 'bias_param_hat'] = bias(Qrc, q_hat)
        df.loc[index, 'r2_param_tilde'] = calculate_r2(Qrc, q_tilde)
        df.loc[index, 'nse_param_tilde'] = calculate_nash_sutcliffe(
            Qrc, q_tilde)
        df.loc[index, 'rmse_param_tilde'] = rmse(Qrc, q_tilde)
        df.loc[index, 'nrmse_param_tilde'] = (rmse(Qrc, q_tilde)
                                                / np.mean(Qrc))
        df.loc[index, 'mae_param_tilde'] = mae(Qrc, q_tilde)
        df.loc[index, 'bias_param_tilde'] = bias(Qrc, q_tilde)
        df.loc[index, 'h_hat'] = np.mean(Hrc - row['z0_hat'])
        df.loc[index, 'rel_std_Q'] = np.std(Qrc) / np.mean(Qrc)

    output_name = os.path.basename(os.path.dirname(args.score_csv_file))
    df.to_csv('score_%s_%s.csv' % (equation_type, output_name),
              sep=";", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Rating Curves using Bayesian inference"
                    " and SWOT data")
    parser.add_argument("data_files", type=str, nargs="+",
                        help="Path to the data file(s)")
    parser.add_argument("--equation", dest="equation_type", type=str,
                        choices=EQUATION_CHOICES, required=True,
                        help="Equation type: classic (Q=αh^β), "
                             "lowfroude (Manning), sfd (Q=αh^βS^δ)")
    parser.add_argument("-r", dest="rootdir", type=str, default=None,
                        help="Path to the root data directory")
    parser.add_argument("-o", dest="output_dir", type=str, default=None,
                        help="Path to the output directory")
    parser.add_argument("--score", dest="score_script", action="store_true")
    parser.add_argument("--score_csv_file", dest="score_csv_file",
                        type=str, default=None)
    args = parser.parse_args()

    if args.equation_type == "lowfroude" and not _HAS_H2IVDI:
        parser.error("H2iVDI package is required for lowfroude equation type")

    if args.rootdir is not None:
        rootdir = args.rootdir
    else:
        rootdir = os.path.dirname(os.path.dirname(args.data_files[0]))

    data_file_format = None
    if args.data_files and os.path.isfile(args.data_files[0]):
        if args.data_files[0].endswith('.csv'):
            data_file_format = "csv"
        elif args.data_files[0].endswith('.nc'):
            data_file_format = "netcdf"
        else:
            data_file_format = "unknown"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "plot"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "saved_samples"),
                    exist_ok=True)

    if not args.score_script:
        calibration(args, rootdir, args.equation_type, data_file_format)
    else:
        compute_score(args, rootdir, args.equation_type, data_file_format)
