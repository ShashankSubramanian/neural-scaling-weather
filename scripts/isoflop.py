import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

# Match plot_scaling_model.py / plot_scaling_data.py
plt.style.use("default")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = False

plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["mathtext.default"] = "rm"
alt_legend_fontsize = 5
point_size = 14

FIGSIZE_MULT = (8.0, 2.35)
PNG_DPI = 400

EXTRA_Y_TICKS = [0.03, 0.006]
Y_LOG_SCALE = True
X_AXIS_MIN_PARAMS_M = 0.2

flare_palette = sns.color_palette("flare", 5)
blue_palette = sns.color_palette("mako", n_colors=5)
minima_color = flare_palette[3]

# Matches FlopsCalculator.param_count (lat/lon: 71 + 3 invariants).
C_IN = 74
C_OUT = 71


def param_count_m(embed: int, depth: int, patch_size: int) -> float:
    """Parameter count in millions (same closed form as training code)."""
    e, d, p = embed, depth, patch_size
    p2 = p * p
    patch = e * C_IN * p2 + e
    pos = e * 4 * p2 + e
    mlp = e * 4 * e * 2
    norm = e
    qkv = e * e * 3
    proj = e * e
    block = qkv + proj + mlp + norm * 4
    head = C_OUT * e * p2
    total = patch + pos + block * d + head
    return total * 1e-6


def read_logs_last_val_loss_and_flops(log_path: str) -> tuple[float, float]:
    with open(log_path, "r") as f:
        parts = f.readlines()[-1].split(",")
    return float(parts[3]), float(parts[10]) * 1e12


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 in the space fitted: e.g. loss vs poly(log10 params), or log10 y vs linear log10 x."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def param_m_from_hydra_config(run_dir: str, fallback_m: float) -> tuple[float, bool]:
    cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg):
        return fallback_m, False
    with open(cfg, encoding="utf-8", errors="replace") as f:
        text = f.read()
    me = re.search(r"(?m)^\s*embed_dim:\s*(\d+)\s*$", text)
    md = re.search(r"(?m)^\s*depth:\s*(\d+)\s*$", text)
    mp = re.search(r"(?m)^\s*patch_size:\s*(\d+)\s*$", text)
    if not (me and md and mp):
        return fallback_m, False
    return param_count_m(int(me.group(1)), int(md.group(1)), int(mp.group(1))), True


def load_1p2b_projection_run(
    train_dir: str, config_name: str, run_tag: str, fallback_params_m: float
) -> dict | None:
    """Measured val loss and training FLOPs from logs.txt; params/M from .hydra/config.yaml when present."""
    run_dir = os.path.join(train_dir, config_name, run_tag)
    log_path = os.path.join(run_dir, "logs.txt")
    if not os.path.isfile(log_path):
        print(f"\n(skip 1.2B projection run: missing {log_path})")
        return None
    loss, flops = read_logs_last_val_loss_and_flops(log_path)
    params_m, from_hydra = param_m_from_hydra_config(run_dir, fallback_params_m)
    print(
        f"\n1.2B projection run ({run_tag}): val_loss={loss:.6f} (logs.txt), "
        f"train_flops={flops:.2e}, params={params_m:.4g}M "
        f"({'hydra .hydra/config.yaml' if from_hydra else 'fallback constant'})."
    )
    return {"params_m": params_m, "loss": loss, "flops": flops, "run_dir": run_dir}


config_name = "pe-scaling-adamw-clip1-qknorm"
train_dir = "/pscratch/sd/s/shas1693/results/neuralscaling"

tags = [
    "p4-e96-d4-lr1p5em3",
    "p4-e96-d6-lr1p5em3",
    "p4-e128-d6-lr1p5em3",
    "p4-e192-d6-lr1p5em3",
    "p4-e256-d8-lr1p5em3",
    "p4-e384-d8-lr1p5em3",
    "p4-e512-d8-lr1p2em3",
    "p4-e512-d12-lr1em3",
    "p4-e640-d12-lr8em4",
    "p4-e768-d12-lr6em4",
    "p4-e768-d16-lr5em4",
    "p4-e1024-d12-lr5em4",
    "p4-e1024-d16-lr5em4",
    "p4-e1024-d24-lr5em4",
    "p4-e1536-d16-lr5em4",
]
param_counts = [
    0.7,
    0.9,
    1.5,
    3.1,
    6.9,
    15.1,
    26.4,
    39,
    60.5,
    86.8,
    115.1,
    153.5,
    203.8,
    304.5,
    456.7,
]
cooldown_to_iters_list = [
    [12035, 20059],
    [8868, 14780],
    [5522, 9203],
    [2752, 4587, 13760, 27520],
    [1269, 2115, 6345, 12691, 21151],
    [2992, 5984, 9973, 29919, 59838, 99730],
    [1736, 3472, 5786, 17358, 34716, 57860],
    [2349, 3915, 11744, 23488, 39147, 117440],
    [7640, 15279, 25465, 76395],
    [5364, 10728, 17880, 53639, 107278],
    [8087, 13478, 40435, 80870],
    [6119, 10198, 30595, 61190],
    [7678, 23035, 46071],
    [15417, 30834],
    [10371, 20741],
]

print("Parameter count (M) per tag:")
for _tag, _pm in zip(tags, param_counts):
    print(f"  {_tag}: {_pm:g}M")

# 1.2B projection validation run: loss/FLOPs from logs.txt; width from run .hydra/config.yaml if present.
RUN_TAG_1P2B_PROJECTION = "p4-e1792-d32-lr5em4-cool288000"
PARAM_COUNT_1P2B_FALLBACK_M = 1238
C_TARGET_FLOPS = 2.25e21
C_MODEL_SIZE_REPORT_FLOPS = 3e20

run_tags = []
params_list = []
for tag, cooldown_to_iters, param_count in zip(tags, cooldown_to_iters_list, param_counts):
    for it in cooldown_to_iters:
        run_tag = f"{tag}-cool{it}-base"
        run_tags.append(run_tag)
        params_list.append(param_count)

losses = []
iters = []
flops_list = []
kept_run_tags = []
kept_params_list = []
budgets = [
    6e16,
    1e17,
    3e17,
    6e17,
    1e18,
    3e18,
    6e18,
    1e19,
    3e19,
    6e19,
]
FLOP_BUDGET_MATCH_RTOL = 0.05

for run_tag, param_count in zip(run_tags, params_list):
    results_dir = os.path.join(train_dir, config_name, run_tag)
    log_file = os.path.join(results_dir, "logs.txt")
    if not os.path.isfile(log_file):
        print(log_file)
        continue
    with open(log_file, "r") as f:
        line = f.readlines()[-1].split(",")
        val_loss = float(line[3])
        run_flops = float(line[10]) * 1e12
        it = int(line[0])
    losses.append(val_loss)
    iters.append(it)
    flops_list.append(run_flops)
    kept_run_tags.append(run_tag)
    kept_params_list.append(param_count)

budgets_array = np.array(budgets)
flops_array = np.array(flops_list)
losses_array = np.array(losses)
params_array = np.array(kept_params_list)

nearest_budget_indices = np.argmin(
    np.abs(flops_array[:, None] - budgets_array[None, :]), axis=1
)
nearest_budgets = budgets_array[nearest_budget_indices]
rel_err_budget = np.abs(flops_array - nearest_budgets) / nearest_budgets
budget_match_ok = rel_err_budget <= FLOP_BUDGET_MATCH_RTOL

if (~budget_match_ok).any():
    print(
        f"\nExcluding {int((~budget_match_ok).sum())} runs "
        f"(|actual FLOPs - nearest budget| / budget > {100 * FLOP_BUDGET_MATCH_RTOL:g}%):"
    )
    for t, f, nb, re in zip(
        np.asarray(kept_run_tags, dtype=object)[~budget_match_ok],
        flops_array[~budget_match_ok],
        nearest_budgets[~budget_match_ok],
        rel_err_budget[~budget_match_ok],
    ):
        print(f"  {t}: F={f:.3e}, B={nb:.3e}, rel_err={re:.4f}")

run_tags_kept = np.asarray(kept_run_tags, dtype=object)[budget_match_ok]
losses_kept = losses_array[budget_match_ok]
flops_kept = flops_array[budget_match_ok]

iters_array = np.array(iters)
iters_kept = iters_array[budget_match_ok]
nearest_kept = nearest_budgets[budget_match_ok]
for t, l, it, f, nb in zip(
    run_tags_kept.tolist(),
    losses_kept.tolist(),
    iters_kept.tolist(),
    flops_kept.tolist(),
    nearest_kept,
):
    print(t, l, it, f / 1e6, nb)

proj_run = load_1p2b_projection_run(
    train_dir, config_name, RUN_TAG_1P2B_PROJECTION, PARAM_COUNT_1P2B_FALLBACK_M
)

fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=FIGSIZE_MULT)

df = pd.DataFrame(
    {
        "params": params_array[budget_match_ok],
        "loss": losses_array[budget_match_ok],
        "flops": [
            f"{b:.0e}".replace("e+", "E")
            for b in nearest_budgets[budget_match_ok]
        ],
    }
)

flops_labels_ordered = [f"{b:.0e}".replace("e+", "E") for b in budgets_array]
line_palette = sns.color_palette("rocket", n_colors=len(budgets_array))[::-1]
flops_to_color = dict(zip(flops_labels_ordered, line_palette))
scatter_colors = [flops_to_color[fl] for fl in df["flops"]]

# --- Panel 1: loss vs params (iso-FLOP curves) ---
ax.scatter(df["params"], df["loss"], c=scatter_colors, s=4, alpha=1)

print("\nR^2 of parabola fit: loss vs (log10 params/M)^2, … (linear in log10)")
minima_x = []
minima_y = []
minima_c = []
for label, color, budget in zip(flops_labels_ordered, line_palette, budgets_array):
    sub = df[df["flops"] == label]
    if len(sub) >= 3:
        x = sub["params"].values
        y = sub["loss"].values
        logx = np.log10(x)
        coeffs = np.polyfit(logx, y, deg=2)
        y_hat = np.polyval(coeffs, logx)
        r2_iso = r2_score(y, y_hat)
        logx_fit = np.linspace(logx.min(), logx.max(), 200)
        y_fit = coeffs[0] * logx_fit**2 + coeffs[1] * logx_fit + coeffs[2]
        x_fit = 10 ** logx_fit
        ax.plot(x_fit, y_fit, color=color, linewidth=1.1, label=label)
        print(
            f"FLOPs {label}: loss = a*(log10 params/M)^2 + b*(log10 params/M) + c "
            f"with a={coeffs[0]:.6e}, b={coeffs[1]:.6e}, c={coeffs[2]:.6e}; "
            f"R^2 = {r2_iso:.6f}"
        )
        if coeffs[0] > 0:
            logx_min = -coeffs[1] / (2 * coeffs[0])
            x_min = 10 ** logx_min
            y_min = coeffs[0] * logx_min**2 + coeffs[1] * logx_min + coeffs[2]
            minima_x.append(x_min)
            minima_y.append(y_min)
            minima_c.append(budget)
            print(f"  -> Minimum at params/M={x_min:.4g}, loss={y_min:.6f}")
    elif len(sub) == 2:
        x = sub["params"].values
        y = sub["loss"].values
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=color, linewidth=1.1, label=label)

if minima_x:
    ax.scatter(
        minima_x,
        minima_y,
        marker="D",
        color=minima_color,
        s=4,
        zorder=10,
        label="Optimal",
    )

ax.set_xlabel("Model size (M)")
ax.set_ylabel("Validation loss")
ax.legend(
    title="FLOPs",
    loc="lower left",
    fontsize=alt_legend_fontsize,
    title_fontsize=alt_legend_fontsize,
)
ax.set_xscale("log")
_, xmax_auto = ax.get_xlim()
ax.set_xlim(left=X_AXIS_MIN_PARAMS_M, right=xmax_auto)
ax.set_yscale("log" if Y_LOG_SCALE else "linear")


def _make_y_decimal_formatter(log_scale: bool):
    def _fmt(val, pos):
        if not np.isfinite(val):
            return ""
        if log_scale and val <= 0:
            return ""
        return np.format_float_positional(float(val), unique=True, trim="-")

    return _fmt


ax.yaxis.set_major_formatter(FuncFormatter(_make_y_decimal_formatter(Y_LOG_SCALE)))
if Y_LOG_SCALE:
    ax.yaxis.set_minor_formatter(NullFormatter())

if EXTRA_Y_TICKS:
    ymin, ymax = ax.get_ylim()
    major_loc = ax.yaxis.get_major_locator()
    auto_ticks = np.asarray(major_loc.tick_values(ymin, ymax), dtype=float)
    _auto_ok = np.isfinite(auto_ticks) & (auto_ticks >= ymin) & (auto_ticks <= ymax)
    if Y_LOG_SCALE:
        _auto_ok &= auto_ticks > 0
    auto_ticks = auto_ticks[_auto_ok]
    extra = np.asarray(EXTRA_Y_TICKS, dtype=float)
    _extra_ok = np.isfinite(extra) & (extra >= ymin) & (extra <= ymax)
    if Y_LOG_SCALE:
        _extra_ok &= extra > 0
    extra = extra[_extra_ok]
    parts = []
    if auto_ticks.size:
        parts.append(auto_ticks)
    if extra.size:
        parts.append(extra)
    if parts:
        ax.set_yticks(np.unique(np.sort(np.concatenate(parts))))

ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="x", which="minor", bottom=True)

# --- Panels 2–3: scaling laws on compute-optimal points + projection ---
if len(minima_x) >= 2:
    minima_x_arr = np.array(minima_x) * 1e6
    minima_c_arr = np.array(minima_c)
    minima_y_arr = np.array(minima_y)

    log_c = np.log10(minima_c_arr)
    log_n = np.log10(minima_x_arr)
    coeffs_nc = np.polyfit(log_c, log_n, deg=1)
    log_n_hat = np.polyval(coeffs_nc, log_c)
    r2_n_vs_c = r2_score(log_n, log_n_hat)
    alpha = coeffs_nc[0]
    A = 10 ** coeffs_nc[1]

    print(f"\nScaling law fit: N (params) = A * C^alpha")
    print(f"  A = {A:.4e}, alpha = {alpha:.4f}")
    print(f"  R^2 (log10 N vs log10 C, compute-optimal points) = {r2_n_vs_c:.6f}")

    line_color = blue_palette[2]
    c_target = C_TARGET_FLOPS
    n_target = A * c_target**alpha
    print(
        f"\nProjection to {c_target:.1e} FLOPs: optimal parameter count N = {n_target:.2e}"
    )
    n_report = A * C_MODEL_SIZE_REPORT_FLOPS**alpha
    print(
        f"Projection to {C_MODEL_SIZE_REPORT_FLOPS:.1e} FLOPs: "
        f"optimal parameter count N = {n_report:.2e} ({n_report / 1e6:.2f}M)"
    )

    c_fit = np.logspace(
        np.log10(minima_c_arr.min()) - 0.2,
        np.log10(c_target) + 0.1,
        100,
    )
    n_fit = A * c_fit**alpha
    ax2.plot(
        c_fit,
        n_fit / 1e6,
        color=line_color,
        linewidth=1.3,
        label=rf"$N = {A:.4e} \times C^{{{alpha:.3f}}}$",
    )
    ax2.scatter(
        minima_c_arr,
        minima_x_arr / 1e6,
        marker="D",
        color=minima_color,
        s=point_size,
        zorder=10,
        edgecolor="white",
        linewidth=0.2,
        label="Compute optimal",
    )
    ax2.scatter(
        [c_target],
        [n_target / 1e6],
        marker="D",
        color="black",
        s=point_size,
        zorder=10,
        edgecolor="white",
        linewidth=0.2,
        label=f"Projection: {n_target/1e9:.2f}B",
    )
    if proj_run is not None:
        ax2.scatter(
            [proj_run["flops"]],
            [proj_run["params_m"]],
            marker="*",
            color="green",
            s=35,
            zorder=11,
            edgecolor="white",
            linewidth=0.2,
            label="1.2B run",
        )

    ax2.set_xlabel("FLOPs")
    ax2.set_ylabel("Optimal model size (M)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="best")
    ax2.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20)
    )
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.yaxis.set_major_locator(LogLocator(base=10.0))
    ax2.yaxis.set_major_formatter(
        LogFormatterMathtext(base=10.0, labelOnlyBase=True)
    )
    ax2.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20)
    )
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis="x", which="minor", bottom=True)
    ax2.tick_params(axis="y", which="minor", left=True)

    loss_line_color = flare_palette[2]
    log_loss = np.log10(minima_y_arr)
    coeffs_lc = np.polyfit(log_c, log_loss, deg=1)
    log_loss_hat = np.polyval(coeffs_lc, log_c)
    r2_l_vs_c = r2_score(log_loss, log_loss_hat)
    gamma = coeffs_lc[0]
    E = 10 ** coeffs_lc[1]

    print(f"\nScaling law fit: L (loss) = E * C^gamma")
    print(f"  E = {E:.4e}, gamma = {gamma:.4f}")
    print(f"  R^2 (log10 L vs log10 C, compute-optimal points) = {r2_l_vs_c:.6f}")

    loss_fit = E * c_fit**gamma
    ax3.plot(
        c_fit,
        loss_fit,
        color=loss_line_color,
        linewidth=1.3,
        label=rf"$L = {E:.2e} \times C^{{{gamma:.3f}}}$",
    )
    ax3.scatter(
        minima_c_arr,
        minima_y_arr,
        marker="D",
        color=minima_color,
        s=point_size,
        zorder=10,
        edgecolor="white",
        linewidth=0.2,
        label="Compute optimal",
    )
    loss_target = E * c_target**gamma
    print(f"  Projected loss at {c_target:.1e} FLOPs: {loss_target:.6f}")
    ax3.scatter(
        [c_target],
        [loss_target],
        marker="D",
        facecolors="none",
        edgecolors="black",
        s=point_size,
        zorder=10,
        linewidth=0.2,
        label=f"Projection: L={loss_target:.4f}",
    )
    if proj_run is not None:
        ax3.scatter(
            [proj_run["flops"]],
            [proj_run["loss"]],
            marker="*",
            color="green",
            s=35,
            zorder=11,
            edgecolor="white",
            linewidth=0.2,
            label=f"1.2B run: L={proj_run['loss']:.4f}",
        )
        print(
            f"  Scaling-law L at {c_target:.1e} FLOPs: {loss_target:.6f}; "
            f"measured L (1.2B run @ {proj_run['flops']:.2e} FLOPs): {proj_run['loss']:.6f}"
        )

    ax3.set_xlabel("FLOPs")
    ax3.set_ylabel("Compute-optimal loss")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend(loc="lower left")
    ax3.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20)
    )
    ax3.xaxis.set_minor_formatter(NullFormatter())
    ax3.yaxis.set_major_locator(LogLocator(base=10.0))
    ax3.yaxis.set_major_formatter(
        LogFormatterMathtext(base=10.0, labelOnlyBase=True)
    )
    ax3.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20)
    )
    ax3.yaxis.set_minor_formatter(NullFormatter())
    ax3.tick_params(axis="x", which="minor", bottom=True)
    ax3.tick_params(axis="y", which="minor", left=True)
else:
    print("\nNeed at least two iso-FLOP minima for scaling panels; panels 2–3 left empty.")

plt.tight_layout()
plt.savefig("loss_vs_model_mult.png", dpi=PNG_DPI, bbox_inches="tight")
plt.savefig("loss_vs_model_mult.pdf", bbox_inches="tight")
plt.close(fig)
