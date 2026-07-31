import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

from scipy.stats import norm, linregress, skew, gmean
from scipy.stats import wasserstein_distance

import os
import re
from tabulate import tabulate

from general_utils import *
from data_analysis import *

# ----------------------------------------------
# ----------- WoC PLOTTING FUNCTIONS -----------
# ----------------------------------------------

labels_full = {
        "scores": "Question-level",
        "domain_level_score": "Domain-level",
        "scores_avg": "Question-level (avg)",
    }

labels = {
    "scores": "per\nquestion",
    "domain_level_score": "per\ndomain",
    "scores_avg": "Question-level (avg)",
}

median_domain_c = np.median([c_per_domain[dom] for dom in df['domain_name'].unique()])
median_task_c = np.median([params['c'] for params in task_params_dict.values()])

def chosen_stats(params, score_type):
    if "inaccurate" in score_type and params['starting_accuracy'][1] >= 0.5:
        return False
    if "accurate" in score_type and "in" not in score_type and params['starting_accuracy'][1] < 0.5:
        return False
    if "high_copying_domains" in score_type and c_per_domain[params['domain']] <= median_domain_c:
        return False
    if "low_copying_domains" in score_type and c_per_domain[params['domain']] > median_domain_c:
        return False
    if "high" in score_type and "domains" not in score_type and c_per_domain[params['domain']] <= median_domain_c: # params['c'] < median_task_c:
        return False
    if "low" in score_type and "domains" not in score_type and c_per_domain[params['domain']] > median_domain_c: # params['c'] >= median_task_c:
        return False
    if "participants" in score_type:
        number = int(score_type.split("_")[0])
        if len(params['cons_ans']) < number:
            return False
    # if "med" in score_type and "domains" not in score_type and (params['c'] < 0.2 or params['c'] > 0.25):
    #     return False
    return True

def plot_avg_score_with_ci(alpha=0.95, title=None):
    """
    Plot average relative scores with parametric alpha confidence intervals.

    :param alpha: confidence level for intervals (default: 0.95)
    :param title: plot title (optional)
    """
    plt.figure(figsize=(5, 4))
    plt.title('Average Score by Condition (relative to Control)')

    x = np.arange(len(conditions)-1)  # numeric positions
    colors = [palette[c] for c in conditions.values()][1:]

    score_list = []

    colors = (colormaps['Set2'](i) for i in range(len(conditions)-1))  # create an iterator
    offset = 0.05  # horizontal offset for the two score types
    
    for score_type in ['scores', 'scores_avg']:
        for task_id, params in task_params_dict.items():
            scores = np.array(params[score_type], dtype=float)
            scores = scores[1:] - scores[0]   # relative to control
            score_list.append(scores)

        score = np.vstack(score_list)
        n_tasks = len(score)

        # --- Compute means and standard errors ---
        final_score = np.mean(score, axis=0)
        std_error = np.std(score, axis=0, ddof=1) / np.sqrt(n_tasks)

        # --- Compute z critical value ---
        delta = 1 - alpha
        z_crit = norm.ppf(1 - delta / 2)

        # --- Compute confidence intervals ---
        conf_low = final_score - z_crit * std_error
        conf_high = final_score + z_crit * std_error

        # --- Plot ---

        plt.errorbar(x - offset if score_type=='scores' else x + offset, final_score, label='Median' if score_type=='scores' else 'Average',
                     yerr=[final_score - conf_low, conf_high - final_score],
                     fmt='o', color=next(colors), capsize=2, markerfacecolor='none', markersize=6)

        
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.xticks(x, list(conditions.values())[1:])
    plt.ylabel('Average Relative Score')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/WOC/simpleWOC.png', dpi=300)
    plt.show()

    # Return results for reuse
    return {
        "final_score": final_score,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "z_crit": z_crit,
        "alpha": alpha
    }

def plot_avg_score_boxplot_with_points_and_ci(
    alpha=0.95,
    only_consensus=False,
    title=None,
    save_path="plots/WOC/simpleWOC_boxplot_points_ci.png",
    jitter=0.045,
    point_alpha=0.65,
    point_size=18,
    seed=0,
    scores_types = ["scores", "accurate_start_(high)", "accurate_start_(low)", "inaccurate_start_(high)", "inaccurate_start_(low)"],
    colors = [colormaps["Set2"](i) for i in range(10)],
    func=[lambda p: p['scores']],
    ax=None
):
    """
    Same data as plot_avg_score_with_ci, shown as:
        - box plots
        - all underlying points
        - mean ± 95% CI

    Values are condition score - control score, multiplied by 100.
    Colors match the previous plot: one Set2 color per score_type.
    """

    assert len(func) == 1 or len(func) >= len(scores_types), "func must be a single function or a list of functions of length >= length of scores_types\nlen(func) = {}, len(scores_types) = {}".format(len(func), len(scores_types))

    if ax is None:
        fig, ax = plt.subplots(figsize=(len(scores_types)*0.8, 3) if only_consensus else (8, 4))

    box_data = []
    box_labels = []
    box_colors = []
    rng = np.random.default_rng(seed)

    for i, score_type in enumerate(scores_types):
        score_list = []

        if "domains" in score_type:
            for dom in df["domain_name"].unique():
                tasks = df[df["domain_name"] == dom]["task_id"].unique()
                flag = 0

                scores = np.zeros(4)
                for task in tasks:
                    params = task_params_dict[task]
                    if not chosen_stats(params, score_type):
                        flag=1
                        break
                    scores += np.array(func[0](params) if len(func) == 1 else func[i](params), dtype=float)
                
                if flag==1:
                    continue
                scores = scores / len(tasks)
                scores = scores[1:(2 if only_consensus else 4)] - scores[0]
                score_list.append(scores)

        elif "domain" in score_type:
            for dom in df["domain_name"].unique():
                task = df[df["domain_name"] == dom]["task_id"].unique()[0]
                params = task_params_dict[task]

                try:
                    scores = np.array(params[score_type], dtype=float)
                except:
                    scores = np.array(
                        [params[score_type][cond] for cond in conditions.values()],
                        dtype=float
                    )

                scores = scores[1:(2 if only_consensus else 4)] - scores[0]
                score_list.append(scores)

        else:
            for task_id, params in task_params_dict.items():
                if not chosen_stats(params, score_type):
                    continue

                scores = np.array(func[0](params) if len(func) == 1 else func[i](params), dtype=float)
                scores = scores[1:(2 if only_consensus else 4)] - scores[0]
                score_list.append(scores)

        # else:
        #     for task_id, params in task_params_dict.items():
        #         if not chosen_stats(params, score_type):
        #             continue

        #         try:
        #             scores = np.array(params[score_type], dtype=float)
        #         except:
        #             scores = np.array(
        #                 [params[score_type][cond] for cond in conditions.values()],
        #                 dtype=float
        #             )

        #         scores = scores[1:(2 if only_consensus else 4)] - scores[0]
        #         score_list.append(scores)

        score = np.vstack(score_list) * 100

        if only_consensus:
            box_data.append(score[:, 0])
            box_labels.append(labels[score_type] if score_type in labels else score_type.replace("_", " "))
            box_colors.append(colors[i])
        else:
            for j, cond in enumerate(list(conditions.values())[1:4]):
                box_data.append(score[:, j])
                box_labels.append(f"{labels_full[score_type]}\n{cond}")
                box_colors.append(colors[i])

    positions = np.arange(1, len(box_data) + 1)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        tick_labels=box_labels,
        showmeans=False,
        patch_artist=True,
        showfliers=False,
        widths=0.3
    )

    # Color boxes using the same Set2 color per score type
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)
        median.set_visible(False)

    for whisker in bp["whiskers"]:
        whisker.set_color("gray")
        whisker.set_visible(False)

    for cap in bp["caps"]:
        cap.set_color("gray")
        cap.set_visible(False)

    # z critical value for alpha confidence interval
    delta = 1 - alpha
    z_crit = norm.ppf(1 - delta / 2)

    means = []
    conf_lows = []
    conf_highs = []

    # Overlay every point + mean ± CI
    for x, vals, color in zip(positions, box_data, box_colors):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]

        # all points
        x_jittered = x + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            x_jittered,
            vals,
            s=point_size,
            color=color,
            alpha=point_alpha,
            edgecolors="none",
            zorder=3
        )

        # mean and 95% CI
        n = len(vals)
        mean = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0
        ci = z_crit * se

        means.append(mean)
        conf_lows.append(mean - ci)
        conf_highs.append(mean + ci)

        ax.errorbar(
            x,
            mean,
            yerr=[[ci], [ci]],
            fmt="o",
            color="black",
            capsize=3,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=1.1,
            zorder=4,
            label=f"{int(alpha * 100)}% CI" if x == positions[0] else None
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Relative Score (%)")
    ax.set_title(title or f'{"Consensus " if only_consensus else ""}Percentile Scores: Box Plot')
    ax.legend(loc='upper right', fontsize=7)

    ax.set_xticks(range(1, len(box_labels) + 1), box_labels, rotation=45, ha="right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    return {
        "box_data": box_data,
        "box_labels": box_labels,
        "box_colors": box_colors,
        "means": means,
        "conf_low": conf_lows,
        "conf_high": conf_highs,
        "z_crit": z_crit,
        "alpha": alpha
    }

def compute_stats_for_regression(
        func_x=lambda task_id: c_per_domain[task_params_dict[task_id]['domain']],
        func_y=lambda task_id: task_params_dict[task_id]['scores'][0],
        acc=(0, 1),
        condition=0,                # 0 for control, 1 for consensus, etc. (index into scores list)
        mae=True,                   # True -> MAE-like; False -> RMSE-like transform (square then sqrt for means)):
):
    """
    Compute X and Y values for regression analysis based on task parameters and specified functions.
    
    :param func_x: Function that takes a task_id and returns the x-value for regression (default: average copying probability for the task's domain)
    :param func_y: Function that takes a task_id and returns the y-value for regression (default: score for the specified condition)
    :param acc: Tuple specifying the accuracy range (min, max) for filtering tasks based on their starting accuracy in the specified condition (default: (0, 1) to include all tasks)
    :param condition: Index of the condition to use for filtering tasks based on starting accuracy (default: 0 for control)
    :param mae: Boolean indicating whether to use MAE-like (True) or RMSE-like (False) transform (default: True)
    """
    X_all, Y_all = [], []

    X_means, Y_means = [], []
    mean_domain_ids = []         # 1..K labels for means (ordered)
    mean_domain_names = []       # domain strings aligned with means
    task_counts = []             # tasks per domain aligned with means
    mean_weights = []            # weights per domain mean (for weighted regression)

    domains_in_order = [d for d in ordered_domains.keys() if d != "spatial_reasoning"]

    eps = 1e-10
    for dom_idx, domain in enumerate(domains_in_order, start=1):
        task_ids = df.loc[df["domain_name"] == domain, "task_id"].unique()

        # per-task lists (for weights + optional label positioning)
        y_list = None
        n = 0

        for task_id in task_ids:
            if not (acc[0] < task_params_dict[task_id]["starting_accuracy"][condition] <= acc[1]):
                continue
            
            n += 1
            x = func_x(task_id)
            val = func_y(task_id) if mae else func_y(task_id)**2

            X_all.append(x)
            Y_all.append(val if isinstance(val, (int, float)) else val[1] - val[0])  # if multiple conditions, take specified one for regression
            if y_list is None:
                y_list = np.array([val.copy()] if isinstance(val, np.ndarray) else [val])
            else:
                y_list = np.append(y_list, np.array([val.copy()]) if isinstance(val, np.ndarray) else np.array([val]), axis=0)

        if n == 0:
            continue

        # domain means
        domain_mean = np.sum(y_list, axis=0) / n if mae else np.sqrt(np.sum(y_list, axis=0) / n)
        y_mean = domain_mean if isinstance(domain_mean, (int, float)) else domain_mean[1] - domain_mean[0]  # if multiple conditions, take difference for regression
        X_means.append(np.mean([func_x(task_id) for task_id in task_ids]))  # mean x for the domain (e.g. mean copying probability)
        Y_means.append(y_mean)
        mean_domain_ids.append(dom_idx)
        mean_domain_names.append(domain)
        task_counts.append(n)

        # mean weights for weighted regression: n / var (or n / sum of variances if multiple conditions)
        if n <= 1:
            mean_weights.append(eps)
        else:
            v = float(np.var(y_list, ddof=1)) if isinstance(domain_mean, (int, float)) else np.var(y_list, axis=0, ddof=1)[1] + np.var(y_list, axis=0, ddof=1)[0]  # if multiple conditions, sum variances for weighting
            mean_weights.append(n / max(eps, v))

    return {
        "X_all": X_all,
        "Y_all": Y_all,
        "X_means": X_means,
        "Y_means": Y_means,
        "mean_domain_ids": mean_domain_ids,
        "mean_domain_names": mean_domain_names,
        "task_counts": task_counts,
        "mean_weights": mean_weights,
        }

# def all_tasks_regression(
#     *,
#     func_x=lambda task_id: c_per_domain[task_params_dict[task_id]['domain']],
#     func_y=lambda d: d['scores'][0],
#     acc=(0, 1),
#     condition=0,                # 0 for control, 1 for consensus, etc. (index into scores list)
#     mae=True,                   # True -> MAE-like; False -> RMSE-like transform (square then sqrt for means)
#     ax=None,
#     color="#41156D",

#     # labels and title
#     title=None,
#     label=None,
#     label_alpha=0.7,
#     label_fontsize=10,

#     # axes labels
#     ylabel=None,
#     xlabel='copying probability',
#     ylabel_fontsize=12,
#     xlabel_fontsize=12,

#     # display options
#     show_tasks=True,
#     alpha_task=0.25,
#     s_task=14,

#     # display means
#     show_means=True,
#     annotate_domain_means=True,   # put domain index text on means (1..K)
#     alpha_mean=0.9,
#     s_mean=25,
#     color_mean=None,
#     marker_mean="s",

#     # regression
#     fit_on="all",               # "all": use all tasks | "means": use domain means
#     show_fit=True,
#     fit_label=True,
#     joint_labels=False,

#     # --- extra features ---
#     weighted=False,             # weighted regression for fit_on="means"
#     equaldots=True,             # if False, mean marker sizes scale with task_count
#     show_zero_line=False,          # horizontal line at y=0 for difference plot

#     return_data=False,          # return dict of arrays for downstream use
#     grid_option='both',           # show axis grid
#     alpha_grid=0.25,
# ):
#     """
#     Function to plot tasks and domain means with regression fit, with flexible options
#     """

#     if ax is None:
#         _, ax = plt.subplots(figsize=(6, 5))

#     def _weighted_linfit(x, y, w):
#         """
#         Weighted least squares line fit y ≈ a x + b
#         Returns (a, b). (No p-value here.)
#         """
#         x = np.asarray(x, dtype=float)
#         y = np.asarray(y, dtype=float)
#         w = np.asarray(w, dtype=float)

#         w = np.clip(w, 0.0, np.inf)
#         if np.all(w == 0):
#             w = np.ones_like(w)

#         W = np.sum(w)
#         xbar = np.sum(w * x) / W
#         ybar = np.sum(w * y) / W

#         denom = np.sum(w * (x - xbar) ** 2)
#         if denom <= 0:
#             return 0.0, ybar
#         a = np.sum(w * (x - xbar) * (y - ybar)) / denom
#         b = ybar - a * xbar
#         return float(a), float(b)

#     # ---------- collect ---------
#     results = compute_stats_for_regression(
#         func_x=func_x,
#         func_y=func_y,
#         acc=acc,
#         condition=condition,
#         mae=mae,
#     )

#     # nothing to plot
#     if (not show_tasks or len(results["X_all"]) == 0) and (not show_means or len(results["X_means"]) == 0):
#         print("No points to plot.")
#         return ax

#     # ---------- plot tasks ----------
#     if show_tasks and len(results["X_all"]) > 0:
#         ax.scatter(
#             np.asarray(results["X_all"]), np.asarray(results["Y_all"]),
#             s=s_task, alpha=alpha_task, color=color,
#             label=label if not joint_labels else None
#         )

#     # ---------- plot means ----------
#     if show_means and len(results["X_means"]) > 0:
#         if equaldots:
#             sizes = np.full(len(results["X_means"]), s_mean, dtype=float)
#         else:
#             sizes = s_mean/20 * np.asarray(results["task_counts"], dtype=float)

#         ax.scatter(
#             np.asarray(results["X_means"]), np.asarray(results["Y_means"]),
#             s=sizes, marker=marker_mean, alpha=alpha_mean,
#             color=make_color_darker(color, factor=0.7) if color_mean is None else color_mean,
#             label=label if not (show_tasks or joint_labels) else None
#         )

#         # place domain labels near means (using domain index for ordering if available, otherwise domain name)
#         if annotate_domain_means:
#             for x, domain, ym in zip(results["X_means"], results["mean_domain_names"], results["Y_means"]):
#                 y_text = ym

#                 label = ordered_domains[domain] if domain in ordered_domains else str(domain)
#                 ax.text(x, y_text, label, fontsize=label_fontsize, alpha=label_alpha, weight="bold")

#     # baseline for diff
#     if show_zero_line:
#         ax.axhline(0, color="black", ls="--", lw=1)

#     # ---------- regression ----------
#     if show_fit:
#         if fit_on == "all":
#             x_fit = np.asarray(results["X_all"])
#             y_fit = np.asarray(results["Y_all"])
#             w_fit = None
#         elif fit_on == "means":
#             x_fit = np.asarray(results["X_means"])
#             y_fit = np.asarray(results["Y_means"])
#             w_fit = np.asarray(results["mean_weights"])
#         else:
#             raise ValueError("fit_on must be one of: 'all', 'means'")

#         if len(x_fit) >= 2:
#             if weighted and (fit_on == "means"):
#                 slope, intercept = _weighted_linfit(x_fit, y_fit, w_fit)
#                 p = None
#             else:
#                 slope, intercept, r, p, se = linregress(x_fit, y_fit)

#             xx = np.linspace(x_fit.min(), x_fit.max(), 200)
#             lab = f"{label}\n" if joint_labels else ""
#             if fit_label:
#                 if p is None:
#                     lab = lab + f"β={slope:.2f}" # (weighted, {fit_on})"
#                 else:
#                     lab = lab + f"β={slope:.2f}, p={p:.3f}" # ({fit_on})"
#             else:
#                 lab = None # f"Fit({fit_on})"

#             ax.plot(xx, slope * xx + intercept, color=color, alpha=0.7, label=lab)
        
#         else:
#             print("Not enough points for regression fit.")

#     ax.set_xlabel(xlabel if xlabel != "default" else "copying parameter", fontsize=xlabel_fontsize)
#     ax.set_ylabel(ylabel if ylabel != "default" else "percentile score", fontsize=ylabel_fontsize)
#     ax.set_title(title, fontsize=ylabel_fontsize+2)
#     ax.grid(grid_option, axis=grid_option if grid_option else 'both', alpha=alpha_grid)
#     ax.legend()

#     if return_data:
#         return {
#             "X_all": np.asarray(results["X_all"]),
#             "Y_all": np.asarray(results["Y_all"]),
#             "X_means": np.asarray(results["X_means"]),
#             "Y_means": np.asarray(results["Y_means"]),
#             "domains": list(results["mean_domain_names"]),
#             "task_counts": np.asarray(results["task_counts"]),
#             "mean_weights": np.asarray(results["mean_weights"]),
#             "fitting": {
#                 "slope": slope,
#                 "intercept": intercept,
#                 "p_value": p if 'p' in locals() else None,
#                 "r_squared": r**2 if 'r' in locals() else None,
#             }
#         }

#     return ax

def all_tasks_regression(
    *,
    func_x=lambda task_id: c_per_domain[task_params_dict[task_id]['domain']],
    func_y=lambda task_id: task_params_dict[task_id]['scores'][0],
    acc=(0, 1),
    condition=0,                # 0 for control, 1 for consensus, etc. (index into scores list)
    mae=True,                   # True -> MAE-like; False -> RMSE-like transform (square then sqrt for means)
    ax=None,
    color="#41156D",

    # labels and title
    title=None,
    label=None,
    label_alpha=0.7,
    label_fontsize=10,
    legend_fontsize=10,

    # axes labels
    ylabel=None,
    xlabel='copying probability',
    ylabel_fontsize=12,
    xlabel_fontsize=12,

    # display options
    show_tasks=True,
    alpha_task=0.25,
    s_task=14,

    # display means
    show_means=True,
    annotate_domain_means=True,   # put domain index text on means (1..K)
    alpha_mean=0.9,
    s_mean=25,
    color_mean=None,
    marker_mean="s",
    empty_negatives=False,        # if True, mean markers with negative y are hollow (white fill)

    # regression
    fit_on="all",               # "all" | "means"
    show_fit=True,
    fit_label=True,
    show_corr=False,
    joint_labels=False,

    # --- extra features ---
    weighted=False,             # weighted regression for fit_on="means"
    equaldots=True,             # if False, mean marker sizes scale with task_count
    show_zero_line=False,          # horizontal line at y=0 for difference plot

    return_data=False,          # return dict of arrays for downstream use
    grid_option='both',           # show axis grid
    alpha_grid=0.25,
):
    """
    x = mu_c(domain)

    y depends on `mode`:
      - "ctrl": control score
      - "cons": consensus score
      - "diff": consensus - control
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    def _weighted_linfit(x, y, w):
        """
        Weighted least squares line fit y ≈ a x + b
        Returns (a, b). (No p-value here.)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)

        w = np.clip(w, 0.0, np.inf)
        if np.all(w == 0):
            w = np.ones_like(w)

        W = np.sum(w)
        xbar = np.sum(w * x) / W
        ybar = np.sum(w * y) / W

        denom = np.sum(w * (x - xbar) ** 2)
        if denom <= 0:
            return 0.0, ybar
        a = np.sum(w * (x - xbar) * (y - ybar)) / denom
        b = ybar - a * xbar
        return float(a), float(b)

    # ---------- collect ---------
    results = compute_stats_for_regression(
        func_x=func_x,
        func_y=func_y,
        acc=acc,
        condition=condition,
        mae=mae,
    )

    # nothing to plot
    if (not show_tasks or len(results["X_all"]) == 0) and (not show_means or len(results["X_means"]) == 0):
        print("No points to plot.")
        return ax

    # ---------- plot tasks ----------
    if show_tasks and len(results["X_all"]) > 0:
        ax.scatter(
            np.asarray(results["X_all"]), np.asarray(results["Y_all"]),
            s=s_task, alpha=alpha_task, color=color,
            label=label if not joint_labels else None
        )

    # ---------- plot means ----------
    if show_means and len(results["X_means"]) > 0:
        if equaldots:
            sizes = np.full(len(results["X_means"]), s_mean, dtype=float)
        else:
            sizes = s_mean/20 * np.asarray(results["task_counts"], dtype=float)

        color_for_means = get(color_mean, make_color_darker(color, factor=0.7))
        
        ax.scatter(
            np.asarray(results["X_means"]), np.asarray(results["Y_means"]),
            s=sizes, marker=marker_mean, alpha=alpha_mean,
            facecolors=np.array(['white' if i < 0 and empty_negatives else color_for_means for i in results["Y_means"]]), linewidths=0.5,
            color=color_for_means,
            label=label if not (show_tasks or joint_labels) else None
        )

        # old labels_position: place ordered_domains[domain] at chosen y
        if annotate_domain_means:
            for x, domain, ym in zip(results["X_means"], results["mean_domain_names"], results["Y_means"]):
                y_text = ym

                txt_label = ordered_domains[domain] if domain in ordered_domains else str(domain)
                ax.text(x, y_text, txt_label, fontsize=label_fontsize, alpha=label_alpha, weight="bold")

    # baseline for diff
    if show_zero_line:
        ax.axhline(0, color="black", ls="--", lw=1)

    # ---------- regression ----------
    if show_fit:
        if fit_on == "all":
            x_fit = np.asarray(results["X_all"])
            y_fit = np.asarray(results["Y_all"])
            w_fit = None
        elif fit_on == "means":
            x_fit = np.asarray(results["X_means"])
            y_fit = np.asarray(results["Y_means"])
            w_fit = np.asarray(results["mean_weights"])
        else:
            raise ValueError("fit_on must be one of: 'all', 'means'")

        if len(x_fit) >= 2:
            if weighted and (fit_on == "means"):
                slope, intercept = _weighted_linfit(x_fit, y_fit, w_fit)
                p = None
            else:
                slope, intercept, r, p, se = linregress(x_fit, y_fit)

            xx = np.linspace(x_fit.min(), x_fit.max(), 200)
            lab = f"{label}\n" if joint_labels else ""
            if fit_label:
                if show_corr:
                    lab = lab + f"ρ={r**2:.2f}{f', p={p:.2g}' if p is not None else ''}" # ({fit_on})"
                else:
                    lab = lab + f"β={slope:.2f}{f', p={p:.2g}' if p is not None else ''}" # ({fit_on})"
            else:
                lab = None # f"Fit({fit_on})"

            ax.plot(xx, slope * xx + intercept, color=color, alpha=0.7, label=lab)
        
        else:
            print("Not enough points for regression fit.")

    ax.set_xlabel(xlabel if xlabel != "default" else "copying parameter", fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel != "default" else "percentile score", fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=ylabel_fontsize+2)
    ax.grid(grid_option, axis=grid_option if grid_option else 'both', alpha=alpha_grid)
    ax.legend(fontsize=legend_fontsize, loc='best', frameon=True, framealpha=0.7)

    if return_data:
        return {
            "X_all": np.asarray(results["X_all"]),
            "Y_all": np.asarray(results["Y_all"]),
            "X_means": np.asarray(results["X_means"]),
            "Y_means": np.asarray(results["Y_means"]),
            "domains": list(results["mean_domain_names"]),
            "task_counts": np.asarray(results["task_counts"]),
            "mean_weights": np.asarray(results["mean_weights"]),
            "fitting": {
                "slope": slope,
                "intercept": intercept,
                "p_value": p if 'p' in locals() else None,
                "r_squared": r**2 if 'r' in locals() else None,
            }
        }

    return ax


def all_tasks_regression_two_conditions(
        func_y=lambda task_id: task_params_dict[task_id]['scores'],  # should return list/array of scores
        color1 =palette['Control'], color2=palette['Consensus'], color_diff="#8828AE",
        label1="Control", label2="Social influence", label_diff="Difference",
        ax=None, title=None,
        base=dict()):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

    base1 = {
        'func_y': lambda d: func_y(d)[0],
        'color': color1,
        'label': label1,
        'annotate_domain_means': False,
        'ax': ax,
        **base
    }
    ret1 = all_tasks_regression(**base1)

    base2 = {
        'func_y': lambda d: func_y(d)[1],
        'color': color2,
        'label': label2,
        'annotate_domain_means': False,
        'ax': ax,
        **base
    }
    ret2 = all_tasks_regression(**base2)

    base_diff = {
        'func_y': func_y,
        'ax': ax,
        'color': color_diff,
        'label': label_diff,
        'annotate_domain_means': True,
        'title': title,
        'show_zero_line': True,
        **base
    }
    ret3 = all_tasks_regression(**base_diff)

    return {
        "control": ret1,
        "consensus": ret2,
        "difference": ret3,
    }


# ----------------------------------------------
# ----------- BIAS-VARIANCE ANALYSIS -----------
# ----------------------------------------------
def bootstrap_bias_variance_ci(finals, correct_answer, sigma, B=2000, alpha=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    correct_answer = correct_answer/sigma  # normalize
    finals = np.asarray(finals)/sigma  # normalize
    n = len(finals)

    mean_final = np.mean(finals)
    bias = mean_final - correct_answer
    bias2 = bias**2
    variance = np.var(finals, ddof=0)
    mse = np.mean((finals - correct_answer)**2)

    boot_bias = np.zeros(B)
    boot_bias2 = np.zeros(B)
    boot_var = np.zeros(B)
    boot_mse = np.zeros(B)

    for b in range(B):
        sample = rng.choice(finals, size=n, replace=True)
        boot_bias[b] = np.mean(sample) - correct_answer
        boot_bias2[b] = boot_bias[b]**2
        boot_var[b] = np.var(sample, ddof=1)
        boot_mse[b] = np.mean((sample - correct_answer)**2)

    q_lo = 100 * alpha / 2
    q_hi = 100 * (1 - alpha / 2)

    return {
        'bias': bias,
        'bias2': bias2,
        'variance': variance,
        'mse': mse,
        'bias_ci': np.percentile(boot_bias, [q_lo, q_hi]),
        'bias2_ci': np.percentile(boot_bias2, [q_lo, q_hi]),
        'variance_ci': np.percentile(boot_var, [q_lo, q_hi]),
        'mse_ci': np.percentile(boot_mse, [q_lo, q_hi]),
    }

def compute_mse_domain_reuse(domain,
                             Ns=[130, 300, 800],
                             num_simulations=200,
                             bootstrap=False,
                             bootstrap_B=2000,
                             config_id=0,
                             load=False,
                             random_state=123):
        maxN = max(Ns)
        rng = np.random.default_rng(random_state)
        task_ids = df[df["domain_name"] == domain]["task_id"].unique()
    
        disp = DisplayProgressBar(
            total=len(task_ids),
            prefix=f"{domain}:".ljust(25),
            suffix="Complete",
            length=50,
        )

        # Run simulations ONCE with largest N

        cached_results = {}

        for task_id in task_ids:

            data = get_answers(task_id)
            sigma = np.std(data)
            correct_answer = get_correct_answer(task_id)

            cached_results[task_id] = {}

            for condition in ["control", "consensus", "consensus_fix"]:
                for statistic in ["mean", "median"]:

                    if condition == "consensus_fix" and statistic == "mean":
                        continue
                    
                    flag = False
                    config = configs(task_id, config_id, N=maxN)
                    args = config["args"].copy()
                    args["condition"] = condition
                    args["statistic"] = statistic
                    
                    sims = [{} for _ in range(num_simulations)]  # placeholder for simulations

                    if load:
                        try:
                            sims = load_from_file(
                                f"data/evaluation_results/{domain}/{task_id}/full_results_{maxN}_{condition}_{statistic}.pkl"
                            )
                            sims = sims if isinstance(sims, list) else sims.to_dict(orient="records")
                        except:
                            print(f"Something went wrong while loading {domain}/{task_id}/{condition}/{statistic}. Running simulations...")
                            flag = True

                    if not load or flag:
                        for sim_idx in range(num_simulations):
                            sims[sim_idx] = sim_from_data(data, **args)

                    cached_results[task_id][(condition, statistic)] = {
                        "sims": sims,
                        "sigma": sigma,
                        "correct": correct_answer,
                    }

                    # save full results for later inspection
                    os.makedirs(
                        f"data/evaluation_results/{domain}/{task_id}",
                        exist_ok=True,
                    )
                    pd.DataFrame(sims).to_pickle(
                        f"data/evaluation_results/{domain}/{task_id}/full_results_{maxN}_{condition}_{statistic}.pkl"
                    )

            disp.update()

        # Evaluate every N using cached simulations

        for j, N in enumerate(Ns):

            rows = [{} for _ in range(len(task_ids) * 5)]  # 5 combinations of condition/statistic
            rows_idx = 0

            for task_id in task_ids:

                for (condition, statistic), info in cached_results[
                    task_id
                ].items():

                    finals = [0 for _ in range(len(info["sims"]))]

                    for sim_idx, sim in enumerate(info["sims"]):
                        finals[sim_idx] = sim["results"][N - 1]

                    # compute bias, variance, mse over sigma-normalized results
                    correct_answer_sigma = correct_answer/sigma  # normalize
                    finals_sigma = np.asarray(finals)/sigma  # normalize

                    mean_final = np.mean(finals_sigma)
                    bias = mean_final - correct_answer_sigma
                    bias2 = bias**2
                    variance = np.var(finals_sigma, ddof=0)
                    mse = np.mean((finals_sigma - correct_answer_sigma)**2)

                    summary = {
                        "bias": bias,
                        "bias2": bias2,
                        "variance": variance,
                        "mse": mse,
                    }

                    if bootstrap:
                        summary.update(
                            bootstrap_bias_variance_ci
                            (
                            finals=finals,
                            correct_answer=info["correct"],
                            sigma=info["sigma"],
                            B=bootstrap_B,
                            random_state=rng.integers(1_000_000_000),
                            )
                        )

                    rows[rows_idx] = {
                            "task_id": task_id,
                            "config_id": config_id,
                            "condition": condition,
                            "statistic": statistic,
                            "label": f"{condition}\n{statistic}",     
                        } | summary
                    rows_idx += 1

            df_help = pd.DataFrame(rows)

            df_main = (
                df_help.groupby(
                    ["condition", "statistic"], as_index=False
                )
                .mean(numeric_only=True)
            )

            df_main["label"] = (
                df_main["condition"] + "\n" + df_main["statistic"]
            )

            df_main.to_pickle(f"data/evaluation_results/{domain}/mse_split_N{N}.pkl")

def compute_mse_multi_domain(
    domains=[0, 4, 9, 14, 19],
    **args):
    for i, id_ in enumerate(domains):
        domain = domains_ordered_by_c[id_] if isinstance(id_, int) else id_
        compute_mse_domain_reuse(
            domain=domain,
            **args
        )


# plot mse as 2 curves, one for bias one for variance for each domain, with x-axis as N and y-axis as mse, color by c value for that domain
def plot_mse_curves(domain, 
                    Ns=[50, 150, 250, 500, 750, 1000, 1250, 1500, 1750, 2000], 
                    conditions=["control", "consensus", "consensus_fix"],
                    statistics=["mean", "median"],
                    title1='default',
                    title2=None,
                    title3=None,
                    ax=None,
                    save=False):

    colors = {
        ("control", "mean"): "#74B4F4",
        ("control", "median"): "#C374F4",
        ("consensus", "mean"): "#AAC03A",
        ("consensus", "median"): "#F46D2E",
        ("consensus_fix", "median"): "#314AC7"
    }

    markers = {
        ("control", "mean"): "o",
        ("control", "median"): "*",
        ("consensus", "mean"): "s",
        ("consensus", "median"): "d",
        ("consensus_fix", "median"): "^"
    }
    
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(7, 15), sharex=True)
    else:
        fig = ax[0].get_figure()
    min_c = min(c_per_domain.values())
    max_c = max(c_per_domain.values())

    for condition in conditions:
        for statistic in statistics:
            if statistic == "mean" and condition == "consensus_fix":
                continue  # skip this combination, as it is not relevant
            bias2_means = []
            variance_means = []
            mse_means = []
            Ns_final = []
            
            for N in Ns:
                # load from pickle if exists
                pickle_path = f"data/evaluation_results/{domain}/mse_split_N{N}.pkl"
                if os.path.exists(pickle_path):
                    df_main = pd.read_pickle(pickle_path)
                else:
                    print(f"Pickle not found for {domain} N={N}, skipping...")
                    continue

                Ns_final.append(N)
                # df_main = df_help.groupby(['condition', 'statistic'], as_index=False).mean(numeric_only=True)


                # compute avg bias^2 and variance across conditions/statistics
        
                row = df_main[(df_main['condition'] == condition) & (df_main['statistic'] == statistic)]
                if not row.empty:
                    bias2_means.append(row['bias2'].values[0])
                    variance_means.append(row['variance'].values[0])
                    mse_means.append(row['mse'].values[0])
                else:
                    bias2_means.append(np.nan)
                    variance_means.append(np.nan)
                    mse_means.append(np.nan)
            ax[0].plot(Ns_final, bias2_means, 
                        marker=markers[condition, statistic],
                        label=f"{condition}-{statistic}", 
                        color=colors[condition, statistic],
                        alpha=0.8)
            ax[1].plot(Ns_final, variance_means,
                        marker=markers[condition, statistic],
                        label=f"{condition}-{statistic}", 
                        color=colors[condition, statistic],
                        alpha=0.8)
            ax[2].plot(Ns_final, mse_means,
                        marker=markers[condition, statistic],
                        label=f"{condition}-{statistic}", 
                        color=colors[condition, statistic],
                        alpha=0.8)

    # ax[0].plot(Ns_final, bias2_means, label=domain, color=color)
    # ax[1].plot(Ns_final, variance_means, label=domain, color=color)

    ax[0].set_ylabel("Average Bias²")
    ax[0].set_title(f"Average Bias² vs N for {domain}" if title1 == 'default' else title1)
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_xlabel("N")
    ax[1].set_ylabel("Average Variance")
    ax[1].set_title(f"Average Variance vs N for {domain}" if title2 == 'default' else title2)
    ax[1].legend()
    ax[1].grid(True)

    ax[2].set_xlabel("N")
    ax[2].set_ylabel("Average MSE")
    ax[2].set_title(f"Average MSE vs N for {domain}" if title3 == 'default' else title3)
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(f"plots/simulations/mse_curves.png", bbox_inches='tight', dpi=300)
    # plt.show()

def plot_mse_over_N_all_statistics(domains=domains_ordered_by_c, 
                                   Ns=[50, 150, 250, 500, 750, 1000, 1250, 1500, 1750, 2000], 
                                   saveto=None):

    # rank domains according to average MSE for N=5000
    domains_ranked_MSE = []
    for domain in df['domain_name'].unique():
        try:
            df_main = pd.read_pickle(f"data/evaluation_results/{domain}/mse_split_N5000.pkl")
            avg_mse = df_main['mse'].mean()
            domains_ranked_MSE.append((domain, avg_mse))
        except:
            print(f"No data for {domain} at N=5000")

    domains_ranked_MSE.sort(key=lambda x: x[1])

    # find percentage of variance in MSE for each domain at N=5000 for consensus median condition
    variance_percentage = {}
    for domain in df['domain_name'].unique():
        try:
            df_main = pd.read_pickle(f"data/evaluation_results/{domain}/mse_split_N5000.pkl")
            mse_consensus_median = df_main[(df_main['condition'] == 'consensus') & (df_main['statistic'] == 'median')]['mse'].values[0]
            variance_consensus_median = df_main[(df_main['condition'] == 'consensus') & (df_main['statistic'] == 'median')]['variance'].values[0]
            variance_percentage[domain] = variance_consensus_median / mse_consensus_median * 100
        except:
            print(f"No data for {domain} at N=5000")

    if domains == "all":
        domains = [d[0] for d in domains_ranked_MSE][::-1]  # reverse order to have highest MSE first
    
    fig1, ax1 = plt.subplots(max(1, len(domains)//4), min(4, len(domains)), 
                             figsize=(3*min(4, len(domains)), 3*max(1, len(domains)//4)), 
                             sharex=True, sharey=True)

    fig2, ax2 = plt.subplots(max(1, len(domains)//4), min(4, len(domains)), 
                             figsize=(3*min(4, len(domains)), 3*max(1, len(domains)//4)), 
                             sharex=True, sharey=True)

    fig3, ax3 = plt.subplots(max(1, len(domains)//4), min(4, len(domains)), 
                             figsize=(3*min(4, len(domains)), 3*max(1, len(domains)//4)), 
                            sharey='row', sharex=True)

    if max(1, len(domains)//4) == 1:
        ax1 = np.array([ax1])
        ax2 = np.array([ax2])
        ax3 = np.array([ax3])

    for i, domain in enumerate(domains):
        title = f'{domain}\nc={c_per_domain[domain]:.2f} VE={variance_percentage.get(domain, 0):.2f}%'
        plot_mse_curves(
            domain=domain,
            title1=title,
            title2=title,
            title3=title,
            Ns=Ns,
            # ax=ax[:, i],
            ax=[ax1[i//4, i%4], ax2[i//4, i%4], ax3[i//4, i%4]],
            save=False)
        ax1[i//4, i%4].set_xscale('log')
        ax2[i//4, i%4].set_xscale('log')
        ax3[i//4, i%4].set_xscale('log')
        
        ax1[i//4, i%4].set_ylabel(None)
        ax2[i//4, i%4].set_ylabel(None)
        ax3[i//4, i%4].set_ylabel(None)

        if i != 3:
            ax1[i//4, i%4].legend_.remove()
            ax2[i//4, i%4].legend_.remove()
            ax3[i//4, i%4].legend_.remove()
        
        if i//4 != max(1, len(domains)//4) - 1:
            ax1[i//4, i%4].set_xlabel(None)
            ax2[i//4, i%4].set_xlabel(None)
            ax3[i//4, i%4].set_xlabel(None)


    fig1.suptitle("Average Bias² vs N for all domains", y=0.99)
    fig2.suptitle("Average Variance vs N for all domains", y=0.99)
    fig3.suptitle("Average MSE vs N for all domains", y=0.99)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    if saveto:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        fig1.savefig(f"{saveto}_bias.png", bbox_inches='tight', dpi=300)
        fig2.savefig(f"{saveto}_variance.png", bbox_inches='tight', dpi=300)
        fig3.savefig(f"{saveto}_mse.png", bbox_inches='tight', dpi=300)

    plt.show()

# ----------------------------------------------
# ----------- THEORY VISUALIZATION -------------
# ----------------------------------------------
def plot_sic_with_traj(task, saveto=None, ax=None, title=None, offset=0, bounds=None):
    """
    Function to plot the SIC (Social Influence Curve) with trajectory for a given task.
    
    :param task: id of the task to plot
    :param saveto: path to save the plot, if None the plot is not saved
    :param ax: matplotlib axis to plot on, if None a new figure and axis are created
    :param title: title of the plot, if None a default title is used
    :param offset: offset for the bounds calculation
    :param bounds: tuple specifying the bounds for the plot, if None they are calculated based on the control answers
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4.5))

    params = task_params_dict[task]
    L, U = params['L'], params['U']
    ctrl_ans = sorted(params['ctrl_ans'])

    if bounds is None: 
        bounds = (ctrl_ans[offset], ctrl_ans[-1-offset])

    ax.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], '--', color='lightgray', linewidth=1, alpha=0.5)

    ax.hlines(L, bounds[0], L, color="#88C5AEFF", linewidth=3, alpha=1)
    ax.hlines(U, U, bounds[1], color="#88C5AEFF", linewidth=3, alpha=1)
    ax.plot([L, U], [L, U], color="#88C5AEFF", linewidth=3, alpha=1)
        
    ax.plot(U, U, 'v', color="#88C5AEFF", label=f'U = {U:.2f}', markersize=6)
    ax.plot(L, L, '^', color="#88C5AEFF", label=f'L = {L:.2f}', markersize=6)

    max_T = np.round(max(len(params['ctrl_ans']), len(params['cons_ans'])), -1)
    max_ctrl = len(params['ctrl_ans'])/max_T*(bounds[1] - bounds[0]) + bounds[0]
    max_cons = len(params['cons_ans'])/max_T*(bounds[1] - bounds[0]) + bounds[0]
    timesteps1 = np.linspace(bounds[0], max_ctrl, len(params['ctrl_ans']))
    timesteps2 = np.linspace(bounds[0], max_cons, len(params['cons_ans']))

    ax.plot(timesteps1, params['ctrl_meds'], color="black", linewidth=1.5,
            label="Control", alpha=0.8)
    ax.plot(timesteps2, params['cons_meds'], color="hotpink", linewidth=1.5,
            label="Social influence", alpha=0.8)

    # shade area between y=L and y=U
    ax.fill_between(ctrl_ans, L, U, color="#88C5AEFF", alpha=0.1)


    ax.plot(np.linspace(bounds[0], bounds[1], 5) , [params['correct'] for _ in range(5)], color='black', 
            alpha=0.2, linestyle='--', marker='*', label=f'Correct = {params["correct"]:.2f}')

    ax.legend(loc='lower right', fontsize=9)

    ax.set_ylim(bounds[0], bounds[1])
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_title(title if title else f'[{task}] {params["prompt"]} [{params["c"]:.2f}]', fontsize=14)

    ax.set_ylabel("Answer value", fontsize=12)
    ax.set_xlabel("Observed answer", fontsize=12)

    # second xaxis on top to show the actual timesteps
    secax = ax.secondary_xaxis('top')
    secax.set_xlim(ax.get_xlim())
    secax.set_xticks(np.linspace(bounds[0], bounds[1], 5))
    secax.set_xticklabels([f"{i*max_T/4:.0f}" for i in range(5)], rotation=45, fontsize=8)
    
    plt.tight_layout()
    if saveto:
        plt.savefig(saveto, dpi=300, bbox_inches="tight")
        # plt.close()

def get_short_prompt(task_id):
    """
    Function to extract a short prompt from the full prompt for a given task for the 'calories' domain
    
    :param task_id: id of the task
    """
    prompt = get_prompt(task_id)
    # prompt = re.split(r'[" contain]"["does" ]', prompt)[1]
    # split by 'contain' or 'does'
    prompt = re.split(r' contain|does ', prompt)[1]
    prompt = prompt.split('(')[0] + prompt.split(') ')[1] if ')' in prompt else prompt
    return prompt

# domain SIC with trajectories
def multi_task_plot_norm_horizontal(
    dom,
    N=20,
    cutoff=5,
    normalization="perc",
    plot=False,
    saveto=False
):
    """
    Plot (normalized) trajectories and predictions for multiple tasks in a domain (horizontal layout).

    - Tasks are on the x-axis; (normalized) answer values on the y-axis.
    - Shows control/consensus median trajectories, final median, correct answer,
      and the theory-predicted interval per task.

    Parameters
    ----------
    dom : str
        Domain name.
    N : int
        Max number of tasks to plot.
    cutoff : int
        Passed to normalization helper.
    normalization : {"perc","linear",...}
        Normalization mode.
    plot : bool
        If True, show the plot; otherwise save to disk.
    """
    task_ids = df[df["domain_name"] == dom]["task_id"].unique()

    fig, ax = plt.subplots(figsize=(min(len(task_ids), N) * 0.6, 5))

    # Domain-level predicted band (in percentile space)
    k_domain = c_per_domain[dom] / (1 - c_per_domain[dom] + 1e-6)
    dom_lower, dom_upper = 0.5 - k_domain / 2, 0.5 + k_domain / 2
    ax.fill_between(
        [-1, min(N, len(task_ids))],
        dom_lower,
        dom_upper,
        color="#88C5AEFF",
        alpha=0.1,
        label="Theory prediction range",
    )

    ticklabels = []
    
    def x_interval(xmin, xmax, N):
        return np.linspace(xmin, xmax, N)


    for idx, task_id in enumerate(sorted(task_ids)):
        if idx >= N:
            break

        params = task_params_dict[task_id]
        ans = sorted(params["ctrl_ans"])
        normalize = normalize_func(normalization, ans, cutoff)

        try:
            ctrl_med = normalize(params["ctrl_meds"][3:])
            cons_med = normalize(params["cons_meds"][3:])
            final_median = normalize(params["final_median"])
        except Exception as e:
            print(f"Error normalizing task {task_id} in domain {dom}: {e}")
            continue

        # Percentile-length interval implied by c
        k = params["c"] / (1 - params["c"] + 1e-6)

        x = idx

        # Plot theory interval (from predict_task) as a thick vertical segment
        M_minus, M_plus = 1/2 - k / 2, 1/2 + k / 2
        ax.vlines(x, M_minus, M_plus, color="gray", linewidth=5, alpha=0.3)

        ax.plot(
            x_interval(x - 0.5, x, len(ctrl_med)),
            ctrl_med,
            "-",
            color="black",
            alpha=0.7,
            linewidth=0.7,
            label="Control" if idx == 0 else "",
        )

        ax.plot(
            x_interval(x - 0.5, x, len(cons_med)),
            cons_med,
            "-",
            color=palette["Consensus"],
            alpha=0.7,
            linewidth=0.7,
            label="Social influence" if idx == 0 else "",
        )

        # Correct answer marker
        ax.plot(
            x,
            normalize(params["correct"]),
            "*",
            color="#0C031B",
            alpha=0.7,
            markersize=7,
            label="Correct" if idx == 0 else "",
        )

        # Final median marker: hollow if outside domain band
        is_correct = dom_lower <= final_median <= dom_upper
        ax.plot(
            x,
            final_median,
            "o",
            markerfacecolor="none" if not is_correct else palette["Consensus"],
            markeredgecolor=palette["Consensus"],
            alpha=0.7,
            markersize=7,
        )

        # Annotate task c near the top of the interval
        ax.text(
            x,
            min(M_plus, 0.9) + 0.02,
            f"{params['c']:.2f}",
            fontsize=8,
            va="bottom",
            ha="center",
            rotation=45,
        )

        ticklabels.append(str(task_id) if dom != "calories" else get_short_prompt(task_id))

    # Reference line
    if normalization == "perc":
        ax.hlines(
            0.5, -0.5, min(N, len(task_ids)) - 0.5,
            linestyles="--", color="lightgray", linewidth=1, alpha=0.5
        )

    # Axes and labels
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, min(N, len(task_ids)) - 0.5)
    ax.set_ylabel(
        f"Normalized Answer", fontsize=12
    )
    ax.set_xticks(range(min(N, len(task_ids))))
    ax.set_xticklabels(ticklabels[:N], rotation=45, ha="right", fontsize=12)
    # ax.set_xlabel("Task ID" if dom != "calories" else "Prompt")

    ax.legend()
    plt.grid(False)
    plt.tight_layout()

    if plot:
        plt.show()
    if saveto:
        os.makedirs(saveto, exist_ok=True)
        plt.savefig(f"{saveto}", dpi=300, bbox_inches="tight")

    plt.close()


def boxplot_domains(normalization='perc1_1', 
                    plot_final_median=True, plot_correct=False, plot_text=False,
                    saveto=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot([0, 21], [0.5, 0.5], '--', color='lightgray', linewidth=1, alpha=0.5)
    w = 0.5
    cs, stds = [], []
    for (dom, idx) in ordered_domains.items():
        task_ids = df[df['domain_name'] == dom]['task_id'].unique()
        c = c_per_domain[dom]
        k = min(1, c/(1 - c + 1e-6))
        L, U = 1/2 - k/2, 1/2 + k/2
        ax.bar(idx, U - L, bottom=L, facecolor="#5FD2A2FF", edgecolor="#3F8568FF", 
               alpha=0.3, width=w, label='prediction range' if idx == 0 else "")

        meds = []
        for task_id in task_ids:
            params = task_params_dict[task_id]
            ans = sorted(params['ctrl_ans'])
            normalize = normalize_func(normalization, ans)
            try:
                final_median = normalize(params['final_median'])
                correct = normalize(params['correct'])
                # plot final median
                if plot_final_median:
                    ax.scatter(idx - w/2 + np.random.random() * w, final_median, marker='o', 
                            edgecolor=palette['Consensus'], facecolor='none' if not (L <= final_median <= U) else palette['Consensus'],
                            alpha=0.6, s=20)
                # plot correct answer as a black x
                if plot_correct:
                    ax.scatter(idx - w/2 + np.random.random() * w, correct, marker='x', 
                            color='black', alpha=0.6, s=20)
            except Exception as e:
                print(f"Error normalizing task {task_id} in domain {dom}: {e}")
                continue
            meds.append(final_median)

        cs.append(c)
        # stds.append(np.std(meds, ddof=1))
        stds.append(sum((m - 0.5)**2 for m in meds) / len(meds))
        if plot_text:
           ax.text(idx, 1.02, f"{np.std(meds):.2f}", fontsize=8, ha='center', va='bottom')

    if plot_text:
        print("Correlation between domain-level c and median std dev:", np.corrcoef(cs, stds)[0,1])
    ax.set_xticks(list(ordered_domains.values()))
    ax.set_xticklabels([x.replace('_', ' ') for x in ordered_domains.keys()], rotation=45, ha='right')

    ax.set_ylabel(f"Normalized Answer Value")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, len(ordered_domains)+0.5)
    plt.tight_layout()
    if saveto:
        plt.savefig(f"plots/domains/boxplot_domains.png", dpi=300, bbox_inches="tight")

