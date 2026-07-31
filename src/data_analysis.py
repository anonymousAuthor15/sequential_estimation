import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

from scipy.stats import norm, linregress, skew, gmean
from scipy.stats import wasserstein_distance

import os
import re

from general_utils import *

LOAD = 1  # Set to False to rebuild task_params_df and lambda_fitting_dict from scratch
C_TOLERANCE = 0  # Set to 0 for exact copying, or a positive value for relative tolerance in consensus copying detection

# load experiment data
df = pd.read_pickle('../experiment_data/crowd_full.pkl')    # Ensure the correct path
tasks = pd.read_csv('../experiment_data/original/tasks.csv.zip')  
domains = pd.read_csv('../experiment_data/original/domains.csv.zip')

# simple get functions to retrieve data for a given task_id and condition
def get_prompt(task_id):
    result = tasks.loc[tasks['task_id'] == task_id, 'prompt']
    return result.iloc[0] if not result.empty else None  # Return first match or None if not found

def get_correct_answer(task_id):
    result = tasks.loc[tasks['task_id'] == task_id, 'correct_answer']
    return float(result.iloc[0]) if not result.empty else None  # Return the correct answer or None if not found

def get_answers(task_id, condition='Control', df=df):
    try:
        # get answers as an array
        return df[df['task_id'] == task_id].groupby(['experimental_condition'])['answer'].apply(list).to_dict()[condition]
    except Exception as e:
        print("Wrong task_id, condition or data type", e)
        return None

def domain_func(domain, func, **kwargs):
    """
    Function to apply a given function to all tasks within a specified domain.
    
    @param domain: The domain name for which to apply the function.
    @param func: The function to apply to each task_id within the domain.
    @param kwargs: Additional keyword arguments to pass to the function.
    @return: A dictionary with task_id as keys and the results of the function as values.
    """

    task_ids = df[df['domain_name'] == domain]['task_id'].unique()
    results = {}
    for task_id in task_ids:
        results[int(task_id)] = func(task_id, **kwargs)
    return results

def build_task_params_df(df=df, c_tolerance=0):
    """
    Precompute and store parameters for all tasks into a DataFrame.

    @param df: DataFrame containing the experimental data.
    @param c_tolerance: Tolerance for consensus copying detection (default: 0 for exact copying).
    @return: DataFrame with task_id as index and computed parameters as columns.
    """
    task_array = []
    all_tasks = df['task_id'].unique()

    disp = DisplayProgressBar(len(all_tasks), prefix='Progress:', suffix='Complete', length=50)
    for task_id in all_tasks:
        try:
            disp.update()
            # Store essential values
            params = task_params(task_id, df=df, c_tolerance=c_tolerance)
            task_array.append(params)
        except Exception as e:
            print(f"Error computing parameters for task {task_id}: {e}")
            continue
    
    res = pd.DataFrame(task_array)
    res.set_index('task_id', inplace=True)
    return res


def task_params(task_id, df=df, c_tolerance=0, confidence=0.9):
    """
    Compute parameters for a given task.

    @param task_id: The task ID for which to compute parameters.
    @param df: DataFrame containing the experimental data.
    @param c_tolerance: Tolerance for consensus copying detection (default: 0 for exact copying).
    @param confidence: Confidence level for interval computation (default: 0.9).
    @return: A dictionary with computed parameters.
    """
    control = get_answers(task_id, condition='Control', df=df)
    control_meds = pd.Series(control).expanding().median().values

    consensus_ans = get_answers(task_id, condition='Consensus', df=df)
    consensus_meds = pd.Series(consensus_ans).expanding().median().values

    correct = get_correct_answer(task_id)

    # compute basic stats
    mu, sigma = norm.fit(control)
    std_dev = np.std(control, ddof=1)
    skewness = skew(control)
    
    if c_tolerance == 0:
        c = sum(consensus_ans[i] == consensus_meds[i-1] for i in range(3, len(consensus_ans))) / (len(consensus_ans) - 3)
    else:
        c = sum(abs(consensus_ans[i] - consensus_meds[i-1])/(consensus_meds[i-1]+0.01) <= c_tolerance for i in range(3, len(consensus_ans))) / (len(consensus_ans) - 3)

    k = min(1, c / (1 - c + 1e-5))

    # compute L, U, interval
    try:
        a, b, c1, c2 = find_ms(control, k, confidence=confidence)
    except Exception as e:
        print("Error in find_ms for task", task_id, e)
        raise e
    try:
        if c > 1/3:
            a_p, b_p, c1_p, c2_p = find_ms(control, 2*k-1, confidence=0.9)
        else:
            a_p, b_p, c1_p, c2_p = None, None, None, None
    except Exception as e:
        print("Error in find_ms for dominant interval for task", task_id, e)
        a_p, b_p, c1_p, c2_p = (None, None, None, None)
    
    linear_normalizer = normalize_func('perc', control)

    scores = np.zeros(len(conditions))
    scores_avg = np.zeros(len(conditions))
    scores_gmean = np.zeros(len(conditions))
    scores_abs = np.zeros(len(conditions))
    absolute_error = np.zeros(len(conditions))
    last_upd = np.zeros(len(conditions))
    correct_perc = np.zeros(len(conditions))
    starting_accuracy = np.zeros(len(conditions))
    scores_mode = np.zeros(len(conditions))

    for idx, condition in conditions.items():
        ans = get_answers(task_id, condition=condition)
        meds = pd.Series(ans).expanding().median().values
        mode = np.mean(pd.Series(ans).mode()) if len(pd.Series(ans).mode()) > 0 else np.nan
        # compute scores for social conditions
        # compute starting point accuracy
        [scores[idx], 
         scores_avg[idx], 
         scores_gmean[idx], 
         starting_accuracy[idx],
         scores_mode[idx]] = percentile_score([meds[-1], 
                                               np.mean(ans), 
                                               gmean(np.array(ans)[np.array(ans) > 0]), 
                                               np.median(ans[:3]),
                                               mode], correct, control)

        # compute scores with mse, mae
        scores_abs[idx] = (linear_normalizer(np.median(ans)) - linear_normalizer(correct))
        # compute last median updates
        last_upd[idx] = max((i for i in range(1,len(meds)) if meds[i] != meds[i-1]), default=0)
        # compute percentage of correct answers within 10% of the correct answer
        correct_perc[idx] = sum(abs(x - correct)/(correct + 1e-2) <= 0.001 for x in ans) / len(ans)
        # compute correct perc in terms of std dev
        correct_perc[idx] = sum(abs(x - correct)/(std_dev + 1e-2) <= 0.01 for x in ans) / len(ans)
        # compute absolute error
        absolute_error[idx] = abs(np.median(ans) - correct)

    return {'task_id' : task_id,
            "domain" : domains.loc[domains['domain_id'] == tasks.loc[tasks['task_id'] == task_id, 'domain_id'].values[0], 'domain_name'].values[0],
            "prompt": get_prompt(task_id),

            "cons_ans" : list(consensus_ans),
            "ctrl_ans" : list(control),
            "cons_meds" : list(consensus_meds),
            "ctrl_meds" : list(control_meds),

            "mean" : np.mean(control),
            "gmean" : gmean(np.array(control)[np.array(control) > 0]),
            "median" : np.median(control),
            "ctrl_mode": np.mean(pd.Series(control).mode()) if len(pd.Series(control).mode()) > 0 else np.nan,
            "cons_mode": np.mean(pd.Series(consensus_ans).mode()) if len(pd.Series(consensus_ans).mode()) > 0 else np.nan,
            "variance" : np.var(control),
            "std_dev" : std_dev,
            "mu" : mu, "sigma" : sigma,
            "skewness" : skewness,
            
            "scores" : list(scores),
            "scores_avg" : list(scores_avg),
            "scores_gmean" : list(scores_gmean),
            "scores_abs" : list(scores_abs),
            'scores_mode' : list(scores_mode),
            "starting_accuracy" : list(starting_accuracy),
            "last_median_update": list(last_upd),
            "perc_of_corrects" : list(correct_perc),
            "absolute_error" : list(absolute_error),

            # "kde" : None if not distr else construct_distribution_kde(control, show_distr, correct),

            "c" : c, "k" : k, 
            "final_median" : consensus_meds[-1],
            "correct" : correct,
            "correct_perc" : sum((x - correct)/(correct + 1e-2) <= 0.1 for x in control) / len(control),
            "median_distance" : abs(np.median(control) - correct),
            "mean_distance" : abs(np.mean(control) - correct),

            "L" : a, "U" : b,
            "conf_minus" : c1, "conf_plus" : c2,
            "dom_int": (a_p, b_p, c1_p, c2_p),
            }

def build_task_params(load=True, c_tolerance=0):
    """
    Function to build a DataFrame and dictionary of task parameters for all tasks in the dataset.

    @param load: whether to load precomputed results from disk if available (default: True)
    @param c_tolerance: Tolerance for consensus copying detection (default: 0 for exact copying).
    @return: A tuple containing the task_params_df and task_params_dict.
    """
    if load:
        try:
            task_params_df = pd.read_pickle("data/task_params_df.pkl")
            print("Loaded precomputed task_params from disk.")
            return task_params_df, task_params_df.to_dict(orient='index')
        except Exception as e:
            print("Error loading precomputed task_params_df:", e)
    
    print("Building task_params from scratch...")

    task_params_df = build_task_params_df(df=df, c_tolerance=c_tolerance)
    task_params_dict = task_params_df.to_dict(orient='index')

    task_params_df.to_pickle("data/task_params_df.pkl")
    save_to_json(task_params_dict, "data/task_params")
    return task_params_df, task_params_dict

# build the task_params_df and dict at module load time for fast access in plotting and analysis functions
task_params_df, task_params_dict = build_task_params(load=LOAD, c_tolerance=C_TOLERANCE)

def get_params(task_id, task_params_df=task_params_df):
    try:
        r = task_params_df.loc[task_id]
    except KeyError:
        raise ValueError(f"Task {task_id} not found in precomputed DataFrame.")
    return r.to_dict()

def compute_avg_c_per_domain():
    mu_c_dict = {}
    ranges = {}
    for domain in df['domain_name'].unique():
        # print(f"Domain: {domain}")
        cs = 0
        max_c = 0
        min_c = 1
        for task in df[df['domain_name'] == domain]['task_id'].unique():
            params = task_params_dict[task]
            cs += params['c']
            max_c = max(max_c, params['c'])
            min_c = min(min_c, params['c'])
        avg_c = cs / len(df[df['domain_name'] == domain]['task_id'].unique())
        mu_c_dict[domain] = avg_c
        ranges[domain] = (avg_c - min_c, max_c - avg_c) 
    return mu_c_dict, ranges


# compute average copying probability per domain and order domains by it for consistent coloring and labeling in plots
c_per_domain, _ = compute_avg_c_per_domain()
domains_ordered_by_c = sorted(c_per_domain.keys(), key=lambda x: c_per_domain[x])
ordered_domains = {dom: i+1 for i, dom in enumerate(domains_ordered_by_c)}


# ---- PLOTTING FUNCTIONS ----
def plot_answers_over_time(task_id, conds=conditions.keys(), df=df,
                           ans=True, avg=True, median=True,
                           ax=None):
    """
    Function to plot the evolution of answers over time for a given task and conditions.
    
    :param task_id: id of the task to plot
    :param conds: list of condition indices to include in the plot (default: [0,1,2,3] -> all conditions)
    :param df: DataFrame containing the data to plot (default: global df)
    :param ans: Whether to plot individual answers (default: True)
    :param avg: Whether to plot the average answer over time (default: True)
    :param median: Whether to plot the median answer over time (default: True)
    :param ax: Matplotlib axis to plot on (optional)
    """

    task_data = df[(df['task_id'] == task_id) & (df["experimental_condition"].isin([conditions[i] for i in conds]))].copy()

    # Compute mean and median over time for each condition
    task_data['avg'] = task_data['answer'].expanding().mean()
    task_data['median'] = task_data['answer'].expanding().median()

    if ax is None:
        f, ax = plt.subplots(figsize=(8, 4))

    # plot correct answer line
    correct_answer = task_data['correct_answer'].iloc[0]
    ax.axhline(y=correct_answer, color='red', linestyle='--', alpha=0.7)

    if ans:
        sns.lineplot(data=task_data, x='start_time_index', y='answer', 
                    hue='experimental_condition', palette=palette,
                    marker='o', markersize=4, alpha=0.7, ax=ax)
    if avg:
        sns.lineplot(data=task_data, x='start_time_index', y='avg', 
                     hue='experimental_condition', palette=palette,
                     linestyle='--', legend=False, ax=ax)
    if median:
        sns.lineplot(data=task_data, x='start_time_index', y='median', 
                     hue='experimental_condition', palette=palette,
                     linestyle=':', legend=False, ax=ax)
        
    ax.set_title(task_data['prompt'].iloc[0])
    ax.set_xlabel('Start Time')
    ax.set_ylabel('Answer')
    ax.legend(title='Condition')
    ax.grid(True)
    # plt.show()

def influence_curve_task(task_id, saveto=None):
    """
    Function to plot the influence curve for a given task.
    
    :param task_id: id of the task to plot
    :param saveto: file path to save the plot (optional, if None will show the plot instead)
    """
    params = task_params_dict[task_id]
    consensus_ans = sorted(params['cons_ans'])
    control_ans = sorted(params['ctrl_ans'])
    
    _, f = influence_curve_new(
        params['L'], params['U'],
        (params['conf_minus'], params['conf_plus']),
        eq=params['final_median'],
        x_range=(control_ans[5], control_ans[-5]),
        correct=params['correct'],
        correct_perc=params['correct_perc'],
        medians=get_answers(task_id, 'Consensus', 'median'),
        title=f"[{task_id}] {params['prompt']} {params['c']:.2f}",
        control_medians=(consensus_ans[5], consensus_ans[-5]), 
        median=params['median'],
    )

    if saveto:
        plt.savefig(saveto, dpi=300)
    else:
        plt.show()
    plt.close()

def influence_curve_domain(dom):
    os.makedirs(f"./plots/{dom}", exist_ok=True)
    for task_id in df[df['domain_name'] == dom]['task_id'].unique():
        # print(f"Task ID: {task_id} - Domain: {get_domain_name(task_id)}")
        influence_curve_task(task_id, saveto=f"./plots/{dom}/sic_{task_id}.")


def build_lambda_fitting_dict(load=True):
    """"
    Function to build a dictionary of lambda fitting results for each domain and task, 
    using the wasserstein distance minimization
    
    @param load: whether to load precomputed results from disk if available (default: True)
    """
    if load:
        try:
            lambda_fitting_dict = load_from_file('data/lambda_fitting_per_domain_and_task.json')
            # load keys as int
            lambda_fitting_dict = {domain: {int(k) if k.isdigit() else k: v for k, v in task_dict.items()} for domain, task_dict in lambda_fitting_dict.items()}
            print("Loaded precomputed lambda_fitting_dict from disk.")
            return lambda_fitting_dict
        except Exception as e:
            print("Error loading precomputed lambda_fitting_dict:", e)
    
    lambda_fitting_dict = {}

    print("Building lambda_fitting_dict from scratch...")
    for domain in df['domain_name'].unique():
        task_ids = df[df['domain_name'] == domain]['task_id'].unique()
        lambda_fitting_dict[domain] = {}
        est = estimate_lambda(task_ids)
        lambda_fitting_dict[domain]['wasserstein'] = est['best_lambda']

        for task in task_ids:
            task = int(task)
            task_est = {}
            est_method = estimate_lambda([task])
            task_est['wasserstein'] = est_method['best_lambda']
            lambda_fitting_dict[domain][task] = task_est

    save_to_json(lambda_fitting_dict, 'data/lambda_fitting_per_domain_and_task')
    return lambda_fitting_dict

# lambda fitting utilities
def D_thresh(n1, n2):
    """Kolmogorov-Smirnov D threshold for significance level alpha=0.05"""
    return 1.36 * np.sqrt((n1 + n2) / (n1 * n2))

def compute_cleaned_consensus(task_id):
    """
    Function to compute the cleaned consensus and meds for a given task by removing answers that are copying the previous median.

    """
    consensus = np.array(task_params_dict[task_id]['cons_ans'])
    consensus_meds = np.array(task_params_dict[task_id]['cons_meds'])

    cleaned = consensus[[x for x in range(1, len(consensus)) if consensus[x] != consensus_meds[x-1]]]
    meds = consensus_meds[[x-1 for x in range(1, len(consensus)) if consensus[x] != consensus_meds[x-1]]]

    return {'cleaned_consensus': cleaned, 'cleaned_meds': meds}

def compute_ctrl_lambda(task_id, lamda):
    """
    Function to compute the control answers for a given task and lambda by removing copying answers from the consensus and adjusting the meds accordingly.
    
    :param task_id: id of the task to compute control answers for
    :param lamda: lambda parameter for weighting the meds in the control calculation (0 <= lamda < 1)
    """
    cleaned, meds = compute_cleaned_consensus(task_id).values()
    control_lambda = sorted((cleaned - lamda * meds) / (1 - lamda))
    return control_lambda


def estimate_lambda(task_ids, return_debug=1, plot=False, ax=None):
    """
    Function to estimate the lambda parameter for a given set of task ids by comparing the control answers to the adjusted consensus answers 
    using a specified method (e.g. minimizing Wasserstein distance).
    
    :param task_ids: list of task ids to use for lambda estimation
    :param return_debug: whether to return additional debug information (default: 1, returns dict with best_lambda and intermediate values; if 0, returns only best_lambda)
    :param plot: whether to plot the lambda estimation results (default: False)
    :param ax: matplotlib axis to use for plotting (default: None)
    """
    cleaned = np.zeros(len(task_ids), dtype=object)
    meds = np.zeros(len(task_ids), dtype=object)
    control = np.zeros(len(task_ids), dtype=object)

    for i, task_id in enumerate(task_ids):
        cleaned[i], meds[i] = compute_cleaned_consensus(task_id).values()
        control[i] = sorted(np.array(task_params_dict[task_id]['ctrl_ans']))

    vals = []
    for lamda in np.linspace(0, 1, 100, endpoint=False):
        value = 0
        for i, task_id in enumerate(task_ids):
            control_lamda = sorted((cleaned[i] - lamda * meds[i]) / (1 - lamda))
            res = wasserstein_distance(control[i], control_lamda)
            value += res
        
        vals.append(value)
        if lamda == 0 or value < best_value:
            best_value = value
            best_lambda = lamda

    if plot:
        if ax is None:
            f, ax = plt.subplots(figsize=(8,4))
        ax.plot(np.linspace(0, 1, 100, endpoint=False), vals, color = 'salmon',
                alpha=1, label=f'WASSERSTEIN, λ = {best_lambda:.2f}', lw=3)
        ax.grid(True)
        ax.legend()

    if return_debug:
        return {'best_lambda': best_lambda,
                'cleaned_consensus': cleaned,
                'cleaned_meds': meds
                }
    return best_lambda

lambda_fitting_dict = build_lambda_fitting_dict(load=1)


# ----------------------------------------------
# --------- SIMULATIONS FROM DATASET -----------
# ----------------------------------------------

def configs(task_id, id, N=100, condition='consensus'):
    """
    Function to generate configuration parameters for simulations based on task_id and a specified configuration id.
    """
    domain = task_params_dict[task_id]['domain']
    if id == 0:
        res = {
            'color': "#629EF1",
            'label': 'domain c, no init',
            'args': {'N': N,
                    'c': c_per_domain[domain],
                    'lambda_': lambda_fitting_dict[domain][task_id]['wasserstein'],
                    'init': [None]*3
                    }
        }

    if id == 0.5:
        res = {
            'color': "#FF7300",
            'label': 'domain c, no init, λ=0',
            'args': {'N': N,
                    'c': c_per_domain[domain],
                    'lambda_': 0,
                    'init': [None]*3
                    }
        }

    if id == 1:
        res = {
            'color': '#E06F6F',
            'label': 'no init',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': lambda_fitting_dict[domain][task_id]['wasserstein'],
                    'init': [None]*3
                    }
        }
    if id == 2:
        res = {
            'color': "#CB3797",
            'label': 'no init, λ=0',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': 0,
                    'init': [None]*3
                    }
        }
    if id == 3:
        res = {
            'color': "#6FE0E0",
            'label': 'no init, fixed',
            'args': {'N': 2*N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': lambda_fitting_dict[domain][task_id]['wasserstein'],
                    'init': [None]*3
                    }
        }
    if id == 4:
        res = {
            'color': "#AAC03A",
            'label': 'init consensus',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': lambda_fitting_dict[domain][task_id]['wasserstein'],
                    'init': get_answers(task_id, condition='Consensus')[:3],
                    }
        }
    if id == 5:
        res = {
            'color': "#4ABB79",
            'label': 'init consensus, fixed',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': lambda_fitting_dict[domain][task_id]['wasserstein'],
                    'init': get_answers(task_id, condition='Consensus')[:3]
                    }
        }
    if id == 6:
        res = {
            'color': "#F5A45D",
            'label': 'init consensus, λ=0',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': 0,
                    'init': get_answers(task_id, condition='Consensus')[:3]
                    }
        }
    if id == 7:
        res = {
            'color': "#6337CB",
            'label': 'random init, λ=0',
            'args': {'N': N,
                    'c': task_params_dict[task_id]['c'],
                    'lambda_': 0,
                    'init': 'random'
                    }
        }
    res['args']['condition'] = condition
    return res

def sim_from_data(
    data,
    N=100,
    c=0.2,
    lambda_=0.2,
    init=[None]*3,
    statistic="median",
    condition="consensus",
):
    """
    Simulate the sequential estimation dynamics for a given dataset and parameters.

    :param data: The dataset to simulate from (list or array of values).
    :param N: Number of participants to simulate (default: 100).
    :param c: Copying probability (default: 0.2).
    :param lambda_: Weighting parameter for the influence of the previous median (default: 0.2).
    :param init: Initial values for the first 3 participants (default: [None, None, None], which will be randomly sampled from the data).
    :param statistic: Statistic to compute over time ("median" or "mean", default: "median").
    :param condition: Condition for the simulation ("consensus", "consensus_fix", or "control").
    :return: A dictionary containing the results of the simulation, including the running statistic, votes, and last update time.
    """
    meds = np.zeros(N)
    votes = np.zeros(N)

    if init == 'random':
        init = np.random.uniform(min(data), max(data), size=3)

    heaps = MedianHeaps()

    for t in range(N):
        # Initial 3 participants
        if t < 3:
            vote = get(init[t], np.random.choice(data))

        # After first 3 participants
        else:
            r = np.random.choice(data)

            if condition == "consensus" or condition == "consensus_fix":
                # current social influence rule
                vote = choice(
                    [meds[t-1], lambda_ * meds[t-1] + (1 - lambda_) * r],
                    p=[c, 1 - c]
                )
                if condition == "consensus_fix" and vote == meds[t-1]:
                    while vote == meds[t-1]:
                        r = np.random.choice(data)
                        vote = choice(
                            [meds[t-1], lambda_ * meds[t-1] + (1 - lambda_) * r],
                            p=[c, 1 - c]
                        )

            elif condition == "control":
                # no social information at all:
                # participant simply answers from data-generating process
                vote = r

            else:
                raise ValueError("condition must be 'consensus', 'consensus_fix', or 'control'")

        votes[t] = vote

        # Update running statistic
        if statistic == "median":
            heaps.add_number(votes[t])
            meds[t] = heaps.get_median()
        else:
            meds[t] = votes[t] if t == 0 else (meds[t-1] * t + votes[t]) / (t + 1)

    return {
        'results': meds,
        'votes': votes,
        'last_update': last_update(meds)
    }


def sims_for_last_update(task_id, N=100, num_simulations=1000, 
                         domain=True, seed=123,
                         lambda_fitting_dict=lambda_fitting_dict, lambda_zero=False):
    """
    Function to run multiple simulations for a given task and compute the distribution of the last update time (the last time the median changes) across simulations
    
    :param task_id: id of the task to simulate
    :param N: Number of users to simulate in each simulation
    :param num_simulations: Number of simulations to run
    :param domain: Whether to use domain-level parameters (True) or task-level parameters (False)
    :param seed: Random seed for reproducibility (default: 123)
    :param lambda_fitting_dict: dictionary containing the fitted lambda values for each domain and task, used to set the lambda parameter for the simulations based on the task's domain
    :param lambda_zero: Whether to set lambda to zero in the simulations
    :return: A dictionary containing the average last update time across simulations
    """
    np.random.seed(seed)
    
    params = task_params_dict[task_id]
    domain_name = params['domain']
    res = 0

    for _ in range(num_simulations):
        sim_res = sim_from_data(
            data=params['ctrl_ans'],
            N=N,
            c=c_per_domain[domain_name] if domain else params['c'],
            lambda_= 0 if lambda_zero else lambda_fitting_dict[domain_name][task_id]['wasserstein'],
        )
        res += sim_res['last_update']
        # last_updates[sim_res['last_update']] += 1
    
    return {'avg_last_update': res / num_simulations}

last_update_dict = {}
try:
    with open('data/last_update_results.json', 'r') as f:
        last_update_dict_str = json.load(f)
        last_update_dict = {int(k): v for k, v in last_update_dict_str.items()}
        print("Loaded simulated last update results from disk.")
except Exception as e:
    print("Running simulations for last update times...")
    NUM_SIMULATIONS = 200
    disp = DisplayProgressBar(total=len(df['task_id'].unique()), prefix='Progress:', suffix='Complete', length=50)
    for task in df['task_id'].unique():
        disp.update()
        last_update_dict[int(task)] = {
            'lambda_domain': sims_for_last_update(task, num_simulations=NUM_SIMULATIONS)['avg_last_update'],
            'lambda_zero': sims_for_last_update(task, num_simulations=NUM_SIMULATIONS, lambda_zero=True)['avg_last_update']
        }
    save_to_json(last_update_dict, 'data/last_update_results')