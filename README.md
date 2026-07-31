# Reproducing the Plots

This repository contains the code used to reproduce the figures in our paper "Social influence and collective wisdom in sequential estimation".

## Data Source

This project uses the dataset introduced by Simoiu et al. (2019). The original data is available at:
[https://github.com/stanford-policylab/wisdom-of-crowds](https://github.com/stanford-policylab/wisdom-of-crowds).


## Dependencies

* Python 3.x
* numpy
* pandas
* scipy
* matplotlib
* seaborn
* jupyter

## Execution Order

### 0. Install requirements
#### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```


### 1. Data preprocessing

```bash
python data_processing.py
```

Loads and merges raw experimental data and produces an intermediate dataset.

### 2. Data cleaning

```bash
python data_cleaning.py
```

Filters invalid responses, removes outliers, and produces the final datasets used in the analysis.

### 3. Plot generation

```bash
jupyter notebook main.ipynb
```

Running all cells reproduces the plots in the paper.

## Code Structure

* `data_processing.py`: merges raw data and computes basic variables
* `data_cleaning.py`: filters and cleans the dataset
* `data_analysis.py`: task- and domain-level statistics from data
* `analysis_utils.py`: helper functions for data analysis
* `general_utils.py`: shared helper functions
* `plots.ipynb`: generates all figures

## References
Simoiu, Camelia, Chiraag Sumanth, Alok Mysore, and Sharad Goel. “Studying the ‘Wisdom of Crowds’ at Scale.” Proceedings of the AAAI Conference on Human Computation and Crowdsourcing, Vol. 7, 2019.
