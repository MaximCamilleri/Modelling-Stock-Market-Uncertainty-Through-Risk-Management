# Modelling Stock Market Uncertainty Through Risk Management

This repository contains the experimental code and data used for the ACM paper:

> Modelling Stock Market Uncertainty Through Risk Management.  
> DOI: [10.1145/3677052.3698652](https://dl.acm.org/doi/10.1145/3677052.3698652)

The project studies reinforcement-learning portfolio management on Dow 30 equities with explicit market-risk controls. It combines custom stock-trading Gym environments, rolling-window retraining, turbulence-based risk management, and value-penalized replay buffers for DDPG/TD3-style algorithms.

## Repository Layout

```text
Algorithms/
  TD4/        Distributional TD3/TD4 implementation and policy classes.
  VPTD3/     Value-penalized TD3 with a turbulence-aware replay buffer.
  VPTD4/     Value-penalized TD4 with distributional critics.
  VPDDPG/    Value-penalized DDPG implementations and replay buffers.

Data/
  dow_30_2009_2020.csv        Raw Dow 30 market data.
  ETF_SPY_2009_2020.csv       SPY ETF data.
  ^DJI.csv                    Dow Jones Industrial Average benchmark data.
  dow30_turbulence_index.csv  Precomputed Dow 30 turbulence index.
  done_data.csv               Preprocessed Dow 30 data.
  done_data_pca90.csv         PCA-transformed preprocessed data.
  data_analysis.ipynb         Data exploration/preparation notebook.

FunctionFiles/
  config/     Data and output path configuration.
  env/        Training, validation, trading, and PCA trading environments.
  pipeline/   Preprocessing, rolling training, hyperparameter search, and backtesting helpers.

Experiment 1/Exp 0.ipynb  Baseline DDPG and turbulence-threshold experiments.
Experiment 2/Exp 1.ipynb  Value-penalized DDPG experiments.
Experiment 3/Exp 2.ipynb  TD3/TD4 and value-penalized variants.
```

## Method Summary

The environments model a 30-stock portfolio with:

- `1,000,000` initial cash balance.
- Continuous actions in `[-1, 1]` for each stock, scaled to a maximum of 100 shares per trade.
- A `0.001` transaction fee.
- State vectors containing cash, prices, current holdings, and market features.
- A turbulence threshold that stops buying and liquidates positions during high-turbulence periods.

The preprocessing pipeline builds adjusted OHLCV features, adds technical indicators (`macd`, `rsi_30`, `cci_30`, `dx_30`), and computes a Dow 30 turbulence index. PCA-based environments consume reduced feature sets from `done_data_pca90.csv`.

The rolling training workflow uses:

- Training data starting in 2009.
- Trading/evaluation dates after October 2015 through mid-2020.
- 63-day rebalance windows.
- 63-day validation windows.
- A historical turbulence quantile, commonly `0.90`, as the risk threshold.

Backtesting compares generated account-value series against the Dow Jones benchmark and reports annual return, cumulative return, annual volatility, Sharpe ratio, Sortino ratio, and max drawdown.

## Setup

This code was developed as a research notebook repository. There is no pinned `requirements.txt`, and the experiments mix older `stable-baselines` APIs with newer `stable-baselines3` code. Use isolated environments.

Recommended starting point:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-learn stockstats ta tabulate pyfolio empyrical optuna joblib gym gymnasium torch stable-baselines3 jupyter
```

For notebooks that import `stable_baselines` rather than `stable_baselines3`, use a Python 3.7-compatible environment and install the legacy package:

```bash
pip install stable-baselines
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Running the Experiments

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then run the notebooks in order:

1. `Data/data_analysis.ipynb` if you need to regenerate or inspect preprocessed datasets.
2. `Experiment 1/Exp 0.ipynb` for baseline DDPG and turbulence-threshold experiments.
3. `Experiment 2/Exp 1.ipynb` for value-penalized DDPG experiments.
4. `Experiment 3/Exp 2.ipynb` for TD3/TD4 and value-penalized TD3/TD4 experiments.

Many cells expect output folders such as `working_files/`, `trained_models/`, `tensorboard_log/`, and `Studies/` to exist relative to the active notebook directory. Create them before long runs if the notebook does not create them automatically.

## Outputs

Training and trading runs write intermediate artifacts such as:

- `working_files/account_value_train.csv`
- `working_files/account_value_trade_<iteration>_<model>.csv`
- `working_files/account_rewards_trade_<iteration>_<model>.csv`
- `working_files/action_history_<model>_<iteration>.csv`
- `working_files/last_state_<model>_<iteration>.csv`
- model checkpoints under `trained_models/`
- Optuna studies under `Studies/`

The backtesting helper in `FunctionFiles/pipeline/backtesting_pipeline.py` aggregates these account-value files and prints summary performance statistics.

## Important Notes

- Paths in several notebooks and helper files are relative and case-sensitive in places (`Data/` vs `data/`). Run notebooks from their original experiment folders or adjust paths before execution.
- Generated outputs, trained models, and TensorBoard logs are not included in the repository.
- The notebooks can be computationally expensive because they retrain models over rolling windows and multiple runs.
- The code is intended for research reproduction and analysis, not live trading.

## Citation

If you use this repository, cite the associated paper:

```bibtex
@inproceedings{stock_market_uncertainty_risk_management,
  title     = {Modelling Stock Market Uncertainty Through Risk Management},
  doi       = {10.1145/3677052.3698652},
  publisher = {Association for Computing Machinery},
  url       = {https://dl.acm.org/doi/10.1145/3677052.3698652}
}
```
