"""
230722作成
"""

import sys, os
os.environ.setdefault("TOOLS_DIR", "/workspace")

sys.path += [os.environ["TOOLS_DIR"]]
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import optuna
from tqdm import tqdm
from addict import Dict
from tools.notice import noticeerror
from tools.notice import notice as notice_
from tools.logger import default_logger
from tools.args import load_config2, clip_config
from copy import deepcopy
from models.dataset import get_dataset

optuna_loglevels = {
    'debug': optuna.logging.DEBUG,
    'info': optuna.logging.INFO,
    'warning': optuna.logging.WARNING,
    'error': optuna.logging.ERROR,
    'critical': optuna.logging.CRITICAL,
}

suggest_type2func = {
    'categorical': lambda trial, name, choices, **args, : trial.suggest_categorical(name, choices),
    'discrete_uniform': lambda trial, name, low, high, q, **args : trial.suggest_discrete_uniform(name, low, high, q),
    'float': lambda trial, name, low, high, step=None, log=False, **args: trial.suggest_float(name, low, high, step=step, log=log),
    'int': lambda trial, name, low, high, step=1, log=False, **args: trial.suggest_int(name, low, high, step, log),
    'loguniform': lambda trial, name, low, high, **args: trial.suggest_loguniform(name, low, high),
    'uniform': lambda trial, name, low, high, **args: trial.suggest_uniform(name, low, high),    
}

def roc_auc_score2(y_true, y_score):
    if np.all(y_true == y_true[0]):
        return np.nan
    else:
        return roc_auc_score(y_true=y_true, y_score=y_score)
def average_precision_score2(y_true, y_score):
    if np.all(y_true == y_true[0]):
        return np.nan
    else:
        return average_precision_score(y_true=y_true, y_score=y_score)

metric_type2class = {
    'RMSE': lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)),
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'R^2': r2_score,
    'AUROC': roc_auc_score2,
    'AUPR': average_precision_score2,
}
metric2direction = {
    'RMSE': 'minimize', 'MAE': 'minimize', 'R^2':'maximize',
    'AUROC': 'maximize', 'AUPR': 'maximize'
}

class XGBRegressor:
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)
    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
    def predict(self, X):
        return self.model.predict(X)
class XGBClassifier:
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)
    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
class LinearRegression_:
    def __init__(self, **params):
        self.model = LinearRegression(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
class LogisticRegression_:
    def __init__(self, **params):
        self.model = LogisticRegression(**params)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

model_type2class = {
    'XGBClassifier': XGBClassifier,
    'XGBRegressor': XGBRegressor,
    'LinaerRegression': LinearRegression_,
    'LogisticRegression': LogisticRegression_
}

@noticeerror(from_="donwstream", notice_end=False)
def main(config, type, model, params, objective, n_trial,
    dfs, data, result_dir, 
    path_best_params=None, notice=False,
    log_file='debug', log_stream='info', log_optuna='warning', show_tqdm=True):
    """
    valを指定せず, trainとtestのみでスコアを計算することも可能。

    Parameters
    ----------
    <mconfigs内で指定>
    type(str): 'reg' or 'class'
    model(str): One of key in model_type2class
    params:
      <param name>: dict or str, float, ...
        dict -> Parameter is optunized
        str, float, ... -> Parameter is fixed at that value.

    <自分で指定>
    objective(str): Objective of optuna. Either RMSE, MAE, R^2, AUROC, AUPR
    n_trial(int): Number of trial in optuna.
    dfs: dataframes for train_data and val_data
        <df_name>(dict): Input for pd.read_csv
    data:
      train:
        input: Input to get_dataset
        target: Input to get_dataset
      val(optional):
        input: Input to get_dataset
        target: Input to get_dataset
      test(optional):
        input: Input to get_dataset
        target: Input to get_dataset
    result_dir(str): Path of directory to save results.
    path_best_params(str): Path to save best params. It defaults to f"{result_dir}/best_params.tsv"
    log_file(str): logging level to log file.
    log_stream(str): logging level to stdout.
    log_optuna(str): logging level of optuna.
    notice(bool)
    """
    # Make logger
    os.makedirs(result_dir, exist_ok=True)
    logger = default_logger(filename=f"{result_dir}/log.txt", file_level=log_file, stream_level=log_stream)
    optuna.logging.set_verbosity(optuna_loglevels[log_optuna])

    # save param
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    # Load or make df_best_param
    default_params = {}
    var_params = []
    for name, pconfig in params.items():
        if isinstance(pconfig.to_dict(), dict):
            var_params.append(name)
        else:
            default_params[name] = pconfig
    if type == 'reg': metrics = ['RMSE', 'MAE', 'R^2']
    else: metrics = ['AUROC', 'AUPR']
    model_class = model_type2class[model]
    
    # prepare X, y
    dfs_config = dfs
    dfs = {}
    for key, config in dfs_config.items():
        dfs[key] = pd.read_csv(**config)

    # get data config
    dconfigs = Dict()
    config = Dict()
    for key, config0 in data.items():
        config.update(config0)
        dconfigs[key] = Dict(deepcopy(config.to_dict()))

    ## training data
    input_train = get_dataset(logger=logger, name='input_train', dfs=dfs, **dconfigs['train'].input).array
    target_train = get_dataset(logger=logger, name='target_train', dfs=dfs, **dconfigs['train'].target).array
    if 'mask' in dconfigs['train']:
        mask = get_dataset(logger=logger, name='mask_train', dfs=dfs, **dconfigs['train'].mask).array
        input_train = input_train[mask]
        target_train = target_train[mask]
    
    ## validation data
    if 'val' in dconfigs:
        input_val = get_dataset(logger=logger, name='input_val', dfs=dfs, **dconfigs['val'].input).array
        target_val = get_dataset(logger=logger, name='target_val', dfs=dfs, **dconfigs['val'].target).array
        if 'mask' in dconfigs['val']:
            mask = get_dataset(logger=logger, name='mask_val', dfs=dfs, **dconfigs['val'].mask).array
            input_val = input_val[mask]
            target_val = target_val[mask]
    else:
        input_val = target_val = None
    
    
    os.makedirs(result_dir, exist_ok=True)

    # optunize params
    best_params = None
    param_configs = params
    if path_best_params is None:
        path_best_params = f"{result_dir}/best_params.tsv"
    if os.path.exists(path_best_params):
        logger.info(f"Cached best params were used.")
        df_best_params = pd.read_csv(path_best_params, index_col=0, sep='\t', header=None)[1]
        best_params = {col: eval(params[col].type)(df_best_params[col])
            for col in df_best_params.index}
    elif len(params) == 0:
        logger.info(f"No params to optunize exists")
        best_params = {}
    else:
        if input_val is None:
            logger.info("Validation set not defined. Not optunized.")
        elif np.all(target_val == target_val[0]):
            logger.warning("Only 1 class present in y_true in validation set. Not optumized.")
        elif np.all(target_train == target_train[0]):
            logger.warning("Only 1 class present in y_true in train set. Not optumized.")
        else:
            logger.info(f"Optunizing...")
            metric = metric_type2class[objective]
            direction = metric2direction[objective]
            if show_tqdm:
                pbar = tqdm(total=n_trial, desc="Trials")
            def objective(trial):
                params = {}
                for name, pconfig in param_configs.items():
                    if isinstance(pconfig, dict):
                        param_type = eval(pconfig.type)
                        params[name] = param_type(suggest_type2func[pconfig.suggest_type](trial, name=name, **pconfig))
                    else:
                        params[name] = pconfig
                model = model_class(**params)
                model.fit(input_train, target_train)
                pred_val = model.predict(input_val)
                if show_tqdm:
                    pbar.update()
                return metric(target_val, pred_val)
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trial)
            best_params = study.best_params
            df_best_params = pd.Series(dtype=object)
            for key, value in best_params.items():
                df_best_params[key] = value
            os.makedirs(os.path.dirname(path_best_params), exist_ok=True)
            df_best_params.to_csv(path_best_params, header=False, sep='\t')

    # calculate scores
    path_df_pred = f"{result_dir}/preds.csv"
    path_model = f"{result_dir}/model.pkl"
    path_train_score = f"{result_dir}/train_scores.tsv"
    path_score = f"{result_dir}/scores.tsv"
    if os.path.exists(path_model) and \
        ('test' not in dconfigs or (os.path.exists(path_df_pred) and os.path.exists(path_score))) and \
        os.path.exists(path_train_score):
        logger.info(f"All results were already calculated.")
    elif best_params is None:
        logger.warning(f"Best params is not defined, score calculation passed.")
    else:
        if 'val' in dconfigs:
            input = np.concatenate([input_train, input_val], axis=0)
            target = np.concatenate([target_train, target_val], axis=0)
        else:
            input, target = input_train, target_train
        model = model_class(**best_params, **default_params)
        model.fit(input, target)
        with open(path_model, 'wb') as f:
            pickle.dump(model, f)
        pred = model.predict(input)
        if np.all(target == target[0]):
            logger.info("Only one class exists in y_true in train & validation set.")
        else:
            df_train_scores = pd.Series(dtype=float)
            for metric in metrics:
                df_train_scores[metric] = metric_type2class[metric](target, pred)
            df_train_scores.to_csv(path_train_score, sep='\t', header=False)
        if 'test' in dconfigs:
            input_test = get_dataset(logger=logger, name='target_test', dfs=dfs, **dconfigs['test'].input).array
            pred_test = model.predict(input_test)
            if 'mask' in dconfigs['test']:
                mask = get_dataset(logger=logger, name='mask_test', dfs=dfs, **dconfigs['test'].mask).array
                mu_pred = np.mean(pred_test[mask])
                pred_test[~mask] = mu_pred

            pd.DataFrame({'pred': pred_test}).to_csv(path_df_pred, index=False)
            
            if 'target' in dconfigs['test']:
                target_test = get_dataset(logger=logger, name='target', dfs=dfs, **dconfigs['test'].target).array
                if np.all(target_test == target_test[0]):
                    logger.info("Only one class exists in y_true in test set.")
                else:
                    df_scores = pd.Series(dtype=float)
                    for metric in metrics:
                        df_scores[metric] = metric_type2class[metric](target_test, pred_test)
                    df_scores.to_csv(path_score, sep='\t', header=False)
    if notice:
        notice_(f"Downstream finished!")   
         
if __name__ == "__main__":
    config = load_config2("", [])
    if 'variables' in config:
        del config['variables']
    main(config, **config)