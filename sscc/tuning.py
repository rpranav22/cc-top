import argparse
import os
import tempfile
from optuna import samplers
import yaml
import pdb
import optuna
import mlflow
from pandas import DataFrame
from optuna import Trial
import plotly


def convert_to_best_config(config: dict, best_params: dict):
    best_config = config.copy()
    for section, params in best_config.items():
        if section == 'search_space':
            continue

        for hp_name in best_config[section].keys():
            if hp_name in best_params.keys():
                config[section][hp_name] = best_params[hp_name]

    return best_config


def retrieve_grid_search_space(config: dict):
    """Retrieve gird search space from search space with 
        'mode' == 'grid' in a dict that the optuna Grid Sampler 
        can process
    """
    sp = config['search_space']
    search_space_dict = {}
    for hp_name, values in sp.items():
        if hp_name.startswith('grid'):
            search_space_key = hp_name.split('rid_')[1]
            search_space_values = sp[hp_name]['values']
            search_space_dict[search_space_key] = search_space_values
    return search_space_dict

def convert_to_tuner_config(config: dict, trial: Trial):
    search_space = config['search_space']
    config = config.copy()

    # Find and replace the hyper parameters defined in search_space
    for section, params in config.items():
        if section == 'search_space':
            continue

        for hp_name in config[section].keys():
            if hp_name in search_space.keys():
                tuner_hp_type = search_space[hp_name]['type']
                if tuner_hp_type == 'categorical':
                    config[section][hp_name] = trial.suggest_categorical(hp_name,
                                                                         choices=search_space[hp_name]['choices'])
                    continue
                if tuner_hp_type == 'grid':
                    config[section][hp_name] = trial.suggest_categorical(hp_name,
                                                                         choices=search_space[hp_name]['choices'])
                    continue
                low = search_space[hp_name]['low']
                high = search_space[hp_name]['high']
                if tuner_hp_type == 'float':
                    config[section][hp_name] = trial.suggest_float(hp_name, low=low, high=high,
                                                                   step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'int':
                    config[section][hp_name] = trial.suggest_int(hp_name, low=low, high=high,
                                                                 step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'log':
                    config[section][hp_name] = trial.suggest_loguniform(hp_name, low=low, high=high)
                else:
                    raise ValueError("Accepted hpar types are: int, float, log, categorical")
    return config


def log_optuna_plots_in_mlflow(study):
    """store optuna plots and log them in mlflow
    """
    with tempfile.TemporaryDirectory() as tmp_dir:

        plot = optuna.visualization.plot_slice(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/sliceplot.png')
        mlflow.log_artifact(f'{tmp_dir}/sliceplot.png')

        plot = optuna.visualization.plot_intermediate_values(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/interm_values.png')
        mlflow.log_artifact(f'{tmp_dir}/interm_values.png')

        plot = optuna.visualization.plot_optimization_history(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/history.png')
        mlflow.log_artifact(f'{tmp_dir}/history.png')

        plot = optuna.visualization.plot_contour(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/contour.png')
        mlflow.log_artifact(f'{tmp_dir}/contour.png')

        plot = optuna.visualization.plot_param_importances(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/importances.png')
        mlflow.log_artifact(f'{tmp_dir}/importances.png')


def save_as_csv_file_in_mlflow(data: DataFrame, filename: str):
    """save csv file
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        data.to_csv(f'{tmp_dir}/{filename}')
        mlflow.log_artifact(f'{tmp_dir}/{filename}')


def save_as_yaml_file_in_mlflow(data: dict, filename: str):

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        mlflow.log_artifact(path)


def get_or_create_experiment_id(experiment_name: str) -> int:
    """
    Lookup experiment name in mlflow database. If the experiment name
    does not exist then it will create a new experiment.

    Args:
        experiment_name:

    Returns:
        mlflow experiment id
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    elif experiment.lifecycle_stage == 'deleted':
        print("The experiment name exists in the .trash. Delete .trash or come up with other name.")

    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def get_experiment_id(config=None, args=None):
    """Given parsed arguments this method will look for
    the `mlflow_name` and `mlflow_id` arguments.
    Either a config file OR an args object should be passed.
    We use configs in the modeling files but need the args functionality for the postprocessing

    Args:
        config: A dict of configurations
        args: Command line args as read by arg parse

    Returns:
        A mlflow experiment id
    """
    if args is not None:
        if not 'mlflow_name' in args or not 'mlflow_id' in args:
            raise KeyError("The args need to have a mlflow_name and a mlflow_id")

        if args.mlflow_name:
            experiment_id = get_or_create_experiment_id(args.mlflow_name)

        elif args.mlflow_id:
            experiment_id = args.mlflow_id

        else:
            # Neither a mlflow name or id is given. We create a default name.
            default_experiment_name = "default_experiment"
            experiment_id = get_or_create_experiment_id(default_experiment_name)

        return experiment_id
    elif config is not None:
        params = config['exp_params']
        if not 'mlflow_name' in params or not 'mlflow_id' in params:
            raise KeyError("The args need to have a mlflow_name and a mlflow_id")

        if params['mlflow_name']:
            experiment_id = get_or_create_experiment_id(params['mlflow_name'])

        elif params['mlflow_id'] and params['mlflow_id'] > 1:
            experiment_id = params['mlflow_id']

        else:
            # Neither a mlflow name or id is given. We create a default name.
            default_experiment_name = "default_experiment"
            experiment_id = get_or_create_experiment_id(default_experiment_name)

    return experiment_id


def save_history_to_mlflow(history: DataFrame):
    """
    Logs all metrics in a history DataFrame per step in mlflow
    and saves the DataFrame as a .csv file artifact in mlflow.
    Args:
        history:
            pandas.DataFrame
    """
    assert 'step' in history, "To save a history in mlflow a `step` column is needed"

    for index, row in history.iterrows():
        mlflow.log_metrics(row.drop('step').to_dict(), step=int(row['step']))

    with tempfile.TemporaryDirectory() as tmp_dir:
        history_path = os.path.join(tmp_dir, "history.csv")
        history.to_csv(history_path)
        mlflow.log_artifact(history_path)