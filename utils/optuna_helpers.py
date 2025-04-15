import glob

import optuna
import optuna.visualization as vis

from utils.logging_utils import print_block


def sample_hyperparams(trial, hyperparams):
    """
    Samples hyperparameters for optuna trial
    :param trial: optuna trial
    :param hyperparams: dict of hyperparameter specifications
    :return: dict
    """
    sampled = {}
    betas = []
    for param, spec in hyperparams.items():
        if 'default' in spec:
            sampled[param] = spec.get('default')

        elif param in ['betas1', 'betas2']:
            beta_val = trial.suggest_float(param, spec['low'], spec['high'], log=True if param == 'betas1' else False)
            betas.append(beta_val)
            if param == 'betas2':
                sampled['betas'] = (betas[0], betas[1])

        elif spec['type'] == 'float':
            sampled[param] = trial.suggest_float(param, spec['low'], spec['high'], log=spec['log'] if 'log' in spec else False)
        elif spec['type'] == 'int':
            sampled[param] = trial.suggest_int(param, spec['low'], spec['high'], log=spec['log'] if 'log' in spec else False)
        elif spec['type'] == 'categorical':
            sampled[param] = trial.suggest_categorical(param, spec['choices'])
        elif spec['type'] == 'bool':
            sampled[param] = trial.suggest_categorical(param, [True, False])

    return sampled


def print_best_optuna(db=None, pareto=False, single_pareto=None):
    """
    Print best trial from optuna study, if db is none, finds all studies in cwd and prints them
    :param db: str, file name of study storage
    :param pareto: bool, if True prints pareto optimal trials
    :param single_pareto: int, index of objective to minimize from pareto trials
    :return: None
    """

    def detail_trial(trial):
        print(f"Trial {trial.number}:")
        for i, value in enumerate(trial.values):
            print(f"  Objective {i + 1} Value: {value:.5g}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 30)
        return None

    if db:
        study_name = db[:-3]
        storage_name = f"sqlite:///{db}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            if pareto or single_pareto is not None:
                pareto_front = study.best_trials
                print_block(f"\nStudy name: {study_name}")
                if single_pareto is not None:
                    trial = min(pareto_front, key=lambda t: t.values[single_pareto])
                    print(f"")
                    detail_trial(trial)
                    return None
                for trial in pareto_front:
                    detail_trial(trial)
            else:
                best_trial = study.best_trial
                print(f"\nStudy name: {study_name}")
                print(f"Best trial value: {best_trial.value}")
                print("Best trial parameters:")
                for key, value in best_trial.params.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print_block(f"Could not load study {study_name}. Exception: {e}", err=True)
        return None
    else:
        studies = glob.glob('*study.db*')
        for study in studies:
            print_best_optuna(study, pareto=pareto, single_pareto=single_pareto)
        return None


def plot_pareto(db, vals=(0, 1, 2), dom=False, plot_param_importances=False, full=False, single=None):
    """
    Plot pareto front of optuna study
    :param db: str, name of database
    :param vals: array-like (len 2 or 3), indicies of objs to plot, any combo of (0,1,2), defaults to all 3
    :param dom: bool, whether to include dominated trials or not
    :param plot_param_importances: bool, if True plots parameter importances
    :param full: bool, if True does full analysis: plots param importances, plots and prints pareto front
    :param single: int, index of singular objective to minimize, overwrites all other options
    :return: None
    """
    study_name = db[:-3]
    storage_name = f"sqlite:///{db}"
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    def targs(t):
        l = []
        for v in vals:
            l.append(t.values[v])
        return tuple(l)

    if single is not None:
        print_best_optuna(db, single_pareto=single)
        return None
    elif full:
        fig1 = vis.plot_param_importances(study)
        fig1.show()
        fig2 = vis.plot_pareto_front(study, targets=targs, include_dominated_trials=dom)
        fig2.show()
        print_best_optuna(db, pareto=True)
        return None
    elif plot_param_importances:
        fig = vis.plot_param_importances(study)
        fig.show()
    else:
        fig = vis.plot_pareto_front(study, targets=targs, include_dominated_trials=dom)
        fig.show()
    return None
