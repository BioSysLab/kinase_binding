import pandas as pd
from keras.callbacks import History, ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import os
import numpy as np
from data_analysis import calculate_metrics, load_weights_and_evaluate
from model_builders import GCN_pretraining
from hyperparameter_tuning_GCN import objective
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle


fspace = {
    'conv1' : hp.quniform('conv1', 32, 96, 8),
    'conv2' : hp.quniform('conv2', 48, 128, 8),
    'conv3' : hp.quniform('conv3', 64, 168, 8),
    'fp' : hp.quniform('fp', 64, 196, 8),
    'dense1' : hp.quniform('dense1',96,256,32),
    'dense2' : hp.quniform('dense2',96,256,32),
    'dense3' : hp.quniform('dense3',48,128,32),
    'dropout_rate' : hp.uniform('dropout_rate',0.1,0.5),
    'lr' : hp.uniform('lr',0.0005,0.01),
    'n_epochs' : hp.quniform('n_epochs',15,70,5)
}

base_path = 'C:/Users/user/Documents/kinase_binding/learning/'
target = 'p38'
model_name = 'gcn_pretraining_ensemble'

# no need for manual changes in this cell
train_files = os.path.join(base_path, f'data/{target}/split_aveb/fold_{{}}/train_{{}}.csv')
val_files = os.path.join(base_path, f'data/{target}/split_aveb/fold_{{}}/val_{{}}.csv')
weight_files = os.path.join(base_path, f'results/{target}/{model_name}/fold_{{}}/model_weights/model_{{}}.h5')

train_sets = [pd.read_csv(train_files.format(i,i), index_col = 0) if 'Unnamed: 0' in pd.read_csv(train_files.format(i,i)) else pd.read_csv(train_files.format(i,i)) for i in range(7)]
val_sets = [pd.read_csv(val_files.format(i,i), index_col = 0) if 'Unnamed: 0' in pd.read_csv(val_files.format(i,i)) else pd.read_csv(val_files.format(i,i)) for i in range(7)]
test_set = pd.read_csv(os.path.join(base_path, f'data/{target}/split_aveb/test.csv'))

fmin_objective = partial(objective, train_sets = train_sets, val_sets = val_sets)

def run_trials():

    trials_step = 100  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 2  # initial max_trials. put something small to not have to wait


    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("gcn.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn = fmin_objective, space = fspace, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    print("Best:", best)

    # save the trials object
    with open("gcn.hyperopt", "wb") as f:
        pickle.dump(trials, f)
    return(trials)

trials = run_trials()
