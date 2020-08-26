import pandas as pd
from keras.callbacks import History, ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import os
import numpy as np
from data_analysis import calculate_metrics
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import dill
from hyper_mining import objective_fn

fspace = {
    'conv1' : hp.quniform('conv1', 32, 96, 8),
    'conv2' : hp.quniform('conv2', 64, 128, 8),
    'conv3' : hp.quniform('conv3', 128, 168, 8),
    'fp' : hp.quniform('fp', 96, 196, 8),
    'dense1' : hp.quniform('dense1',96,512,32),
    'dense2' : hp.quniform('dense2',96,512,32),
    'dense3' : hp.quniform('dense3',64,512,32),
    'dropout_rate' : hp.uniform('dropout_rate',0.1,0.5),
    'lr' : hp.uniform('lr',0.0000001,0.01),
    'n_epochs' : hp.quniform('n_epochs',15,60,5),
    'batch_size' : hp.quniform('batch_size',64,512,32),
    'colsample_bylevel' : hp.uniform('colsample_bylevel', 0.1, 1), 
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.1, 1), 
    'gamma' : hp.uniform('gamma', 0.1, 1), 
    'learning_rate' : hp.uniform('learning_rate', 0.1, 1),
    'max_delta_step' : hp.quniform('max_delta_step',1,10,1),
    'max_depth' : hp.quniform('max_depth',6, 12, 1),
    'min_child_weight' : hp.quniform('min_child_weight',10 ,500 ,5),
    'reg_alpha' : hp.uniform('reg_alpha',0.1,100),
    'reg_lambda' : hp.uniform('reg_lambda',0.1,100),
    'subsample' : hp.uniform('subsample',0.1,1.0),
    'max_bin' : hp.quniform('max_bin',16,256,16)
    #'margin' : hp.uniform('margin',0.2,2)
}

target = 'p38'
base_path = f'C:/Users/tomas/Documents/GitHub/kinase_binding'

data_fpath = base_path+f'/data/{target}/data.csv'
df=pd.read_csv(data_fpath).set_index('biolab_index')

with open(base_path+f'/data/{target}/train_val_folds.pkl', "rb") as in_f:
    train_val_folds = dill.load(in_f)
with open(base_path+f'/data/{target}/train_test_folds.pkl', "rb") as in_f:
    train_test_folds = dill.load(in_f)
    
training_list = [df.loc[train_val_folds[0][0]],
                 df.loc[train_val_folds[1][0]],
                 df.loc[train_val_folds[2][0]],
                 df.loc[train_val_folds[3][0]],
                 df.loc[train_val_folds[4][0]],
                 df.loc[train_val_folds[5][0]],
                 ]
validation_list = [df.loc[train_val_folds[0][1]],
                   df.loc[train_val_folds[1][1]],
                   df.loc[train_val_folds[2][1]],
                   df.loc[train_val_folds[3][1]],
                   df.loc[train_val_folds[4][1]],
                   df.loc[train_val_folds[5][1]],
                   ]

fmin_objective = partial(objective_fn, train_sets = training_list, val_sets = validation_list)
def run_trials():

    trials_step = 4  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("gcn_xgb.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn = fmin_objective, space = fspace, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    print("Best:", best)
    
    # save the trials object
    with open("gcn_xgb.hyperopt", "wb") as f:
        pickle.dump(trials, f)
    return(trials)

trials = run_trials()