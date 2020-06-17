import pandas as pd
import os
import numpy as np
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, \
    auc, average_precision_score, pairwise_distances
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import xgboost as xgb
from hyperopt import STATUS_OK


class XGB_hyper(object):
    
    def __init__(self,  model_params):
        self.model_params = model_params
        
    def build_model(self,dmatrix,evalist,verbosity,num_round):
        model = xgb.train(self.model_params,dmatrix,num_round,evalist,verbose_eval = verbosity)
        
        
        return model
    
    def to_xgb_input(self,input_data):
        smi = input_data.rdkit
        mols = [Chem.MolFromSmiles(smi) for smi in smi]
        ECFP = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in mols]
        a = np.array(ECFP)
        ECFP_smiles = a.astype(np.float32)
        binary = input_data.Binary
        dmatrix = xgb.DMatrix(data = ECFP_smiles,label = binary)
        return(dmatrix,binary)
    

def objective(fspace, train_sets, val_sets):
    maps = []
    model_params = {
        "colsample_bylevel" : fspace['colsample_bylevel'],
        "colsample_bytree" : fspace['colsample_bytree'],
        "gamma" : fspace['gamma'],
        "eta" : fspace['learning_rate'],
        "max_delta_step" : int(fspace['max_delta_step']),
        "max_depth" : int(fspace['max_depth']),
        "min_child_weight" : int(fspace['min_child_weight']),
        "alpha" : fspace['reg_alpha'],
        "lambda" : fspace['reg_lambda'],
        "subsample" : fspace['subsample'],
        "eval_metric":'auc',
        "objective":'binary:logistic',
        "booster":'gbtree',
        "tree_method" : 'gpu_hist',
        "single_precision_histogram" : True
        }
    class_xgb = XGB_hyper(model_params)
    for i in range(len(train_sets)):
        dmatrix_val,val_labels = class_xgb.to_xgb_input(val_sets[i])
        dmatrix_train,train_labels = class_xgb.to_xgb_input(train_sets[i])
        evalist = [(dmatrix_val,'eval'),(dmatrix_train,'train')]
        xgb_model = class_xgb.build_model(dmatrix_train,evalist,False,150)
        y_pred_val = xgb_model.predict(dmatrix_val)
        maps.append(average_precision_score(val_labels, y_pred_val))
    ave_map = np.mean(maps,axis = 0)
    return {'loss': -ave_map ,  'status': STATUS_OK}
        