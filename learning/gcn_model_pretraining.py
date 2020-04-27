from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import keras
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
import keras.backend as K
from keras.models import model_from_json
from keras.models import load_model, Model
from tempfile import TemporaryFile
from keras import layers
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
# matplotlib inline
from keras.callbacks import History, ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout, Layer
from keras.initializers import glorot_normal
from keras.regularizers import l2
from functools import partial
from multiprocessing import cpu_count, Pool
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from math import ceil
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from keras.layers import Conv1D

from custom_layers.model_creator import multistage_autoenc
from custom_layers.model_creator import stage_creator, encode_smiles, add_new_layer
from NGF.preprocessing import tensorise_smiles


def gcn_encoder_fun(model_params):
    atoms, bonds, edges = encode_smiles(model_params["max_atoms"],
                            model_params["num_atom_features"],
                            model_params["max_degree"],
                            model_params["num_bond_features"])

    [model_enc_1, model_dec_pre_act_1, model_dec_after_act_1] = stage_creator(model_params,1,conv = True)
    [model_enc_2, model_dec_pre_act_2, model_dec_after_act_2] = stage_creator(model_params,2,conv = True)
    [model_enc_3, model_dec_pre_act_3, model_dec_after_act_3] = stage_creator(model_params,3,conv = True)

    [model_dec_fp_1, model_enc_fp_1] = stage_creator(model_params,1,conv = False)
    [model_dec_fp_2, model_enc_fp_2] = stage_creator(model_params,2,conv = False)
    [model_dec_fp_3, model_enc_fp_3] = stage_creator(model_params,3,conv = False)

    graph_conv_1 = model_enc_1([atoms,bonds,edges])
    graph_conv_2 = model_enc_2([graph_conv_1,bonds,edges])
    graph_conv_3 = model_enc_3([graph_conv_2,bonds,edges])


    fingerprint_1 = model_enc_fp_1([graph_conv_1,bonds,edges])
    fingerprint_1 = keras.layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_1)

    fingerprint_2 = model_enc_fp_2([graph_conv_2,bonds,edges])
    fingerprint_2 = keras.layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_2)

    fingerprint_3 = model_enc_fp_3([graph_conv_3,bonds,edges])
    fingerprint_3 = keras.layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_3)

    final_fingerprint = keras.layers.add([fingerprint_1,fingerprint_2,fingerprint_3])

    encoder = Model([atoms,bonds,edges],[final_fingerprint])
    encoder.summary()
    return(encoder)

def gcn_model_fun(model_params, params, encoder):
    atoms = Input(name='atom_inputs', shape=(model_params["max_atoms"], model_params["num_atom_features"]),dtype = 'float32')
    bonds = Input(name='bond_inputs', shape=(model_params["max_atoms"], model_params["max_degree"], model_params["num_bond_features"]),dtype = 'float32')
    edges = Input(name='edge_inputs', shape=(model_params["max_atoms"], model_params["max_degree"]), dtype='int32')
    encode_drug = encoder([atoms,bonds,edges])
    # Fully connected
    FC1 = Dense(params["dense_size"][0], activation='relu',kernel_initializer='random_normal')(encode_drug)
    FC2 = Dropout(params["dropout_rate"][0])(FC1)
    FC2 = Dense(params["dense_size"][1], activation='relu',kernel_initializer='random_normal')(FC2)
    FC2 = Dropout(params["dropout_rate"][1])(FC2)
    FC2 = Dense(params["dense_size"][2], activation='relu',kernel_initializer='random_normal')(FC2)
    predictions = Dense(1,activation='sigmoid', kernel_initializer='random_normal')(FC2)
    gcn_model = Model(inputs=[atoms,bonds,edges], outputs=predictions)
    adam = keras.optimizers.Adam(lr=params["lr"],beta_1=0.9,beta_2=0.999,decay=0.0,amsgrad=False)
    gcn_model.compile(optimizer= adam,loss= 'binary_crossentropy',metrics = ['accuracy'])
    print(gcn_model.summary())
    return gcn_model
