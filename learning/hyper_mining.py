import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import keras
import keras.backend as K
from keras.layers import Dense, Dropout, Input, Lambda, concatenate,Flatten
from keras.models import Model, load_model
import seaborn as sns
from keras.callbacks import History, ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from distance_and_mask_fn import pairwise_distance,masked_maximum,masked_minimum

from NGF.preprocessing import tensorise_smiles
from custom_layers.model_creator import encode_smiles, stage_creator
from data_analysis import calculate_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, auc, average_precision_score
from hyperopt import STATUS_OK
import xgboost as xgb

class GCN_online_mining(object):
    
    def __init__(self,  model_params):
        self.model_params = model_params
    
    def build_encoder(self):
        model_enc_1 = stage_creator(self.model_params, 1, conv=True)[0]
        model_enc_2 = stage_creator(self.model_params, 2, conv=True)[0]
        model_enc_3 = stage_creator(self.model_params, 3, conv=True)[0]
        
        model_enc_fp_1 = stage_creator(self.model_params, 1, conv=False)[1]
        model_enc_fp_2 = stage_creator(self.model_params, 2, conv=False)[1]
        model_enc_fp_3 = stage_creator(self.model_params, 3, conv=False)[1]
        
        atoms, bonds, edges = encode_smiles(self.model_params["max_atoms"],
                                            self.model_params["num_atom_features"],
                                            self.model_params["max_degree"],
                                            self.model_params["num_bond_features"])
        
        graph_conv_1 = model_enc_1([atoms, bonds, edges])
        graph_conv_2 = model_enc_2([graph_conv_1, bonds, edges])
        graph_conv_3 = model_enc_3([graph_conv_2, bonds, edges])
        
        fingerprint_1 = model_enc_fp_1([graph_conv_1, bonds, edges])
        fingerprint_1 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_1)
        
        fingerprint_2 = model_enc_fp_2([graph_conv_2, bonds, edges])
        fingerprint_2 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_2)
        
        fingerprint_3 = model_enc_fp_3([graph_conv_3, bonds, edges])
        fingerprint_3 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_3)
        
        final_fingerprint = keras.layers.add([fingerprint_1, fingerprint_2, fingerprint_3])
        return Model([atoms, bonds, edges], [final_fingerprint])
    
    def build_model(self, encoder, verbose=False):
        atoms = Input(name='atom_inputs',shape=(self.model_params['max_atoms'],
                                                self.model_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree'],
                                                 self.model_params['num_bond_features']),dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree']),dtype='int32')
        encode_drug = encoder([atoms, bonds, edges])
        
        # Fully connected
        FC1 = Dense(self.model_params["dense_size"][0], 
                    activation='relu',kernel_initializer='random_normal')(encode_drug)
        FC2 = Dropout(self.model_params["dropout_rate"][0])(FC1)
        FC2 = Dense(self.model_params["dense_size"][1], 
                    activation='relu',kernel_initializer='random_normal')(FC2)
        FC2 = Dropout(self.model_params["dropout_rate"][1])(FC2)
        FC2 = Dense(self.model_params["dense_size"][2], 
                    activation = None,kernel_initializer='random_normal')(FC2)
        
        
        embeddings = Lambda(lambda x: K.l2_normalize(x,axis=1),name = 'Embeddings')(FC2)
        
        gcn_model = Model(inputs=[atoms, bonds, edges], outputs = embeddings)
        
        if verbose:
            print('encoder')
            encoder.summary()
            print('GCN_model')
        return gcn_model
    
    def triplet_loss_adapted_from_tf(self,y_true, y_pred):
        del y_true
        margin = self.model_params['margin']
        labels = y_pred[:, :1]
        labels = tf.cast(labels, dtype='int32')
        embeddings = y_pred[:, 1:]

        ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        # lshape=array_ops.shape(labels)
        # assert lshape.shape == 1
        # labels = array_ops.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix.
        pdist_matrix = pairwise_distance(embeddings, squared=False)
        # Build pairwise binary adjacency matrix.
        adjacency = math_ops.equal(labels, array_ops.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = math_ops.logical_not(adjacency)

        # global batch_size  
        batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

        # Compute the mask.
        pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
        mask = math_ops.logical_and(
            array_ops.tile(adjacency_not, [batch_size, 1]),
            math_ops.greater(
                pdist_matrix_tile, array_ops.reshape(
                    array_ops.transpose(pdist_matrix), [-1, 1])))
        mask_final = array_ops.reshape(
            math_ops.greater(
                math_ops.reduce_sum(
                    math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
                0.0), [batch_size, batch_size])
        mask_final = array_ops.transpose(mask_final)

        adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
        mask = math_ops.cast(mask, dtype=dtypes.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = array_ops.reshape(
            masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = array_ops.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = array_ops.tile(
            masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
        semi_hard_negatives = array_ops.where(
            mask_final, negatives_outside, negatives_inside)

        loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = math_ops.cast(
            adjacency, dtype=dtypes.float32) - array_ops.diag(
            array_ops.ones([batch_size]))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = math_ops.reduce_sum(mask_positives)

        semi_hard_triplet_loss_distance = math_ops.truediv(
            math_ops.reduce_sum(
                math_ops.maximum(
                    math_ops.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss')
    
        ### Code from Tensorflow function semi-hard triplet loss ENDS here.
        return semi_hard_triplet_loss_distance

    def build_mining(self,gcn_model):
        atoms = Input(name='atom_inputs',shape=(self.model_params['max_atoms'],
                                                self.model_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree'],
                                                 self.model_params['num_bond_features']),dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree']),dtype='int32')
        labels = Input(name = 'labels_inputs',shape = (1,),dtype = 'float32')
        encoded = gcn_model([atoms,bonds,edges])
        labels_plus_embeddings = concatenate([labels, encoded])
        mining_net = Model(inputs = [atoms,bonds,edges,labels],outputs = labels_plus_embeddings)
        adam = keras.optimizers.Adam(lr = self.model_params["lr"], 
                                     beta_1=0.9, 
                                     beta_2=0.999, 
                                     decay=0.0, 
                                     amsgrad=False)
        mining_net.compile(optimizer=adam , loss = triplet_loss_adapted_from_tf)
        return mining_net
    
    def dataframe_to_gcn_input(self,input_data):
        x_atoms_cold, x_bonds_cold, x_edges_cold = tensorise_smiles(input_data['rdkit'],
                                                                    max_degree=self.model_params['max_degree'],max_atoms=self.model_params['max_atoms'])
        return [x_atoms_cold, x_bonds_cold, x_edges_cold]


class GCN_online_mining_test(object):
    
    def __init__(self,  model_params):
        self.model_params = model_params
    
    def build_encoder(self):
        model_enc_1 = stage_creator(self.model_params, 1, conv=True)[0]
        model_enc_2 = stage_creator(self.model_params, 2, conv=True)[0]
        model_enc_3 = stage_creator(self.model_params, 3, conv=True)[0]
        
        model_enc_fp_1 = stage_creator(self.model_params, 1, conv=False)[1]
        model_enc_fp_2 = stage_creator(self.model_params, 2, conv=False)[1]
        model_enc_fp_3 = stage_creator(self.model_params, 3, conv=False)[1]
        
        atoms, bonds, edges = encode_smiles(self.model_params["max_atoms"],
                                            self.model_params["num_atom_features"],
                                            self.model_params["max_degree"],
                                            self.model_params["num_bond_features"])
        
        graph_conv_1 = model_enc_1([atoms, bonds, edges])
        graph_conv_2 = model_enc_2([graph_conv_1, bonds, edges])
        graph_conv_3 = model_enc_3([graph_conv_2, bonds, edges])
        
        fingerprint_1 = model_enc_fp_1([graph_conv_1, bonds, edges])
        fingerprint_1 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_1)
        
        fingerprint_2 = model_enc_fp_2([graph_conv_2, bonds, edges])
        fingerprint_2 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_2)
        
        fingerprint_3 = model_enc_fp_3([graph_conv_3, bonds, edges])
        fingerprint_3 = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(fingerprint_3)
        
        final_fingerprint = keras.layers.add([fingerprint_1, fingerprint_2, fingerprint_3])
        return Model([atoms, bonds, edges], [final_fingerprint])
    
    def build_model(self, encoder, verbose=False):
        atoms = Input(name='atom_inputs',shape=(self.model_params['max_atoms'],
                                                self.model_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree'],
                                                 self.model_params['num_bond_features']),dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree']),dtype='int32')
        encode_drug = encoder([atoms, bonds, edges])
        
        # Fully connected
        FC1 = Dense(self.model_params["dense_size"][0], 
                    activation='relu',kernel_initializer='random_normal')(encode_drug)
        FC2 = Dropout(self.model_params["dropout_rate"][0])(FC1)
        FC2 = Dense(self.model_params["dense_size"][1], 
                    activation='relu',kernel_initializer='random_normal')(FC2)
        FC2 = Dropout(self.model_params["dropout_rate"][1])(FC2)
        FC2 = Dense(self.model_params["dense_size"][2], 
                    activation = None,kernel_initializer='random_normal')(FC2)
        
        
        embeddings = Lambda(lambda x: K.l2_normalize(x,axis=1),name = 'Embeddings')(FC2)
        
        gcn_model = Model(inputs=[atoms, bonds, edges], outputs = embeddings)
        
        if verbose:
            print('encoder')
            encoder.summary()
            print('GCN_model')
        return gcn_model
    
    def build_mining(self,gcn_model):
        atoms = Input(name='atom_inputs',shape=(self.model_params['max_atoms'],
                                                self.model_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree'],
                                                 self.model_params['num_bond_features']),dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.model_params['max_atoms'], 
                                                 self.model_params['max_degree']),dtype='int32')
        labels = Input(name = 'labels_inputs',shape = (1,),dtype = 'float32')
        encoded = gcn_model([atoms,bonds,edges])
        labels_plus_embeddings = concatenate([labels, encoded])
        mining_net = Model(inputs = [atoms,bonds,edges,labels],outputs = labels_plus_embeddings)
        adam = keras.optimizers.Adam(lr = self.model_params["lr"], 
                                     beta_1=0.9, 
                                     beta_2=0.999, 
                                     decay=0.0, 
                                     amsgrad=False)
        mining_net.compile(optimizer=adam , loss = triplet_loss_adapted_from_tf_2)
        return mining_net
    
    def dataframe_to_gcn_input(self,input_data):
        x_atoms_cold, x_bonds_cold, x_edges_cold = tensorise_smiles(input_data['rdkit'],
                                                                    max_degree=self.model_params['max_degree'],max_atoms=self.model_params['max_atoms'])
        return [x_atoms_cold, x_bonds_cold, x_edges_cold]
    
    
def triplet_loss_adapted_from_tf_2(y_true, y_pred,margin = 0.6011401246738063):
        del y_true
        margin = margin
        labels = y_pred[:, :1]
        labels = tf.cast(labels, dtype='int32')
        embeddings = y_pred[:, 1:]

        ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        # lshape=array_ops.shape(labels)
        # assert lshape.shape == 1
        # labels = array_ops.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix.
        pdist_matrix = pairwise_distance(embeddings, squared=False)
        # Build pairwise binary adjacency matrix.
        adjacency = math_ops.equal(labels, array_ops.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = math_ops.logical_not(adjacency)

        # global batch_size  
        batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

        # Compute the mask.
        pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
        mask = math_ops.logical_and(
            array_ops.tile(adjacency_not, [batch_size, 1]),
            math_ops.greater(
                pdist_matrix_tile, array_ops.reshape(
                    array_ops.transpose(pdist_matrix), [-1, 1])))
        mask_final = array_ops.reshape(
            math_ops.greater(
                math_ops.reduce_sum(
                    math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
                0.0), [batch_size, batch_size])
        mask_final = array_ops.transpose(mask_final)

        adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
        mask = math_ops.cast(mask, dtype=dtypes.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = array_ops.reshape(
            masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = array_ops.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = array_ops.tile(
            masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
        semi_hard_negatives = array_ops.where(
            mask_final, negatives_outside, negatives_inside)

        loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = math_ops.cast(
            adjacency, dtype=dtypes.float32) - array_ops.diag(
            array_ops.ones([batch_size]))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = math_ops.reduce_sum(mask_positives)

        semi_hard_triplet_loss_distance = math_ops.truediv(
            math_ops.reduce_sum(
                math_ops.maximum(
                    math_ops.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss')
    
        ### Code from Tensorflow function semi-hard triplet loss ENDS here.
        return semi_hard_triplet_loss_distance
class XGB_predictor(object):
    
    def __init__(self,  xgb_params):
        self.xgb_params = xgb_params
    
    def build_model(self,dmatrix,evalist,num_round):
        model = xgb.train(self.xgb_params,dmatrix,num_round,evalist,verbose_eval = False)
        
        return model
    
    def to_xgb_input(self,binary,embeddings):
        dmatrix = xgb.DMatrix(data = embeddings,label = binary)
        return(dmatrix)
        
def objective_fn(fspace,train_sets,val_sets):
    K.clear_session()
    maps = []
    es = EarlyStopping(monitor='loss',patience=8, min_delta=0)
    rlr = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=4, verbose=1, min_lr=0.0000001)
    es2 = EarlyStopping(monitor='loss',patience=15, min_delta=0)
    rlr2 = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=2, verbose=1, min_lr=0.000000001)
    gcn_params = {
        "num_layers" : 3,
        "max_atoms" : 70,
        "num_atom_features" : 62,
        "num_atom_features_original" : 62,
        "num_bond_features" : 6,
        "max_degree" : 5,
        "conv_width" : [int(fspace['conv1']), int(fspace['conv2']), int(fspace['conv3'])],
        "fp_length" : [int(fspace['fp']), int(fspace['fp']), int(fspace['fp'])],
        "activ_enc" : "selu",
        "activ_dec" : "selu",
        "learning_rates" : [0.001,0.001,0.001],
        "learning_rates_fp": [0.005,0.005,0.005],
        "losses_conv" : {
                    "neighbor_output": "mean_squared_error",
                    "self_output": "mean_squared_error",
                    },
        "lossWeights" : {"neighbor_output": 1.0, "self_output": 1.0},
        "metrics" : "mse",
        "loss_fp" : "mean_squared_error",
        "enc_layer_names" : ["enc_1", "enc_2", "enc_3"],
        'callbacks' : [es,rlr],
        'adam_decay': 0.0005329142291371636,
        'beta': 5,
        'p': 0.004465204118126482,
        'dense_size' : [int(fspace['dense1']), int(fspace['dense2']), int(fspace['dense3'])],
        'dropout_rate' : [fspace['dropout_rate'], fspace['dropout_rate']],
        'lr' : fspace['lr'],
        'batch_size' : int(fspace['batch_size']),
        'n_epochs' : int(fspace['n_epochs']),
        'margin' : fspace['margin']
        }
    xgb_params = {
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
        "max_bin" : int(fspace['max_bin']),
        "eval_metric":'auc',
        "objective":'binary:logistic',
        "booster":'gbtree'
        #"single_precision_histogram" : True
        }
    class_XGB = XGB_predictor(xgb_params)
    class_GCN = GCN_online_mining(gcn_params)
    for i in range(len(train_sets)):
        X_atoms_cold,X_bonds_cold,X_edges_cold = class_GCN.dataframe_to_gcn_input(val_sets[i])
        Y_cold = val_sets[i].Binary
        Y_dummy_cold = np.empty((X_atoms_cold.shape[0],gcn_params['dense_size'][2]+1))
        X_atoms_train, X_bonds_train, X_edges_train = class_GCN.dataframe_to_gcn_input(train_sets[i])
        Y = train_sets[i].Binary
        Y_dummy_train = np.empty((X_atoms_train.shape[0],gcn_params['dense_size'][2]+1))
        
        gcn_encoder = class_GCN.build_encoder()
        gcn_model = class_GCN.build_model(gcn_encoder)
        gcn_mining = class_GCN.build_mining(gcn_model)
        
        gcn_mining.fit([X_atoms_train,X_bonds_train,X_edges_train,Y],
                       Y_dummy_train,
                       epochs = gcn_params['n_epochs'],
                       batch_size = gcn_params['batch_size'],
                       shuffle = True,
                       validation_data = ([X_atoms_cold,X_bonds_cold,X_edges_cold,Y_cold],Y_dummy_cold),
                       callbacks=[es2,rlr2]
                      )
        #Predict Embeddings
        embeddings_cold = gcn_model.predict([X_atoms_cold,X_bonds_cold,X_edges_cold])
        embeddings_train = gcn_model.predict([X_atoms_train, X_bonds_train, X_edges_train])
        
        #Prepare data for XGBoost
        dmatrix_train = class_XGB.to_xgb_input(Y,embeddings_train)
        dmatrix_cold = class_XGB.to_xgb_input(Y_cold,embeddings_cold)
        
        evalist = [(dmatrix_train,'train'),(dmatrix_cold,'eval')]
        xgb_model = class_XGB.build_model(dmatrix_train,evalist,300)
        xgb_pred_cold = xgb_model.predict(dmatrix_cold)
        maps.append(average_precision_score(Y_cold, xgb_pred_cold))
        
    
    ave_map = np.mean(maps,axis = 0)
    return {'loss': -ave_map ,  'status': STATUS_OK}
        