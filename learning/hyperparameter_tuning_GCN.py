from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, auc, average_precision_score
#import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import History, ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import keras
import keras.backend as K
from NGF.preprocessing import tensorise_smiles
from custom_layers.model_creator import encode_smiles, stage_creator
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, auc, average_precision_score, pairwise_distances
from hyperopt import STATUS_OK

class GCN_hyper(object):

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
        atoms = Input(name='atom_inputs',
                      shape=(self.model_params['max_atoms'], self.model_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(
        self.model_params['max_atoms'], self.model_params['max_degree'], self.model_params['num_bond_features']),
                      dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.model_params['max_atoms'], self.model_params['max_degree']),
                      dtype='int32')
        encode_drug = encoder([atoms, bonds, edges])

        # Fully connected
        FC1 = Dense(self.model_params["dense_size"][0], activation='relu',kernel_initializer='random_normal')(encode_drug)
        FC2 = Dropout(self.model_params["dropout_rate"][0])(FC1)
        FC2 = Dense(self.model_params["dense_size"][1], activation='relu',kernel_initializer='random_normal')(FC2)
        FC2 = Dropout(self.model_params["dropout_rate"][1])(FC2)
        FC2 = Dense(self.model_params["dense_size"][2], activation='relu',kernel_initializer='random_normal')(FC2)
        predictions = Dense(1, activation='sigmoid', kernel_initializer='random_normal')(FC2)
        gcn_model = Model(inputs=[atoms, bonds, edges], outputs=predictions)

        adam = keras.optimizers.Adam(lr=self.model_params["lr"], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        gcn_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        if verbose:
            print('encoder')
            encoder.summary()
            print('GCN_model')
            gcn_model.summary()

        return gcn_model

    def dataframe_to_gcn_input(self, input_data):
        x_atoms_cold, x_bonds_cold, x_edges_cold = tensorise_smiles(input_data['rdkit'],
                                                                    max_degree=self.model_params['max_degree'],
                                                                    max_atoms=self.model_params['max_atoms'])
        return [x_atoms_cold, x_bonds_cold, x_edges_cold]

def objective(fspace, train_sets, val_sets):
    K.clear_session()
    maps = []
    es = EarlyStopping(monitor='loss',patience=8, min_delta=0)
    rlr = ReduceLROnPlateau(monitor='loss',factor=0.5, patience=4, verbose=1, min_lr=0.0000001)
    model_params = {
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
        'batch_size' : int(64),
        'n_epochs' : int(fspace['n_epochs'])
        }
    gcn = GCN_hyper(model_params)
    for i in range(len(train_sets)):
        X_atoms_cold,X_bonds_cold,X_edges_cold = gcn.dataframe_to_gcn_input(val_sets[i])
        Y_cold = val_sets[i].Binary
        X_atoms_train, X_bonds_train, X_edges_train = gcn.dataframe_to_gcn_input(train_sets[i])
        Y = train_sets[i].Binary
        gcn_encoder = gcn.build_encoder()
        gcn_model = gcn.build_model(gcn_encoder)
        gcn_model.fit([X_atoms_train,X_bonds_train,X_edges_train],Y,
                    batch_size = model_params['batch_size'],
                    epochs = model_params['n_epochs'],
                    verbose = 0,
                    shuffle=True,
                    validation_data = None)
        y_pred_val = gcn_model.predict([X_atoms_cold,X_bonds_cold,X_edges_cold])
        maps.append(average_precision_score(Y_cold, y_pred_val))
    ave_map = np.mean(maps,axis = 0)
    return {'loss': -ave_map ,  'status': STATUS_OK}
