from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, auc, average_precision_score
#import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import keras.backend as K
from NGF.preprocessing import tensorise_smiles
from custom_layers.model_creator import encode_smiles, stage_creator
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model, load_model


class GCN_pretraining(object):

    def __init__(self, encoder_params, model_params):
        self.encoder_params = encoder_params
        self.model_params = model_params

    def build_encoder(self):
        model_enc_1 = stage_creator(self.encoder_params, 1, conv=True)[0]
        model_enc_2 = stage_creator(self.encoder_params, 2, conv=True)[0]
        model_enc_3 = stage_creator(self.encoder_params, 3, conv=True)[0]

        model_enc_fp_1 = stage_creator(self.encoder_params, 1, conv=False)[1]
        model_enc_fp_2 = stage_creator(self.encoder_params, 2, conv=False)[1]
        model_enc_fp_3 = stage_creator(self.encoder_params, 3, conv=False)[1]

        atoms, bonds, edges = encode_smiles(self.encoder_params["max_atoms"],
                                            self.encoder_params["num_atom_features"],
                                            self.encoder_params["max_degree"],
                                            self.encoder_params["num_bond_features"])

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
                      shape=(self.encoder_params['max_atoms'], self.encoder_params['num_atom_features']), dtype='float32')
        bonds = Input(name='bond_inputs', shape=(
        self.encoder_params['max_atoms'], self.encoder_params['max_degree'], self.encoder_params['num_bond_features']),
                      dtype='float32')
        edges = Input(name='edge_inputs', shape=(self.encoder_params['max_atoms'], self.encoder_params['max_degree']),
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
                                                                    max_degree=self.encoder_params['max_degree'],
                                                                    max_atoms=self.encoder_params['max_atoms'])
        return [x_atoms_cold, x_bonds_cold, x_edges_cold]

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ',y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12

    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = tf.reduce_mean(K.maximum(basic_loss,0.0))

    return loss

class GCN_siam_model(object):

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
        FC2 = Dense(self.model_params["dense_size"][2], activation = None,kernel_initializer='random_normal')(FC2)
        #predictions = Dense(1, activation='sigmoid', kernel_initializer='random_normal')(FC2)
        embeddings = Lambda(lambda x: K.l2_normalize(x,axis=1))(FC2)
        gcn_model = Model(inputs=[atoms, bonds, edges], outputs = embeddings)



        if verbose:
            print('encoder')
            encoder.summary()
            print('GCN_model')
            gcn_model.summary()

        return gcn_model

    def build_siam(self, model, verbose = False):
        anchor_atoms = Input(name='atom_inputs_anchor',
                      shape=(self.model_params['max_atoms'], self.model_params['num_atom_features']), dtype='float32')
        anchor_bonds = Input(name='bond_inputs_anchor', shape=(
        self.model_params['max_atoms'], self.model_params['max_degree'], self.model_params['num_bond_features']),
                      dtype='float32')
        anchor_edges = Input(name='edge_inputs_anchor', shape=(self.model_params['max_atoms'], self.model_params['max_degree']),
                      dtype='int32')

        pos_atoms = Input(name='atom_inputs_pos',
                      shape=(self.model_params['max_atoms'], self.model_params['num_atom_features']), dtype='float32')
        pos_bonds = Input(name='bond_inputs_pos', shape=(
        self.model_params['max_atoms'], self.model_params['max_degree'], self.model_params['num_bond_features']),
                      dtype='float32')
        pos_edges = Input(name='edge_inputs_pos', shape=(self.model_params['max_atoms'], self.model_params['max_degree']),
                      dtype='int32')

        neg_atoms = Input(name='atom_inputs_neg',
                      shape=(self.model_params['max_atoms'], self.model_params['num_atom_features']), dtype='float32')
        neg_bonds = Input(name='bond_inputs_neg', shape=(
        self.model_params['max_atoms'], self.model_params['max_degree'], self.model_params['num_bond_features']),
                      dtype='float32')
        neg_edges = Input(name='edge_inputs_neg', shape=(self.model_params['max_atoms'], self.model_params['max_degree']),
                      dtype='int32')

        encoded_a = model([anchor_atoms,anchor_bonds,anchor_edges])
        encoded_p = model([pos_atoms,pos_bonds,pos_edges])
        encoded_n = model([neg_atoms,neg_bonds,neg_edges])

        merged_vector = keras.layers.concatenate([encoded_a, encoded_p, encoded_n], axis=-1, name='merged_layer')

        siamese_net = Model(inputs=[anchor_atoms,anchor_bonds,anchor_edges,pos_atoms,pos_bonds,pos_edges,neg_atoms,neg_bonds,neg_edges],outputs=merged_vector)
        adam = keras.optimizers.Adam(lr=self.model_params["lr"], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        siamese_net.compile(loss=triplet_loss, optimizer=adam)
        return(siamese_net)

    def dataframe_to_gcn_input(self, input_data):
        x_atoms_cold, x_bonds_cold, x_edges_cold = tensorise_smiles(input_data,
                                                                    max_degree=self.model_params['max_degree'],
                                                                    max_atoms=self.model_params['max_atoms'])
        return [x_atoms_cold, x_bonds_cold, x_edges_cold]
