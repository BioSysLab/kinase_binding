from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, \
    auc, average_precision_score, pairwise_distances
#import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import functools
import time
from tqdm import tqdm
tqdm.pandas()

from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

important_rdkit_features = [
    'BalabanJ', 'BertzCT', 'MaxAbsPartialCharge','MolLogP',
     'MolWt', 'NumAliphaticCarbocycles', 'NumRotatableBonds', 'RingCount','SlogP_VSA10','TPSA']

def get_morgan_fingerprints(
    smiles,
    radius=2,
    fplength=1024
):
    fingerprint_function = functools.partial(GetMorganFingerprintAsBitVect, radius=radius, nBits=fplength)
    mols = pd.Series([Chem.MolFromSmiles(s) for s in smiles], smiles.index)
    return mols.apply(fingerprint_function)


def get_distance_matrix(mols):
    item_values = mols.values
    print('Calculating Distance Matrix')
    t = time.time()
    distances = pairwise_distances(np.asarray(item_values.tolist()), metric='jaccard', n_jobs=-1)
    print(f'Calculated Distance Matrix in {(time.time() - t):.2f} seconds')

    return pd.DataFrame(distances, index=mols.index, columns=mols.index)


def get_rdkit_features(
    df,
    smiles_col='rdkit'
):
    mols = pd.Series([Chem.MolFromSmiles(s) for s in df[smiles_col]], df.index)
    for name, func in tqdm(Descriptors.descList):
        if name in important_rdkit_features:
            df[name] = mols.progress_apply(func)
    return df


def plot_murcko_scaffolds(df, smiles_col='rdkit', max_elements=96, molsPerRow=10):
    mols = [AllChem.MolFromSmiles(sm) for sm in df[smiles_col].sort_values().unique()]
    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(350, 350), maxMols=max_elements)


def murcko_scaffolds_analysis(train_df, val_df, class_label_col='Binary', max_elements=96, molsPerRow=10):
    actives_train_msp = plot_murcko_scaffolds(train_df[train_df[class_label_col] == 1.0], max_elements=max_elements,
                                              molsPerRow=molsPerRow)
    inactives_train_msp = plot_murcko_scaffolds(train_df[train_df[class_label_col] == 0.0], max_elements=max_elements,
                                                molsPerRow=molsPerRow)
    actives_valid_msp = plot_murcko_scaffolds(val_df[val_df[class_label_col] == 1.0], max_elements=max_elements,
                                              molsPerRow=molsPerRow)
    inactives_valid_msp = plot_murcko_scaffolds(val_df[val_df[class_label_col] == 0.0], max_elements=max_elements,
                                                molsPerRow=molsPerRow)
    fig, ax = plt.subplots(1, 2, figsize=(16, 12))
    ax[0].imshow(actives_train_msp)
    ax[0].set_title('Actives - Training Set')
    ax[1].imshow(actives_valid_msp)
    ax[1].set_title('Actives - Validation Set')
    fig.tight_layout()
    plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(16, 12))
    ax[0].imshow(inactives_train_msp)
    ax[0].set_title('Inactives - Training Set')
    ax[1].imshow(inactives_valid_msp)
    ax[1].set_title('Inactives - Validation Set')
    fig.tight_layout()
    plt.show()


def calculate_metrics(y_true, y_pred, plots=False):
    assert isinstance(y_true, np.ndarray), 'y_true should be np.array'
    assert len(y_true.shape) == len(y_pred.shape) == 1, 'y_true or y_pred shapes are not 1 (probably not squeezed)'
    y_pred_bin = y_pred > 0.5

    cf = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cf.ravel()

    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'map': average_precision_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_bin),
        'recall': recall_score(y_true, y_pred_bin),
        'accuracy': accuracy_score(y_true, y_pred_bin),
    }

    if plots:
        print('predictions histogram')
        plt.figure()
        plt.hist(y_pred, bins=int(len(y_pred) / 3))
        plt.show()

        print('confusion matrix')
        plt.figure()
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in
                        cf.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                             cf.flatten() / np.sum(cf)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                  zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf, annot=labels, fmt='', cmap='Blues')
        plt.show()

        print('roc curve')
        random_probs = [0 for _ in range(len(y_true))]
        auc = roc_auc_score(y_true, y_pred)
        print('Logistic: ROC AUC=%.3f' % (auc))
        ns_fpr, ns_tpr, _ = roc_curve(y_true, random_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='random')
        plt.plot(lr_fpr, lr_tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    return metrics

def load_weights_and_evaluate(eval_params):
    preds_val = []
    preds_test = []
    dfs = []
    model = eval_params['model']
    if eval_params['model_type'] == 'gcn':
        test_data = eval_params['model_class'].dataframe_to_gcn_input(eval_params['test_set'])
    for i, df_val in enumerate(eval_params['val_sets']):
        y_true = df_val.Binary
        en_preds_val = []
        en_preds_test = []
        if eval_params['model_type'] == 'gcn':
            val_data = eval_params['model_class'].dataframe_to_gcn_input(df_val)
        for j in range(eval_params['n_ensemble']):
            model.load_weights(eval_params['weight_file_format'].format(i,j))
            pred_val = model.predict(val_data, batch_size = 1024)
            en_preds_val.append(pred_val)
            en_preds_test.append(model.predict(test_data,batch_size = 1024))
        preds_val.append(np.mean(en_preds_val, axis = 0))
        preds_test.append(np.mean(en_preds_test, axis = 0))
        dfs.append(calculate_metrics(y_true.values, np.mean(en_preds_val, axis = 0).squeeze(), plots=True))
    ave_preds = np.mean(preds_test,axis = 0)
    dfs.append(calculate_metrics(eval_params['test_set'].Binary.values, ave_preds.squeeze(), plots=True))
    metrics = pd.DataFrame(dfs)
    metrics.rename(index={(len(eval_params['val_sets'])):'test_set'}, inplace=True)
    return(metrics)
