"""
This splitter works with the following strategy:
 - start with a molecule (the one with minimum average distance with all the others)
 - grab the molecule which is closest to it (in tanimoto similarity). Put these in the same fold.
 - Continue and grab the next molecule which is the closest to them (closest to either of those) and add it.
- When reaching fold_size stop.
"""
import pandas as pd
from tqdm import tqdm


class BiolabSplitter:

    def __init__(self, distance_matrix, k):
        self.distance_matrix = distance_matrix
        self.k = k

    @staticmethod
    def create_group(distance_matrix, fold_size):
        group_mols = [distance_matrix.mean().idxmin()]
        for _ in range(fold_size):
            mol_with_nearest_neighbor = distance_matrix.loc[group_mols].drop(columns=group_mols).max(axis=1).idxmax()
            next_mol = distance_matrix.loc[mol_with_nearest_neighbor].drop(group_mols).idxmax()
            group_mols.append(next_mol)
        return group_mols

    def create_folds(self):
        used_mols = []
        folds = []
        fold_size = int(len(self.distance_matrix) / self.k)
        for _ in tqdm(range(self.k - 1)):
            new_dm = self.distance_matrix.drop(used_mols).drop(columns=used_mols)
            mols_added = self.create_group(new_dm, fold_size)
            used_mols += mols_added
            folds.append(mols_added)
        final_group = self.distance_matrix.drop(used_mols).drop(columns=used_mols).index
        folds = [pd.Index(x) for x in folds] + [final_group]

        for i, f in enumerate(folds):
            for j in range(self.k):
                if j != i:
                    assert len(f.intersection(folds[j])) == 0, 'Whoops. There is data leakage between the folds. ' \
                                                               'Something went wrong.'

            assert len(set(f)) == len(f), 'Whoops. There are duplicates inside the folds. Something went wrong.'

        # return the folds in the form [(test_0, val_0), (test_1, val_1),..., (test_k, val_k)]
        folds = [(self.distance_matrix.index.difference(f), f) for f in folds]
        return folds


def create_test_set_from_folds(folds: list, i: int):
    """
    Gets a list of k folds in the form [(test_0, val_0), (test_1, val_1),..., (test_k, val_k)] and a number of fold i
    and separates this fold as test set ensuring no data leakage.

    Returns:
        The new list of train_val sets without the test set and the (all_train_set, test_set) tuple.
    """
    assert i < len(folds)
    test = folds[i][1]
    train = folds[i][0]
    new_folds = []
    for k, fold in enumerate(folds):
        if k != i:
            new_folds.append((fold[0].intersection(train), fold[1].intersection(train)))

    return new_folds, (train, test)
