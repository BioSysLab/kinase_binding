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
        for _ in tqdm(range(self.k - 1)):
            new_dm = self.distance_matrix.drop(used_mols).drop(columns=used_mols)
            mols_added = self.create_group(new_dm, int(len(self.distance_matrix) / self.k))
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

        return folds
