import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def split(data, repeat, input_path):
    kf = KFold(n_splits=5, random_state=repeat, shuffle=True)
    seed = 42
    i = 1
    for train_index, test_index in kf.split(data):
        data_train_val = data.iloc[train_index]
        data_test = data.iloc[test_index]
        data_train, data_val = train_test_split(data_train_val, test_size=0.25, random_state=seed)

        os.makedirs(input_path, exist_ok=True)
        path_train = os.path.join(input_path, f"repeat_{str(repeat)}_fold_{str(i)}_train.csv")
        path_val = os.path.join(input_path, f"repeat_{str(repeat)}_fold_{str(i)}_val.csv")
        path_test = os.path.join(input_path, f"repeat_{str(repeat)}_fold_{str(i)}_test.csv")

        i += 1
        print(path_train)
        data_train.to_csv(path_train)
        data_val.to_csv(path_val)
        data_test.to_csv(path_test)


class load_data(Dataset):
    def __init__(self, csv_path, target_name, drug_feat, cell_feat,
                 drug_scaler=None,
                 cell_scaler=None,
                 fit_preprocessor=False):

        data = pd.read_csv(csv_path)
        self.labels = data[target_name].values
        self.inputs = data[['drug_row', 'drug_col', 'cell_line_name']]

        self.drug_feats = drug_feat.drop(['drug_name'], axis=1).set_index('ex_drug_id')

        self.cell_feats = cell_feat.drop(['cell_name', 'model_id'], axis=1).set_index('ex_cell_id')

        self._collect_all_features()

        if fit_preprocessor:
            self.drug_scaler = StandardScaler().fit(self.raw_drug_features)
            self.cell_scaler = StandardScaler().fit(self.raw_cell_features)
        else:
            self.drug_scaler = drug_scaler
            self.cell_scaler = cell_scaler

        self._preprocess_features()

    def _collect_all_features(self):
        all_drug_ids = pd.unique(self.inputs[['drug_row', 'drug_col']].values.ravel('K'))
        self.raw_drug_features = self.drug_feats.loc[all_drug_ids].values

        all_cell_ids = pd.unique(self.inputs['cell_line_name'])
        cell_features = self.cell_feats.loc[all_cell_ids].values
        self.raw_cell_features = cell_features

    def _preprocess_features(self):
        drugs_scaled = self.drug_scaler.transform(self.drug_feats.values)
        self.drugs_processed = pd.DataFrame(
            drugs_scaled,
            index=self.drug_feats.index,
            columns=self.drug_feats.columns
        )

        cell_scaled = self.cell_scaler.transform(self.cell_feats.values)
        self.cells_processed = pd.DataFrame(
            cell_scaled,
            index=self.cell_feats.index,
            columns=self.cell_feats.columns
        )

    def __getitem__(self, index):
        row = self.inputs.iloc[index]

        drug_row = self.drugs_processed.loc[row['drug_row']].values
        drug_col = self.drugs_processed.loc[row['drug_col']].values

        cell = self.cells_processed.loc[row['cell_line_name']].values

        features = np.concatenate([drug_row, drug_col, cell])
        return (
            torch.from_numpy(features).float(),
            self.labels[index]
        )

    def __len__(self):
        return len(self.labels)