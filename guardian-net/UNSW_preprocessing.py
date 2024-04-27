import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

from utils import set_seed


class Preprocessor:
    def __init__(self, train_path, test_path, save_path, classification_m, label_col_name, norm_method):
        self.test_df = None
        self.train_df = None
        self.train_path = train_path
        self.test_path = test_path
        self.save_path = save_path
        self.classification_m = classification_m
        self.label_col_name = label_col_name
        self.norm_method = norm_method

    def __getitem__(self):
        return self.train_df, self.test_df

    def preprocess(self):
        self._read_data()
        self._drop_columns()
        self._categorize_columns()
        self._one_hot_encoding()
        self._scaling()
        self._replace_labels()
        self._reorder_columns()
        self._sort()
        self._shuffle()
        self._save()

    def _read_data(self):
        base_path = Path(__file__).resolve().parent
        dataset_path = os.path.join(base_path, '../dataset-orginal', 'UNSW', 'original')

        self.train_df = pd.read_csv(os.path.join(dataset_path, 'UNSW_NB15_training-set.csv'))
        print(len(self.train_df))
        self.test_df = pd.read_csv(os.path.join(dataset_path, 'UNSW_NB15_testing-set.csv'))
        print(len(self.test_df))

    def _drop_columns(self):
        self.train_df = self.train_df.drop(['id', 'label'], axis=1)
        self.test_df = self.test_df.drop(['id', 'label'], axis=1)

        self.train_df = self.train_df.rename(columns={'attack_cat': 'label'})
        self.test_df = self.test_df.rename(columns={'attack_cat': 'label'})

    def _categorize_columns(self):
        self.listBinary = ['is_ftp_login', 'is_sm_ips_ports']
        self.listNominal = ['proto', 'service', 'state']

        columns = set(self.train_df.columns)
        self.listNumerical = set(columns) - set(self.listNominal) - set(self.listBinary)
        self.listNumerical.remove('label')

    def _one_hot_encoding(self):
        train, test = self.train_df, self.test_df
        all_data = pd.concat((train, test))
        for column in all_data.select_dtypes(include='object').columns:
            train[column] = train[column].astype(
                pd.api.types.CategoricalDtype(categories=all_data[column].unique()))
            test[column] = test[column].astype(pd.api.types.CategoricalDtype(categories=all_data[column].unique()))

        for cat in self.listNominal:
            trainDum = pd.get_dummies(train[cat], prefix=cat)
            testDum = pd.get_dummies(test[cat], prefix=cat)
            train = pd.concat([train, trainDum.reindex(sorted(trainDum.columns), axis=1).astype(int)], axis=1)
            test = pd.concat([test, testDum.reindex(sorted(testDum.columns), axis=1).astype(int)], axis=1)
            train = train.drop(cat, axis=1)
            test = test.drop(cat, axis=1)

        try:
            train = train[[col for col in train.columns if col != self.label_col_name] + [self.label_col_name]]
            test = test[[col for col in test.columns if col != self.label_col_name] + [self.label_col_name]]
        except:
            print('COULD NOT MOVE LABEL COL TO LAST COLUMN OF DATAFRAME')

        self.train_df, self.test_df = train, test

    def _scaling(self):
        train, test = self.train_df, self.test_df
        listContent = list(self.listNumerical)
        if self.norm_method == 'normalization':
            scaler = MinMaxScaler()
        elif self.norm_method == 'standardization':
            scaler = StandardScaler()
        scaler.fit(train[listContent].values)
        train[listContent] = scaler.transform(train[listContent])
        test[listContent] = scaler.transform(test[listContent])
        self.train_df, self.test_df = train, test

    def _replace_labels(self):
        self.train_df[self.label_col_name].replace(
            to_replace=dict(Normal=0, Reconnaissance=1, DoS=1, Exploits=1, Fuzzers=1, Shellcode=1, Analysis=1,
                            Backdoor=1, Generic=1, Worms=1), inplace=True)

        self.test_df[self.label_col_name].replace(
            to_replace=dict(Normal=0, Reconnaissance=1, DoS=1, Exploits=1, Fuzzers=1, Shellcode=1, Analysis=1,
                            Backdoor=1, Generic=1, Worms=1), inplace=True)

        if sorted(self.train_df[self.label_col_name].unique()) == \
                sorted(self.train_df[self.label_col_name].unique()) == \
                [0, 1]:
            pass
        else:
            print('REPLACING LABELS WENT WRONG !!!')

        self.train_df[self.label_col_name] = self.train_df[self.label_col_name].astype('uint8')
        self.test_df[self.label_col_name] = self.test_df[self.label_col_name].astype('uint8')

    def _reorder_columns(self):
        self.train_df = self.train_df[[col for col in self.train_df.columns
                                       if col != self.label_col_name] + [self.label_col_name]]
        self.test_df = self.test_df[[col for col in self.test_df.columns
                                     if col != self.label_col_name] + [self.label_col_name]]

    def _sort(self):
        train_cols = self.train_df.columns
        test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
        for col in test_sortedBasedOnTrain:
            test_sortedBasedOnTrain[col] = self.test_df[col]

        self.test_df = test_sortedBasedOnTrain

    def _shuffle(self):
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)

    def _save(self):
        self.train_df.to_csv(os.path.join(self.save_path, 'UNSW_train_binary.csv'), index=False)
        self.test_df.to_csv(os.path.join(self.save_path, 'UNSW_test_binary.csv'), index=False)


if __name__ == '__main__':
    set_seed(0)
    base_path = Path(__file__).resolve().parent
    dataset_path = os.path.join(base_path, '../dataset-orginal', 'UNSW', 'original')

    train_path = os.path.join(dataset_path, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(dataset_path, 'UNSW_NB15_testing-set.csv')

    save_path = os.path.join(base_path, 'dataset-processed')
    classification_m = 'binary'

    preprocessor = Preprocessor(train_path, test_path, save_path, classification_m, 'label', 'normalization')
    preprocessor.preprocess()
    train_preprocessed, test_preprocessed = preprocessor.__getitem__()
    print(train_preprocessed.head())
    print(test_preprocessed.head())
