import random

import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from pathlib import Path

from utils import parse_data, set_seed

np.random.seed(0)
random.seed(0)


class BuildDataFrames:
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 classification_mode: str = 'binary',
                 label_col_name: str = 'label'
                 ):
        self.listNumerical = None
        self.listNominal = None
        self.listBinary = None
        self.test = None
        self.train = None
        self.train_path = train_path
        self.test_path = test_path
        self.classification_mode = classification_mode
        self.label_col_name = label_col_name
        self.read_data_frames()

    def read_data_frames(self):
        feature = ["duration", "protocol_type", "service", "flag", "src_bytes",
                   "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                   "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                   "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                   "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

        self.train = pd.read_csv(self.train_path, names=feature)
        self.test = pd.read_csv(self.test_path, names=feature)

        self.train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        self.test.drop(['num_outbound_cmds'], axis=1, inplace=True)

        # This dataset is clean version, so these two following lines does not necessary
        self.train.drop_duplicates(keep='first')
        self.test.drop_duplicates(keep='first')

        self.listBinary = ['land', 'logged_in', 'root_shell', 'su_attempted', 'is_host_login',
                           'is_guest_login']
        self.listNominal = ['protocol_type', 'service', 'flag']
        self.listNumerical = set(self.train.columns) - set(self.listNominal) - set(self.listBinary)
        self.listNumerical.remove(self.label_col_name)

    def label_mapping(self):
        """
        this function specifically is used for original dataset
        """
        if self.classification_mode == 'multi':
            self.train.label.replace(['normal.'], 'normal', inplace=True)
            self.train.label.replace(
                ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'Dos', inplace=True)
            self.train.label.replace(
                ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'R2L', inplace=True)
            self.train.label.replace(
                ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'Probe', inplace=True)
            self.train.label.replace(
                ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'],
                'U2R', inplace=True)

            self.test.label.replace(['normal.'], 'normal', inplace=True)
            self.test.label.replace(
                ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'Dos', inplace=True)
            self.test.label.replace(
                ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'R2L', inplace=True)
            self.test.label.replace(
                ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'Probe', inplace=True)
            self.test.label.replace(
                ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'],
                'U2R',
                inplace=True)
        elif self.classification_mode == 'binary':
            self.train[self.label_col_name] = self.train[self.label_col_name].apply(
                lambda x: 'attack.' if x != 'normal.' else x)
            self.test[self.label_col_name] = self.test[self.label_col_name].apply(
                lambda x: 'attack.' if x != 'normal.' else x)

    def scaling(self, normalization_method):
        train, test = self.train.copy(), self.test.copy()
        listContent = list(self.listNumerical)

        scaler = MinMaxScaler()
        if normalization_method == 'standardization':
            scaler = StandardScaler()

        scaler.fit(train[listContent].values)
        train[listContent] = scaler.transform(train[listContent].values)

        test[listContent] = scaler.transform(test[listContent].values)
        self.train, self.test = train, test

    def onehot_encoding(self):
        train_df, test_df = self.train, self.test
        categorical_column_list = list(train_df.select_dtypes(include='object').columns)

        one_hot_train = pd.get_dummies(train_df, columns=categorical_column_list, dtype='int')
        one_hot_test = pd.get_dummies(test_df, columns=categorical_column_list, dtype='int')

        # Combine one-hot encoded columns to make sure train and test have the same features
        all_columns = set(one_hot_train.columns) | set(one_hot_test.columns)
        df_train = pd.concat([one_hot_train, pd.DataFrame(columns=list(all_columns))]).fillna(0.0)
        df_test = pd.concat([one_hot_test, pd.DataFrame(columns=list(all_columns))]).fillna(0.0)

        train_label_col = df_train.pop(self.label_col_name)
        df_train.insert(len(df_train.columns), self.label_col_name, train_label_col)

        test_label_col = df_test.pop(self.label_col_name)
        df_test.insert(len(df_test.columns), self.label_col_name, test_label_col)

        train_cols = df_train.columns
        test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
        for col in test_sortedBasedOnTrain:
            test_sortedBasedOnTrain[col] = df_test[col]

        df_test = test_sortedBasedOnTrain

        self.train = df_train
        self.test = df_test

        return self.train, self.test

    def label_binarizing(self):
        if self.classification_mode == 'multi':
            label_encoder = LabelBinarizer().fit(self.train[self.label_col_name])
            TrainBinarizedLabel = label_encoder.transform(self.train[self.label_col_name])
            TrainBinarizedLabelDataFrame = pd.DataFrame(TrainBinarizedLabel,
                                                        columns=label_encoder.classes_)
            self.train = pd.concat([self.train.drop([self.label_col_name], axis=1),
                                    TrainBinarizedLabelDataFrame], axis=1)

            TestBinarizedLabel = label_encoder.transform(self.test[self.label_col_name])
            TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel,
                                                       columns=label_encoder.classes_)
            self.test = pd.concat([self.test.drop([self.label_col_name], axis=1), TestBinarizedLabelDataFrame], axis=1)

        elif self.classification_mode == 'binary':
            label_mapping = {'normal.': 0, 'attack.': 1}
            self.train[self.label_col_name] = self.train[self.label_col_name].map(label_mapping)
            self.test[self.label_col_name] = self.test[self.label_col_name].map(label_mapping)
            self.train[self.label_col_name] = self.train[self.label_col_name].astype('uint8')
            self.test[self.label_col_name] = self.test[self.label_col_name].astype('uint8')

    def sort_columns(self):
        train_cols = self.train.columns
        test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
        for col in test_sortedBasedOnTrain:
            test_sortedBasedOnTrain[col] = self.test[col]

        self.test = test_sortedBasedOnTrain

        if list(self.train.columns) == list(self.test.columns):
            cols = list(self.train.columns)
            cols.remove(self.label_col_name)
            cols.append(self.label_col_name)
            self.train = self.train[cols]
            self.test = self.test[cols]

    def shuffle(self):
        self.train = self.train.sample(frac=1).reset_index(drop=True)

    def save_data_frames(self, output_path):
        train_file_name = f'train_{self.classification_mode}.csv'
        test_file_name = f'test_{self.classification_mode}.csv'

        if output_path is not None:
            self.train.to_csv(os.path.join(output_path, train_file_name), index=False)
            self.test.to_csv(os.path.join(output_path, test_file_name), index=False)
            print(f'Dataframes Saved in: {output_path}')

    def get_data_frames(self):
        return self.train, self.test


if __name__ == "__main__":
    set_seed(0)
    base_path = Path(__file__).resolve().parent.joinpath('KDDCUP')
    train_file_path = base_path.joinpath('original', 'kddcup.data_10_percent_corrected')
    test_file_path = base_path.joinpath('original', 'corrected.gz')
    classification_mode = 'binary'
    # classification_mode = 'multi'

    preprocess = BuildDataFrames(train_path=str(train_file_path),
                                 test_path=str(test_file_path),
                                 classification_mode=classification_mode,
                                 label_col_name='label')

    preprocess.label_mapping()
    preprocess.label_binarizing()
    preprocess.scaling(normalization_method='normalization')
    preprocess.onehot_encoding()
    preprocess.shuffle()
    preprocess.save_data_frames(base_path)
    train_preprocessed, test_preprocessed = preprocess.get_data_frames()
    print(train_preprocessed.head())
    print(test_preprocessed.head())

    X, y = parse_data(train_preprocessed, dataset_name='KDDCUP', classification_mode=classification_mode)
    print(f'train shape: x=>{X.shape}, y=>{y.shape}')

    X, y = parse_data(test_preprocessed, dataset_name='KDDCUP', classification_mode=classification_mode)
    print(f'test shape: x=>{X.shape}, y=>{y.shape}')
