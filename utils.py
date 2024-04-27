import os
import time
from functools import wraps

import numpy as np
from pathlib import Path
import pandas as pd
from keras.callbacks import Callback
from pandas import DataFrame
import tensorflow as tf
import random
from keras.optimizers import Adam, SGD, RMSprop


def parse_data(df, dataset_name: str, classification_mode: str, mode: str = 'np'):
    classes = []
    if classification_mode == 'binary':
        classes = df.columns[-1:]
    elif classification_mode == 'multi':
        if dataset_name in ['NSL_KDD', 'KDDCUP']:
            classes = df.columns[-5:]
        elif dataset_name == 'UNSW':
            classes = df.columns[-10:]
        elif dataset_name == 'CICIDS':
            classes = df.columns[-15:]

    assert classes is not None, 'Something Wrong!!\nno class columns could be extracted from dataframe'
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(c) for c in list(classes)])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb


def get_result(cm):
    tp = cm[0][0]  # normal as normal
    fp = cm[0][1]  # normal predicted as attack
    fn = cm[1][0]  # attack predicted as normal
    tn = cm[1][1]  # attack as attack

    OA = (tp + tn) / (tn + fn + fp + tp)
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (tn + fp)

    return {"OA": OA, "P": P, "R": R, "F1": F1, "FAR": FAR}


def set_seed(seed):
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of '{func.__name__}': {execution_time} seconds")
        return result

    return wrapper


def shuffle_dataframe(dataframe_path):
    dataframe = pd.read_csv(dataframe_path)
    shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    base_path = os.path.dirname(dataframe_path)
    base_name = os.path.splitext(os.path.basename(dataframe_path))[0]
    file_extension = os.path.splitext(dataframe_path)[-1]
    new_file_name = Path(base_path).joinpath(base_name + file_extension)

    shuffled_dataframe.to_csv(new_file_name, index=False)

    print(f"Shuffled DataFrame saved as {new_file_name}")


def save_dataframe(dataframe: DataFrame, save_path: Path, dataframe_type: str = 'train',
                   classification_mode: str = 'binary') -> None:
    file_name = dataframe_type
    if classification_mode == 'binary':
        file_name = file_name + '_binary'
    elif classification_mode == 'multi':
        file_name = file_name + '_multi'
    file_path = os.path.join(save_path, file_name + '.csv')
    dataframe.to_csv(file_path, index=False)
    print('SAVED:', file_path)


def sort_columns(train_df: DataFrame, test_df: DataFrame, label_col_name: str) -> (DataFrame, DataFrame):
    train_cols = train_df.columns
    test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
    for col in test_sortedBasedOnTrain:
        test_sortedBasedOnTrain[col] = test_df[col]

    test_df = test_sortedBasedOnTrain

    try:
        train_df = train_df[[col for col in train_df.columns if col != label_col_name] + [label_col_name]]
        test_df = test_df[[col for col in test_df.columns if col != label_col_name] + [label_col_name]]
    except:
        print('COULD NOT MOVE LABEL COL TO LAST COLUMN OF DATAFRAME')

    return train_df, test_df


def shuffle(dataframe: DataFrame):
    return dataframe.sample(frac=1).reset_index(drop=True)


class OptimizerFactory:
    def __init__(self,
                 opt: str = 'adam',
                 lr_schedule: bool = True,
                 len_dataset: int = 494021,
                 epochs: int = 50,
                 batch_size: int = 100,
                 init_lr: float = 0.1,
                 final_lr: float = 0.00001):
        self.opt = opt
        self.lr_schedule = lr_schedule
        self.len_dataset = len_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.final_lr = final_lr

    def lr_scheduler(self):
        pretraining_learning_rate_decay_factor = (self.final_lr / self.init_lr) ** (1 / self.epochs)
        pretraining_steps_per_epoch = int(self.len_dataset / self.batch_size)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=pretraining_steps_per_epoch,
            decay_rate=pretraining_learning_rate_decay_factor,
            staircase=True)
        return lr_scheduler

    def get_opt(self):
        if self.opt == 'adam':
            if self.lr_schedule:
                return Adam(self.lr_scheduler())
            else:
                return Adam(learning_rate=self.init_lr)

        elif self.opt == 'sgd':
            if self.lr_schedule:
                return SGD(self.lr_scheduler())
            else:
                return SGD(learning_rate=0.1, decay=0.1, momentum=.95, nesterov=True)

        elif self.opt == 'rmsprop':
            if self.lr_schedule:
                return RMSprop(self.lr_schedule)
            else:
                return RMSprop(learning_rate=self.init_lr)


class GetEpoch(Callback):
    def __init__(self, monitor='val_loss'):
        super(GetEpoch, self).__init__()
        self.monitor = monitor
        self.stopped_epoch = int

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print('CALLBACK DOES NOT WORK PROPERLY!')
            return

        self.stopped_epoch = epoch + 1


if __name__ == "__main__":
    # df_path = 'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\dataset\\' \
    #           'KDDCUP\\Train_standard.csv'
    # df = pd.read_csv(df_path)
    # shuffle_dataframe(df)

    TP = 709442

    FP = 13262

    FN = 7133

    TN = 170163

    cm = [[TP, FP], [FN, TN]]
    res = get_result(cm)
    print(res)
