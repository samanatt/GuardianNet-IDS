import sys
import argparse
import csv
import time
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from tensorflow import expand_dims

from tensorflow.python.keras.layers.recurrent_v2 import LSTM

from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from partial_autoencoder import partial_ae_factory
import os
from pathlib import Path
from configs.setting import setting
import json
import gc

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import parse_data, OptimizerFactory, set_seed, GetEpoch, get_result

# tf.config.set_visible_devices([], 'GPU')
# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')
# print(device_lib.list_local_devices())

config = dict()

XGlobal = list()
YGlobal = list()

XValGlobal = list()
YValGlobal = list()

XTestGlobal = list()
YTestGlobal = list()

SavedParameters = list()
SavedParametersAE = list()
Mode = str()
Name = str()
result_path = str()

tid = 0
best_loss = float('inf')
best_val_acc = 0
best_ae = None
best_params = dict()
load_previous_result = True
continue_loading = True


def DAE(params_ae, method: str = 'layer-wise'):
    tf.keras.backend.clear_session()
    print(params_ae)

    time_file = open(Path(config['save_path']).joinpath('time.txt'), 'w')
    train_time = 0
    tr_loss = 0
    val_loss = 0

    dataset_name = config['dataset']['name']
    epoch = config['autoencoder']['epoch']
    hidden_size = [params_ae['ae_unit'], params_ae['ae_unit'], params_ae['ae_unit']]
    batch_size = params_ae['ae_batch']
    hidden_activation = params_ae['ae_activation']
    out_activation = params_ae['ae_out_activation']
    loss_fn = params_ae['ae_loss']
    opt = params_ae['ae_optimizer']

    ae_filename = f'AE_{dataset_name}_' + '_'.join(
        map(str,
            hidden_size)) + f'_A-{hidden_activation}_OA-{out_activation}_{loss_fn}_E{epoch}_B{batch_size}_{opt}_{method}'

    BASE_DIR = Path(__file__).resolve().parent
    ae_path = os.path.join(BASE_DIR, 'trained_ae', f'{ae_filename}.keras')

    # ae_path = Path(
    #     f'C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\trained_ae\\{ae_filename}.keras')

    X_train = np.array(XGlobal)
    X_val = np.array(XValGlobal)

    if os.path.isfile(ae_path) and not config['autoencoder']['trainable']:
        deep_autoencoder = tf.keras.models.load_model(ae_path)
        print(f'DAE LOADED FROM: {ae_path}')
        try:
            tr_loss = deep_autoencoder.evaluate(X_train, X_train)
            val_loss = deep_autoencoder.evaluate(X_val, X_val)
            print(f'LOSS: {tr_loss}, VAL LOSS: {val_loss}')
        except:
            pass
    else:
        if method == 'standard':

            deep_autoencoder = tf.keras.models.Sequential()
            deep_autoencoder.add(tf.keras.Input(shape=(X_train.shape[1],)))

            for n in range(config['autoencoder']['n_hidden_layer']):
                deep_autoencoder.add(Dense(hidden_size[n],
                                           activation=params_ae['ae_activation'],
                                           name=f'encode{n + 1}',
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                                           bias_initializer=tf.keras.initializers.Zeros()))

            deep_autoencoder.add(Dense(X_train.shape[1],
                                       activation=params_ae['ae_out_activation'],
                                       name=f'out',
                                       kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                                       bias_initializer=tf.keras.initializers.Zeros()))

            opt_factory_ae = OptimizerFactory(opt=params_ae['ae_optimizer'],
                                              lr_schedule=config['autoencoder']['lr_schedule'],
                                              len_dataset=len(X_train),
                                              epochs=epoch,
                                              batch_size=params_ae['ae_batch'],
                                              init_lr=config['autoencoder']['initial_lr'],
                                              final_lr=config['autoencoder']['finial_lr'])

            sgd = opt_factory_ae.get_opt()
            deep_autoencoder.compile(loss=params_ae['ae_loss'], optimizer=sgd)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            ae_start_time = time.time()
            history = deep_autoencoder.fit(X_train, X_train,
                                           validation_data=(X_val, X_val),
                                           epochs=epoch,
                                           batch_size=params_ae['ae_batch'],
                                           shuffle=False,
                                           callbacks=[early_stop],
                                           verbose=2
                                           )
            ae_elapsed_time = time.time() - ae_start_time

            train_time = int(ae_elapsed_time)
            time_file.write(f'Autoencoder training (sec): {train_time}\n')

            stopped_epoch = early_stop.stopped_epoch

            tr_loss = history.history['loss'][stopped_epoch]
            val_loss = history.history['val_loss'][stopped_epoch]

            deep_autoencoder.save(Path(config['save_path']).joinpath("DAE.keras"))
            deep_autoencoder.save(ae_path)

        else:
            # ================= LAYER 1 =================
            autoencoder1, encoder1 = partial_ae_factory(in_shape=X_train.shape[1],
                                                        hidden_size=hidden_size[0],
                                                        activation=params_ae['ae_activation'])

            # ================= LAYER 2 =================
            autoencoder2, encoder2 = partial_ae_factory(in_shape=hidden_size[0],
                                                        hidden_size=hidden_size[1],
                                                        activation=params_ae['ae_activation'])
            # ================= LAYER 3 =================
            autoencoder3, encoder3 = partial_ae_factory(in_shape=hidden_size[1],
                                                        hidden_size=hidden_size[2],
                                                        activation=params_ae['ae_activation'])

            opt_factory_ae = OptimizerFactory(opt=params_ae['ae_optimizer'],
                                              lr_schedule=config['autoencoder']['lr_schedule'],
                                              len_dataset=len(X_train),
                                              epochs=epoch,
                                              batch_size=params_ae['ae_batch'],
                                              init_lr=config['autoencoder']['initial_lr'],
                                              final_lr=config['autoencoder']['finial_lr'])

            sgd1 = opt_factory_ae.get_opt()
            sgd2 = opt_factory_ae.get_opt()
            sgd3 = opt_factory_ae.get_opt()

            autoencoder1.compile(loss=params_ae['ae_loss'], optimizer=sgd1)
            autoencoder2.compile(loss=params_ae['ae_loss'], optimizer=sgd2)
            autoencoder3.compile(loss=params_ae['ae_loss'], optimizer=sgd3)

            encoder1.compile(loss=params_ae['ae_loss'], optimizer=sgd1)
            encoder2.compile(loss=params_ae['ae_loss'], optimizer=sgd2)
            encoder3.compile(loss=params_ae['ae_loss'], optimizer=sgd3)

            early_stop1 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop2 = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )
            early_stop_last = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            print('========== LAYER 1 ==========')
            pae_train_start_time = time.time()
            autoencoder1.fit(X_train, X_train,
                             validation_data=(X_val, X_val),
                             epochs=epoch,
                             batch_size=params_ae['ae_batch'],
                             shuffle=False,
                             callbacks=[early_stop1],
                             verbose=2
                             )

            print('========== LAYER 2 ==========')
            first_layer_code = encoder1.predict(X_train, verbose=2)
            first_layer_code_val = encoder1.predict(X_val, verbose=2)
            autoencoder2.fit(first_layer_code, first_layer_code,
                             validation_data=(first_layer_code_val, first_layer_code_val),
                             epochs=epoch,
                             batch_size=params_ae['ae_batch'],
                             shuffle=False,
                             callbacks=[early_stop2],
                             verbose=2
                             )

            print('========== LAYER 3 ==========')
            second_layer_code = encoder2.predict(first_layer_code, verbose=2)
            second_layer_code_val = encoder2.predict(first_layer_code_val, verbose=2)
            history = autoencoder3.fit(second_layer_code, second_layer_code,
                                       validation_data=(second_layer_code_val, second_layer_code_val),
                                       epochs=epoch,
                                       batch_size=params_ae['ae_batch'],
                                       shuffle=False,
                                       callbacks=[early_stop_last],
                                       verbose=2
                                       )

            pae_train_end_time = time.time()
            pae_train_elapsed_time = int(pae_train_end_time - pae_train_start_time)

            input_img = tf.keras.Input(shape=(X_train.shape[1],))
            encoded1_da = Dense(hidden_size[0], activation=params_ae['ae_activation'], name='encode1')(input_img)
            encoded2_da = Dense(hidden_size[1], activation=params_ae['ae_activation'], name='encode2')(encoded1_da)
            encoded3_da = Dense(hidden_size[2], activation=params_ae['ae_activation'], name='encode3')(encoded2_da)
            decoded1_da = Dense(X_train.shape[1], activation=params_ae['ae_out_activation'], name='out',
                                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                                bias_initializer=tf.keras.initializers.Zeros()
                                )(encoded3_da)

            deep_autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded1_da)

            sgd4 = opt_factory_ae.get_opt()
            deep_autoencoder.compile(loss=params_ae['ae_loss'], optimizer=sgd4)

            deep_autoencoder.get_layer('encode1').set_weights(autoencoder1.layers[1].get_weights())
            deep_autoencoder.get_layer('encode2').set_weights(autoencoder2.layers[1].get_weights())
            deep_autoencoder.get_layer('encode3').set_weights(autoencoder3.layers[1].get_weights())

            deep_autoencoder.get_layer('encode1').trainable = True
            deep_autoencoder.get_layer('encode2').trainable = True
            deep_autoencoder.get_layer('encode3').trainable = True

            early_stop_last = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=10,
                mode="auto",
                restore_best_weights=True
            )

            print('========== LAYER 4 ==========')
            outLayer_pae_start_time = time.time()
            history = deep_autoencoder.fit(X_train, X_train,
                                           validation_data=(X_val, X_val),
                                           epochs=epoch,
                                           batch_size=params_ae['ae_batch'],
                                           shuffle=False,
                                           callbacks=[early_stop_last],
                                           verbose=2
                                           )
            outLayer_pae_elapsed_time = time.time() - outLayer_pae_start_time

            train_time = int(pae_train_elapsed_time + outLayer_pae_elapsed_time)
            time_file.write(f'Autoencoder training (sec): {train_time}\n')

            stopped_epoch = early_stop_last.stopped_epoch

            tr_loss = history.history['loss'][stopped_epoch]
            val_loss = history.history['val_loss'][stopped_epoch]

            deep_autoencoder.save(Path(config['save_path']).joinpath("DAE.keras"))
            deep_autoencoder.save(ae_path)

    return deep_autoencoder, {"loss": tr_loss,
                              "val_loss": val_loss,
                              "ae_loss": params_ae['ae_loss'],
                              "ae_optimizer": params_ae['ae_optimizer'],
                              "ae_activation": params_ae['ae_activation'],
                              "ae_out_activation": params_ae['ae_out_activation'],
                              "ae_batch": params_ae['ae_batch'],
                              "ae_unit": params_ae['ae_unit'],
                              "train_time": train_time
                              }


def hyperopt_DAE(params_ae, method):
    global SavedParametersAE
    global best_loss
    global best_ae
    global best_params

    ae, param = DAE(params_ae, method)

    SavedParametersAE.append(param)
    # Save model
    if SavedParametersAE[-1]["val_loss"] < best_loss:
        print("new saved model:" + str(SavedParametersAE[-1]))
        best_ae = ae
        best_params = param
        ae.save(os.path.join(config['save_path'], Name.replace(".csv", "ae_model.h5")))
        del ae
        best_loss = SavedParametersAE[-1]["val_loss"]

    SavedParametersAE = sorted(SavedParametersAE, key=lambda i: i['val_loss'])

    try:
        with open((os.path.join(config['save_path'], 'best_result_ae.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParametersAE[0].keys())
            writer.writeheader()
            writer.writerows(SavedParametersAE)
    except IOError:
        print("I/O error")
    gc.collect()


def train_cf(x_train, y_train, x_val, y_val, params):
    global tid
    global best_ae

    tid += 1

    tf.keras.backend.clear_session()
    print(params)
    x_train = np.array(x_train)
    x_val = np.array(x_val)

    n_features = x_train.shape[2]

    cf = tf.keras.models.Sequential()
    if config['classifier']['name'] == 'BILSTM':

        cf.add(tf.keras.Input(shape=(1, n_features)))

        for n in range(config['classifier']['n_layer']):
            if n == config['classifier']['n_layer'] - 1:
                forward_layer = LSTM(units=params[f'unit{n + 1}'], use_bias=True, unroll=True,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal(
                                         seed=config['seed']),
                                     bias_initializer=tf.keras.initializers.Zeros())
                backward_layer = LSTM(units=params[f'unit{n + 1}'], go_backwards=True,
                                      unroll=True,
                                      kernel_initializer=tf.keras.initializers.GlorotNormal(
                                          seed=config['seed']),
                                      bias_initializer=tf.keras.initializers.Zeros())
            else:
                forward_layer = LSTM(units=params[f'unit{n + 1}'], return_sequences=True, use_bias=True, unroll=True,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal(
                                         seed=config['seed']),
                                     bias_initializer=tf.keras.initializers.Zeros())
                backward_layer = LSTM(units=params[f'unit{n + 1}'], return_sequences=True, go_backwards=True,
                                      unroll=True,
                                      kernel_initializer=tf.keras.initializers.GlorotNormal(
                                          seed=config['seed']),
                                      bias_initializer=tf.keras.initializers.Zeros())
            cf.add(Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode=params[f'merge_mode{n + 1}']))

        cf.add(Dropout(params['dropout']))
        cf.add(Dense(y_train.shape[1],
                     activation="softmax",
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                     bias_initializer=tf.keras.initializers.Zeros(),
                     # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                     # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                     # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                     ))

    elif config['classifier']['name'] == 'LSTM':

        cf.add(tf.keras.Input(shape=(1, n_features)))

        for n in range(config['classifier']['n_layer']):
            if n == config['classifier']['n_layer'] - 1:
                cf.add(LSTM(params[f'unit{n + 1}'],
                            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                            bias_initializer=tf.keras.initializers.Zeros()))
            else:
                cf.add(LSTM(params[f'unit{n + 1}'],
                            return_sequences=True,
                            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                            bias_initializer=tf.keras.initializers.Zeros()))

        cf.add(Dropout(params['dropout']))
        cf.add(Dense(y_train.shape[1],
                     activation="softmax",
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config['seed']),
                     bias_initializer=tf.keras.initializers.Zeros(),
                     # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                     # bias_regularizer=tf.keras.regularizers.L2(1e-4),
                     # activity_regularizer=tf.keras.regularizers.L2(1e-5)
                     ))

    trainable_params = sum([tf.size(w).numpy() for w in cf.trainable_variables])
    cf.compile(loss='categorical_crossentropy',
               optimizer=Adam(params["learning_rate"]),
               metrics=['acc'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      patience=10,
                                                      restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['checkpoint_path'],
                                                                                'best_model.h5'),
                                                          monitor='val_loss',
                                                          save_best_only=True)
    get_epoch = GetEpoch()

    train_start_time = time.time()
    cf.fit(
        x_train,
        y_train,
        epochs=config['classifier']["epoch"],
        verbose=2,
        validation_data=(x_val, y_val),
        batch_size=params["batch"],
        workers=4,
        callbacks=[early_stopping, get_epoch, model_checkpoint]
    )
    train_end_time = time.time()

    cf.load_weights(os.path.join(config['checkpoint_path'], 'best_model.h5'))
    Y_predicted = cf.predict(x_val, workers=4, verbose=2)

    y_val = np.argmax(y_val, axis=1)
    Y_predicted = np.argmax(Y_predicted, axis=1)

    cm_val = confusion_matrix(y_val, Y_predicted)
    results_val = get_result(cm_val)
    epochs = get_epoch.stopped_epoch

    del x_train, x_val, y_train, y_val, Y_predicted

    param = {
        "tid": tid,
        "n_params": trainable_params,
        "epochs": epochs,
        "train_time": int(train_end_time - train_start_time),
        "learning_rate": params["learning_rate"],
        "batch": params["batch"],
        "dropout": params["dropout"],
        "TP_val": cm_val[0][0],
        "FP_val": cm_val[0][1],
        "TN_val": cm_val[1][1],
        "FN_val": cm_val[1][0],
        "OA_val": results_val['OA'],
        "P_val": results_val['P'],
        "R_val": results_val['R'],
        "F1_val": results_val['F1'],
        "FAR_val": results_val['FAR'],
    }
    for n in range(config['classifier']['n_layer']):
        param[f'unit{n + 1}'] = params[f'unit{n + 1}']
        if config['classifier']['name'] == 'BILSTM':
            param[f'merge_mode{n + 1}'] = params[f'merge_mode{n + 1}']

    return cf, param


def hyperopt_cf(params):
    global SavedParameters
    global best_val_acc
    global best_ae
    global result_path
    global load_previous_result
    global continue_loading
    global tid

    if (result_path is not None) and continue_loading:
        result_table = pd.read_csv(result_path)

        tid += 1
        selected_row = result_table[result_table['tid'] == tid]
        print(selected_row)
        loss_hp = selected_row['F1_val'].values[0]
        loss_hp = -loss_hp
        if tid == len(result_table):
            continue_loading = False

        if load_previous_result:
            best_val_acc = result_table['F1_val'].max()

            result_table = result_table.sort_values('F1_val', ascending=False)
            SavedParameters = result_table.to_dict(orient='records')
            with open((os.path.join(config['save_path'], 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)

            load_previous_result = False

    else:
        print("start training")
        model, param = train_cf(XGlobal, YGlobal, XValGlobal, YValGlobal, params)

        print("start predicting")
        test_start_time = time.time()
        y_predicted = model.predict(XTestGlobal, workers=4, verbose=2)
        test_elapsed_time = time.time() - test_start_time

        y_predicted = np.argmax(y_predicted, axis=1)
        YTestGlobal_temp = np.argmax(YTestGlobal, axis=1)
        cm_test = confusion_matrix(YTestGlobal_temp, y_predicted)
        results_test = get_result(cm_test)

        tf.keras.backend.clear_session()

        SavedParameters.append(param)

        SavedParameters[-1].update({
            "test_time": int(test_elapsed_time),
            "TP_test": cm_test[0][0],
            "FP_test": cm_test[0][1],
            "FN_test": cm_test[1][0],
            "TN_test": cm_test[1][1],
            "OA_test": results_test['OA'],
            "P_test": results_test['P'],
            "R_test": results_test['R'],
            "F1_test": results_test['F1'],
            "FAR_test": results_test['FAR'],
        })

        if SavedParameters[-1]["F1_val"] > best_val_acc:
            print("new model saved:" + str(SavedParameters[-1]))
            model.save(os.path.join(config['save_path'], Name.replace(".csv", "_model.h5")))
            del model
            best_val_acc = SavedParameters[-1]["F1_val"]

        SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

        try:
            with open((os.path.join(config['save_path'], 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)
        except IOError:
            print("I/O error")

        loss_hp = -param["F1_val"]
        gc.collect()

    return {'loss': loss_hp, 'status': STATUS_OK}


def train_GuardianNet(dataset_name):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global best_ae
    global best_params

    global config

    config = setting(dataset_name=dataset_name, project='GuardianNet')

    # TODO: if these files exist read them, if not run preprocessing .py code for given dataset
    base_path = Path(__file__).resolve().parent
    train = pd.read_csv(os.path.join(base_path, 'dataset-processed', f'{dataset_name}_train_binary.csv'))
    test = pd.read_csv(os.path.join(base_path, 'dataset-processed', f'{dataset_name}_test_binary.csv'))

    XGlobal, YGlobal = parse_data(df=train,
                                  dataset_name=dataset_name,
                                  classification_mode=config['dataset']['classification_mode'])
    XTestGlobal, YTestGlobal = parse_data(df=test,
                                          dataset_name=dataset_name,
                                          classification_mode=config['dataset']['classification_mode'])

    YGlobal = tf.keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = tf.keras.utils.to_categorical(YTestGlobal, num_classes=2)

    XGlobal, XValGlobal, YGlobal, YValGlobal = train_test_split(XGlobal,
                                                                YGlobal,
                                                                test_size=config['dataset']['val_size'],
                                                                stratify=YGlobal,
                                                                random_state=config['seed']
                                                                )

    print('TRAINING SET SHAPE:', XGlobal.shape, YGlobal.shape)
    print('VALIDATION SET SHAPE:', XValGlobal.shape, YValGlobal.shape)
    print('TEST SET SHAPE:', XTestGlobal.shape, YTestGlobal.shape)

    ae_hyperparameters_to_optimize = {
        "ae_activation": config['autoencoder']['hidden_activation'],
        "ae_out_activation": config['autoencoder']['out_activation'],
        "ae_loss": config['autoencoder']['loss'],
        "ae_optimizer": config['autoencoder']['optimizer']
    }
    keys = list(ae_hyperparameters_to_optimize.keys())
    values = list(ae_hyperparameters_to_optimize.values())
    for combination in product(*values):
        params_ae = {keys[i]: combination[i] for i in range(len(keys))}
        params_ae['ae_unit'] = config['autoencoder']['unit']
        params_ae['ae_batch'] = config['autoencoder']['batch']
        hyperopt_DAE(params_ae, config['autoencoder']['method'])

    print(best_params)

    # best_ae = keras.models.load_model('C:\\Users\\Faraz\\PycharmProjects\\deep-layer-wise-autoencoder\\'
    #                                   'trained_ae\\'
    #                                   'AE_KDD_CUP99_128_128_128_A-tanh_OA-linear_mse_E150_B32_sgd_layer-wise.keras')

    print('DATASET RECONSTRUCTION USING LAYER-WISE AUTOENCODER')
    XGlobal = best_ae.predict(XGlobal, verbose=2)
    XValGlobal = best_ae.predict(XValGlobal, verbose=2)
    XTestGlobal = best_ae.predict(XTestGlobal, verbose=2)

    XGlobal = np.reshape(XGlobal, (-1, 1, XGlobal.shape[1]))
    XValGlobal = np.reshape(XValGlobal, (-1, 1, XValGlobal.shape[1]))
    XTestGlobal = np.reshape(XTestGlobal, (-1, 1, XTestGlobal.shape[1]))

    print('TRAINING SET SHAPE:', XGlobal.shape, YGlobal.shape)
    print('VALIDATION SET SHAPE:', XValGlobal.shape, YValGlobal.shape)
    print('TEST SET SHAPE:', XTestGlobal.shape, YTestGlobal.shape)

    tf.keras.backend.clear_session()

    cf_hyperparameters = {
        "batch": hp.choice("batch", config['classifier']['batch']),
        # "epoch": hp.choice("epoch", config['classifier']['EPOCH']),
        'dropout': hp.uniform("dropout", config['classifier']['min_dropout'], config['classifier']['max_dropout']),
        "learning_rate": hp.uniform("learning_rate", config['classifier']['min_lr'], config['classifier']['max_lr'])
    }
    for n in range(config['classifier']["n_layer"]):
        cf_hyperparameters[f'unit{n + 1}'] = hp.choice(f'unit{n + 1}', config['classifier']['unit'])
        if config['classifier']['name'] == 'BILSTM':
            cf_hyperparameters[f'merge_mode{n + 1}'] = hp.choice(f'merge_mode{n + 1}',
                                                                 config['classifier']['merge_mode'])

    trials = Trials()
    # spark_trials = SparkTrials()
    fmin(hyperopt_cf, cf_hyperparameters,
         trials=trials,
         algo=tpe.suggest,
         max_evals=config['classifier']['max_evals'],
         rstate=np.random.default_rng(config['seed']))

    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW', required=True,
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')
    parser.add_argument('--result', type=str, required=False,
                        help='path of hyper-parameter training result table .csv file')
    args = parser.parse_args()

    if args.result is not None:
        result_path = args.result
    else:
        result_path = None

    train_GuardianNet(args.dataset)
