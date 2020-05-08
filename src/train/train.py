import warnings
warnings.filterwarnings("ignore")

import time
import os
from os.path import join
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model, save_model
from utils import utils
import model_skeleton.featuristic as featuristic
import model_skeleton.malfusion as malfusion
import model_skeleton.echelon as echelon
import model_skeleton.echelon_multi as echelon_multi
from keras import optimizers
from trend import activation_trend_identification as ati
import config.constants as cnst
from .train_args import DefaultTrainArguments
from plots.plots import plot_history
from predict import predict
from predict.predict_args import Predict as pObj, DefaultPredictArguments, QStats
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
from analyzers.collect_available_sections import collect_sections
import random
from plots.plots import display_probability_chart
from analyzers.collect_exe_files import get_partition_data, partition_pkl_files
import gc


# ####################################
# Set USE_GPU=FALSE to RUN IN CPU ONLY
# ####################################
# '''print('GPU found') if tf.test.gpu_device_name() else print("No GPU found")'''
if not cnst.USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def train(args):
    train_steps = len(args.t1_x_train) // args.t1_batch_size
    args.t1_train_steps = train_steps - 1 if len(args.t1_x_train) % args.t1_batch_size == 0 else train_steps + 1

    if args.t1_x_val is not None:
        val_steps = len(args.t1_x_val) // args.t1_batch_size
        args.t1_val_steps = val_steps - 1 if len(args.t1_x_val) % args.t1_batch_size == 0 else val_steps + 1

    args.t1_ear = EarlyStopping(monitor='acc', patience=10)
    args.t1_mcp = ModelCheckpoint(join(args.save_path, args.t1_model_name),
                               monitor="acc", save_best_only=args.save_best, save_weights_only=False)

    history = args.t1_model_base.fit_generator(
        utils.data_generator(args.train_partition, args.t1_x_train, args.t1_y_train, args.t1_max_len, args.t1_batch_size, args.t1_shuffle),
        class_weight=args.t1_class_weights,
        steps_per_epoch=args.t1_train_steps,
        epochs=args.t1_epochs,
        verbose=args.t1_verbose,
        callbacks=[args.t1_ear, args.t1_mcp]
        # , validation_data=utils.data_generator(args.t1_x_val, args.t1_y_val, args.t1_max_len, args.t1_batch_size,
        # args.t1_shuffle) , validation_steps=val_steps
    )

    # plot_history(history, cnst.TIER1)
    return history


def train_by_section(args):
    train_steps = len(args.t2_x_train)//args.t2_batch_size
    args.t2_train_steps = train_steps - 1 if len(args.t2_x_train) % args.t2_batch_size == 0 else train_steps + 1

    if args.t2_x_val is not None:
        val_steps = len(args.t2_x_val) // args.t2_batch_size
        args.t2_val_steps = val_steps - 1 if len(args.t2_x_val) % args.t2_batch_size == 0 else val_steps + 1

    args.t2_ear = EarlyStopping(monitor='acc', patience=10)
    args.t2_mcp = ModelCheckpoint(join(args.save_path, args.t2_model_name),
                               monitor="acc", save_best_only=args.save_best, save_weights_only=False)

    # Check MAX_LEN modification is needed - based on proportion of section vs whole file size
    # args.max_len = cnst.MAX_FILE_SIZE_LIMIT + (cnst.CONV_WINDOW_SIZE * len(args.q_sections))
    history = args.t2_model_base.fit_generator(
        utils.data_generator_by_section(args.b1_train_partition, args.q_sections, args.train_section_map, args.t2_x_train, args.t2_y_train, args.t2_max_len, args.t2_batch_size, args.t2_shuffle),
        class_weight=args.t2_class_weights,
        steps_per_epoch=len(args.t2_x_train)//args.t2_batch_size + 1,
        epochs=args.t2_epochs,
        verbose=args.t2_verbose,
        callbacks=[args.t2_ear, args.t2_mcp]
        # , validation_data=utils.data_generator_by_section(args.q_sections, args.t2_x_val, args.t2_y_val
        # , args.t2_max_len, args.t2_batch_size, args.t2_shuffle)
        # , validation_steps=args.val_steps
    )
    # plot_history(history, cnst.TIER2)
    return history


def train_featuristic(model, model_name, x_train, x_val, y_train, y_val, class_weights, save_path,  batch_size, verbose, epochs, shuffle, features_to_drop=None, save_best=True):
    history = model.fit_generator(
        utils.data_generator_by_features(x_train, y_train, batch_size, shuffle, features_to_drop)
        , class_weight=class_weights
        , steps_per_epoch=len(x_train)
        , epochs=epochs
        , verbose=verbose
        , callbacks=[ear, mcp]
        , validation_data=utils.data_generator_by_features(x_val, y_val, batch_size, shuffle, features_to_drop)
        , validation_steps=len(x_val)
    )
    return history


def train_fusion(model, model_name, x_train, x_val, y_train, y_val, class_weights, max_len, save_path, batch_size, verbose, epochs, shuffle, save_best=True):
    history = model.fit_generator(
        utils.data_generator_by_fusion(x_train, y_train, max_len, batch_size, shuffle)
        , class_weight=class_weights
        , steps_per_epoch=len(x_train)
        , epochs=epochs
        , verbose=verbose
        , callbacks=[ear, mcp]
        , validation_data=utils.data_generator_by_fusion(x_val, y_val, max_len, batch_size, shuffle)
        , validation_steps=len(x_val)
    )
    return history


def get_model1(args):
    # prepare TIER-1 model
    model1 = None
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER1:
            print("[ CAUTION ] : Resuming with pretrained model for TIER1 - "+args.pretrained_t1_model_name)
            model1 = load_model(args.model_path + args.pretrained_t1_model_name)
        else:
            print("[ CAUTION ] : Resuming with old model")
            model1 = load_model(args.model_path + args.t1_model_name)
    else:
        if args.byte:
            model1 = echelon.model(args.t1_max_len, args.t1_win_size)
        elif args.featuristic:
            model1 = featuristic.model(args.total_features)
        elif args.fusion:
            model1 = malfusion.model(args.max_len, args.win_size)

        # ##################################################################################################################
        #                                                  Optimization
        # ##################################################################################################################
        # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # ##################################################################################################################
        #                                             Hyper-parameter Tuning
        # ##################################################################################################################
        # param_dict = {'lr': [0.00001, 0.0001, 0.001, 0.1]}
        # model_gs = GridSearchCV(model, param_dict, cv=10)

    # model1.summary()
    return model1


def get_model2(args):
    # prepare TIER-2 model
    model2 = None
    if args.resume:
        if cnst.USE_PRETRAINED_FOR_TIER2:
            print("[ CAUTION ] : Resuming with pretrained model for TIER2 - "+args.pretrained_t2_model_name)
            model2 = load_model(args.model_path + args.pretrained_t2_model_name)
        else:
            print("[ CAUTION ] : Resuming with old model")
            model2 = load_model(args.model_path + args.t2_model_name)
    else:
        # print("*************************** CREATING new model *****************************")
        if args.byte:
            model2 = echelon.model(args.t2_max_len, args.t2_win_size)
        elif args.featuristic:
            model2 = featuristic.model(len(args.selected_features))
        elif args.fusion:
            model2 = malfusion.model(args.max_len, args.win_size)

        # optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = optimizers.Adam(lr=0.001)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # model2.summary()
    return model2


def train_tier1(args):
    print("************************ TIER 1 TRAINING - STARTED ****************************       # Samples:", len(args.t1_x_train))
    if args.tier1:
        if args.byte:             history = train(args)
        if args.featuristic:      history = train_featuristic(args)
        if args.fusion:           history = train_fusion(args)
    print("************************ TIER 1 TRAINING - ENDED   ****************************")


def train_tier2(args):
    # print("************************ TIER 2 TRAINING - STARTED ****************************")
    if args.tier2:
        if args.byte:             section_history = train_by_section(args)
        if args.featuristic:      history = train_featuristic(args)
        if args.fusion:           history = train_fusion(args)
    # print("************************ TIER 2 TRAINING - ENDED   ****************************")


# ######################################################################################################################
# OBJECTIVES:
#     1) Train Tier-1 and select its decision threshold for classification using Training data
#     2) Perform ATI over training data and select influential sections to be used by Tier-2
#     3) Train Tier-2 on selected features
#     4) Save trained models for Tier-1 and Tier-2
# ######################################################################################################################


def init(model_idx, traindata, valdata, fold_index):
    t_args = DefaultTrainArguments()

    # limit gpu memory
    if t_args.limit > 0: utils.limit_gpu_memory(t_args.limit)
    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:                 t_args.byte = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:        t_args.featuristic = True
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:             t_args.fusion = True

    t_args.t1_model_name = cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5"
    t_args.t2_model_name = cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5"

    # print("######################################   TRAINING TIER-1  ###############################################")
    t_args.t1_x_train, t_args.t1_x_val, t_args.t1_y_train, t_args.t1_y_val = traindata.xdf.values, valdata.xdf.values, traindata.ydf.values, valdata.ydf.values
    t_args.t1_class_weights = class_weight.compute_class_weight('balanced', np.unique(t_args.t1_y_train), t_args.t1_y_train)  # Class Imbalance Tackling - Setting class weights
    t_args.t1_model_base = get_model1(t_args)

    # ~~~~~~~~~~~~~~~~~~~
    q_sections_by_q_criteria = {0: ['.header']}  
    if not cnst.SKIP_TIER1_TRAINING:
        t_args.train_partition = get_partition_data("train", fold_index)
        train_tier1(t_args)
        print("Initiating garbage collection")
        del t_args.train_partition  # Release Memory
        gc.collect()
        print("Completed garbage collection")
    # ~~~~~~~~~~~~~~~~~~~

    # TIER-1 PREDICTION OVER TRAINING DATA [Select THD1]
    print("*** Prediction over Validation data in TIER-1 to select THD1 and Boosting Bound")
    predict_t1_val_data = pObj(cnst.TIER1, cnst.TIER1_TARGET_FPR, valdata.xdf.values, valdata.ydf.values)
    predict_t1_val_data.partition = get_partition_data("val", fold_index)
    predict_t1_val_data = predict.predict_tier1(model_idx, predict_t1_val_data, fold_index)
    print("Initiating garbage collection")
    del predict_t1_val_data.partition  # Release Memory
    gc.collect()
    print("Completed garbage collection")

    val_b1datadf = pd.concat([pd.DataFrame(predict_t1_val_data.xB1), pd.DataFrame(predict_t1_val_data.yB1)], axis=1)
    val_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None, index=None)

    partition_pkl_files("b1_val", fold_index, val_b1datadf.iloc[:, 0])

    print("\n*** Prediction over Training data in TIER-1 to generate B1 data for TIER-2 Training")
    predict_t1_train_data = pObj(cnst.TIER1, cnst.TIER1_TARGET_FPR, traindata.xdf.values, traindata.ydf.values)
    predict_t1_train_data.thd = predict_t1_val_data.thd
    predict_t1_train_data.boosting_upper_bound = predict_t1_val_data.boosting_upper_bound
    predict_t1_train_data.partition = get_partition_data("train", fold_index)
    predict_t1_train_data = predict.predict_tier1(model_idx, predict_t1_train_data, fold_index)
    print("Initiating garbage collection")
    del predict_t1_train_data.partition  # Release Memory
    gc.collect()
    print("Completed garbage collection")

    train_b1datadf = pd.concat([pd.DataFrame(predict_t1_train_data.xB1), pd.DataFrame(predict_t1_train_data.yB1)], axis=1)
    train_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_"+str(fold_index)+"_pkl.csv", header=None, index=None)

    partition_pkl_files("b1_train", fold_index, train_b1datadf.iloc[:, 0])

    print("Loading stored B1 Data from Training set to train Tier-2 model")
    train_b1datadf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_train_"+str(fold_index)+"_pkl.csv", header=None)  # xxs.csv
    t_args.t2_x_train, t_args.t2_y_train = train_b1datadf.iloc[:, 0], train_b1datadf.iloc[:, 1]
    t_args.b1_train_partition = get_partition_data("b1_train", fold_index)

    # ATI PROCESS - SELECTING QUALIFIED_SECTIONS - ### Pass B1 data
    if not cnst.SKIP_ATI_PROCESSING:
        print("Performing ATI over B1 data of Training set")
        # t_args.t2_x_train, t_args.t2_y_train = predict_t1_train_data.xB1, predict_t1_train_data.yB1
        q_sections_by_q_criteria = ati.init(t_args) if t_args.ati else None

    print("Collecting section map over B1 data of Training set")
    # t_args.train_section_map = collect_sections(t_args.t2_x_train, t_args.t2_y_train)
    # print("Train section map:\n", t_args.train_section_map)

    print("************************ TIER 2 TRAINING - STARTED ****************************       # Samples:", len(t_args.t2_x_train))
    if cnst.DO_SUBSAMPLING:
        ben_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.BENIGN].tolist()
        mal_idx = t_args.t2_y_train.index[t_args.t2_y_train == cnst.MALWARE].tolist()
        t_args.t2_x_train = pd.concat([t_args.t2_x_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_x_train.loc[mal_idx]], ignore_index=True)
        t_args.t2_y_train = pd.concat([t_args.t2_y_train.loc[random.sample(ben_idx, len(mal_idx))], t_args.t2_y_train.loc[mal_idx]], ignore_index=True)
        print("Sub-sampling complete for Tier-2 training data")

    # Need to decide the TRAIN:VAL ratio for tier2
    t_args.t2_x_val, t_args.t2_y_val = None, None
    t_args.t2_class_weights = class_weight.compute_class_weight('balanced', np.unique(t_args.t2_y_train), t_args.t2_y_train)  # Class Imbalance Tackling - Setting class weights
    t_args.t2_model_base = get_model2(t_args)

    # Iterate through different q_criterion to find the best suitable sections for training TIER-2
    # Higher the training TPR on B1, more suitable the set of sections are - for use in Tier-2
    #predict_t2_train_data = pObj(cnst.TIER2, cnst.TIER2_TARGET_FPR, t_args.t2_x_train, t_args.t2_y_train)
    #t2_fpr = cnst.OVERALL_TARGET_FPR - predict_t1_train_data.fpr 
    #t2_fpr = t2_fpr if 0 < t2_fpr < 1 else cnst.TIER2_TARGET_FPR

    t2_fpr = cnst.TIER2_TARGET_FPR
    print("Updated Tier-2 target FPR:", t2_fpr)

    #print("Loading stored B1 Data from Training set for Tier-2 Training")
    #predict_t2_train_data = pObj(cnst.TIER2, t2_fpr, t_args.t2_x_train, t_args.t2_y_train)
    print("Loading stored B1 Data from Validation set for THD2 selection")
    val_b1datadf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_val_"+str(fold_index)+"_pkl.csv", header=None)  # xxs.csv
    predict_t2_val_data = pObj(cnst.TIER2, t2_fpr, val_b1datadf.iloc[:, 0], val_b1datadf.iloc[:, 1])
    predict_t2_val_data.partition = get_partition_data("b1_val", fold_index)

    thd2 = 0
    maxdiff = 0
    q_sections_selected = None
    q_criterion_selected = None
    best_t2_model = None
    predict_args = DefaultPredictArguments()
    print(q_sections_by_q_criteria.keys())

    qstats = QStats(cnst.PERCENTILES, q_sections_by_q_criteria.keys(), q_sections_by_q_criteria.values())
    for q_criterion in q_sections_by_q_criteria:

        print("\n", q_sections_by_q_criteria.keys(), "\nChecking Q_Criterion: {:6.2f}".format(q_criterion), q_sections_by_q_criteria[q_criterion])

        # print("************************ Q_Criterion ****************************", q_criterion)
        t_args.q_sections = q_sections_by_q_criteria[q_criterion]
        predict_t2_val_data.q_sections = q_sections_by_q_criteria[q_criterion]
        predict_t2_val_data.predict_section_map = t_args.train_section_map

        # TIER-2 TRAINING & PREDICTION OVER B1 DATA for current set of q_sections
        # Retrieve TPR at FPR=0 or relax till 1%
        if not cnst.SKIP_TIER2_TRAINING:
            train_tier2(t_args)

        # predict_t2_train_data.thd = None
        # predict_t2_train_data = predict.predict_tier2(model_idx, predict_t2_train_data, fold_index)
        predict_t2_val_data.thd = None
        predict_t2_val_data = predict.predict_tier2(model_idx, predict_t2_val_data, fold_index)
        display_probability_chart(predict_t2_val_data.ytrue, predict_t2_val_data.yprob, predict_t2_val_data.thd, "Training_TIER2_PROB_PLOT_" + str(fold_index + 1) + "{:6.2f}".format(q_criterion))
        print("FPR: {:6.2f}".format(predict_t2_val_data.fpr), "TPR: {:6.2f}".format(predict_t2_val_data.tpr), "\tTHD2: {:6.2f}".format(predict_t2_val_data.thd))

        curdiff = predict_t2_val_data.tpr - predict_t2_val_data.fpr
        if curdiff != 0 and curdiff > maxdiff:
            maxdiff = curdiff

            q_criterion_selected = q_criterion
            best_t2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
            thd2 = predict_t2_val_data.thd
            q_sections_selected = q_sections_by_q_criteria[q_criterion]
            print("Best Q-criterion so far . . . ", q_criterion_selected, predict_t2_val_data.thd, predict_t2_val_data.fpr, predict_t2_val_data.tpr)

        qstats.thds.append(predict_t2_val_data.thd)
        qstats.fprs.append(predict_t2_val_data.fpr)
        qstats.tprs.append(predict_t2_val_data.tpr)

    # Save the best model found
    try:
        best_t2_model.save(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
        # save_model(model=best_t2_model, filepath=predict_args.model_path + cnst.TIER2_MODELS[model_idx],
        # save_weights_only=False, overwrite=True)
        # for section in q_sections_selected:
        #    print(section)
    except Exception as e:
        print("Best Model not available to save - ", str(e))

    print("Percentile\t#Sections\tQ-Criterion\tTHD\t\tFPR\t\tTPR\t\t[TPR-FPR]")
    for i,p in enumerate(cnst.PERCENTILES):
        print(str(qstats.percentiles[i])+"\t\t"+str(len(list(qstats.sections)[i]))+"\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}\t\t{:6.2f}".format(list(qstats.qcriteria)[i], qstats.thds[i], qstats.fprs[i], qstats.tprs[i], qstats.tprs[i]-qstats.fprs[i]))

    # Get the sections that had maximum TPR and low FPR over B1 training data as Final Qualified sections
    print("\n\tBest Q_Criterion:", q_criterion_selected, "Related Q_Sections:", q_sections_selected)

    print("Initiating garbage collection")
    del t_args.b1_train_partition  # Release Memory
    del predict_t2_val_data.partition  # Release Memory
    gc.collect()
    print("Completed garbage collection")

    print("************************ TIER 2 TRAINING - ENDED   ****************************")
    # return None, None, thd2, q_sections_selected, t_args.train_section_map
    return predict_t1_train_data.thd, predict_t1_train_data.boosting_upper_bound, thd2, q_sections_selected, t_args.train_section_map


if __name__ == '__main__':
    init()

'''with open(join(t_args.save_path, 'history\\history'+section+'.pkl'), 'wb') as f:
    print(join(t_args.save_path, 'history\\history'+section+'.pkl'))
    pickle.dump(section_history.history, f)'''
