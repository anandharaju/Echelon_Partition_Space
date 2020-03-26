import pandas as pd
import numpy as np
from train import train
from predict import predict
import time
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import config.constants as cnst
import analyzers.analyze_dataset as analyzer
from plots.plots import plot_vars as pv
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
from datetime import datetime
from plots.auc import plot_cv_auc
from plots.auc import cv as cv_info


def generate_cv_folds_data(dataset_path):
    mastertraindata = analyzer.ByteData()
    testdata = analyzer.ByteData()
    valdata = analyzer.ByteData()
    cv_obj = cv_info()

    # Analyze Data set - Required only for Huawei pickle files
    if cnst.GENERATE_BENIGN_MALWARE_FILES: analyzer.analyze_dataset(cnst.CHECK_FILE_SIZE)
    # Load Data set info from CSV files
    adata, bdata, mdata = analyzer.load_dataset(dataset_path)

    # SPLIT TRAIN AND TEST DATA
    # Save them in files -- fold wise
    skf = StratifiedKFold(n_splits=cnst.CV_FOLDS, shuffle=True, random_state=cnst.RANDOM_SEED)
    for index, (master_train_indices, test_indices) in enumerate(skf.split(adata.xdf, adata.ydf)):
        mastertraindata.xdf, testdata.xdf = adata.xdf[master_train_indices], adata.xdf[test_indices]
        mastertraindata.ydf, testdata.ydf = adata.ydf[master_train_indices], adata.ydf[test_indices]
        mastertraindata.xdf, valdata.xdf, mastertraindata.ydf, valdata.ydf = train_test_split(mastertraindata.xdf, mastertraindata.ydf, test_size=0.00001, stratify=mastertraindata.ydf) # 0.00005

        cv_obj.train_data[index] = mastertraindata
        cv_obj.val_data[index] = valdata
        cv_obj.test_data[index] = testdata

        pd.concat([mastertraindata.xdf, mastertraindata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH + "\\data\\master_train_"+str(index)+"_pkl.csv", header=None, index=None)
        pd.concat([valdata.xdf, valdata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH + "\\data\\master_val_" + str(index)+ "_pkl.csv", header=None, index=None)
        pd.concat([testdata.xdf, testdata.ydf], axis=1).to_csv(cnst.PROJECT_BASE_PATH+"\\data\\master_test_"+str(index)+ "_pkl.csv", header=None, index=None)

    return cv_obj


def train_predict(model_idx, dataset_path=None):
    tst = time.time()
    print("START TIME  [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
    cv_obj = generate_cv_folds_data(dataset_path)

    for fold_index in range(cnst.CV_FOLDS):
        traindata = cv_obj.train_data[fold_index]
        valdata = cv_obj.val_data[fold_index]
        testdata = cv_obj.test_data[fold_index]

        if fold_index == 0:
            train_len = traindata.xdf.shape[0]
            val_len = valdata.xdf.shape[0]
            test_len = testdata.xdf.shape[0]

        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [ CV-FOLD " + str(fold_index + 1) + "/" + str(cnst.CV_FOLDS) + " ]", "Training: " + str(train_len), "Validation: " + str(val_len), "Testing: " + str(test_len))

        # traindatadf = pd.read_csv(cnst.PROJECT_BASE_PATH + "\\data\\master_train_pkl.csv", header=None)
        # testdatadf = pd.read_csv(cnst.PROJECT_BASE_PATH + "\\data\\master_test_pkl.csv", header=None)
        # mastertraindata.xdf, testdata.xdf = traindatadf.iloc[:, 0], testdatadf.iloc[:, 0]
        # mastertraindata.ydf, testdata.ydf = traindatadf.iloc[:, 1], testdatadf.iloc[:, 1]

        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # traindata.xdf, traindata.ydf = mastertraindata.xdf, mastertraindata.ydf
        # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????

        # TIER 1&2 Training + ATI 24.10
        thd1, thd2, q_sections = 23.60, 24.10, ['SUPPORT','','/41','.petite','BSS','bero^fr','.didata','imports','.clam01','.adata','.flat','.code','.data2','.wtq','.data','.lif','.FISHPEP','.nkh','.vmp0','.vc++','.MPRESS2','DATA','.textbss','.rmnet','.wixburn','.mjg','.trace','code','.RLPack','.arch','.imports','.clam03','.bT','.link','.text1','.spm','cji8','D','data','.rodata','.SF','.dtc','.aspack','.text','.zero','.sdata','relocs','.rrdata','.clam04','.dtd','.RGvaB','.MPRESS1','.tqn','.ifc','.phx','kkrunchy','.data5','/67','TYSGDGYS','.rsrc','.ydata','.text','.header','.','.sxdata','.itext','Shared','.clam02','.version','UPX2','.bGPSwOt','packerBY','.packed','.vmp1','EODE','.cdata','.rdata','.gda','.lrdata','.heb','.rloc','.iIEiZ','/29','.reloc','.vsp','/55','.crt0','.tc','petite','reloc','.data','.iPRMaL','.NewSec','.imdata','.res']  # T1TPR: 99.89 T2TPR: 2.11
        thd1, thd2, q_sections = train.init(model_idx, traindata, valdata, fold_index)

        # TIER 1&2 Prediction over Test data
        print("**********************  PREDICTION TIER 1&2 - STARTED  ************************")
        cv_obj = predict.init(model_idx, thd1, thd2, q_sections, testdata, cv_obj, fold_index)
        print("**********************  PREDICTION TIER 1&2 - ENDED    ************************")
        tet = time.time() - tst
        print("\nTIME ELAPSED :", str(int(tet) / 60), " minutes   [", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "]")
        # return

    cvdf = pd.DataFrame([cv_obj.t1_mean_fpr_auc, cv_obj.t1_mean_tpr_auc, cv_obj.recon_mean_fpr_auc, cv_obj.recon_mean_tpr_auc])
    scoredf = pd.DataFrame([np.mean(cv_obj.t1_mean_auc_score_restricted), np.mean(cv_obj.t1_mean_auc_score), np.mean(cv_obj.recon_mean_auc_score_restricted), np.mean(cv_obj.recon_mean_auc_score)])
    cvdf.to_csv(cnst.PROJECT_BASE_PATH+"\\out\\result\\mean_cv.csv", index=False, header=None)
    scoredf.to_csv(cnst.PROJECT_BASE_PATH + "\\out\\result\\score_cv.csv", index=False, header=None)
    plot_cv_auc(cv_obj)

        # predict.display_probability_chart(y_tier1, pred_tier1, tier1_thd, "TRAINING_TIER1_PROB_PLOT_" + str(fold + 1))
        # predict.display_probability_chart(y_recon, pred_recon, tier2_thd, "TRAINING_TIER2_PROB_PLOT_" + str(fold + 1))
        # Tier1 roc auc
        # Reconciled roc auc - TIER1 + TIER2
        # Train FPR vs Test FPR
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################

        # print("Tier-1 AUC:", metrics.roc_auc_score(y_tier1, pred_tier1))
        # print("Reconciled AUC:", metrics.roc_auc_score(y_reconciled, pred_reconciled))
        # print("Restricted Tier-1 AUC:", metrics.roc_auc_score(y_tier1, pred_tier1, max_fpr=0.02))
        # print("Restricted Reconciled AUC:", metrics.roc_auc_score(y_reconciled, pred_reconciled, max_fpr=0.02))

    # print("\n\nAverage results on " + str(CV_FOLDS) + "-FOLD CV")
    # print("AUC: {:0.5f}\t{:0.5f}".format(np.mean(list_auc1), np.mean(list_aucr)))
    # print("Average Acc  : {:0.2f}\t{:0.2f}".format(np.mean(list_acc1), np.mean(list_accr)))
    # print("Average BAcc : {:0.2f}\t{:0.2f}".format(np.mean(list_bacc1), np.mean(list_baccr)))
    # print("TPR: {:0.5f}\t{:0.5f}".format(np.mean(list_tpr1), np.mean(list_tprr)))
    # print("FPR: {:0.5f}\t{:0.5f}".format(np.mean(list_fpr1), np.mean(list_fprr)))
    # print("Average FNR: {:0.2f}\t{:0.2f}".format(np.mean(list_fnr1), np.mean(list_fnrr)))
    # print("Prediction Time: " + str(np.mean(list_pred_time))[:7] + " ms / sample")
    # print("Down selected features: " + str(int(np.mean(list_num_selected_features))) + " out of 53")
