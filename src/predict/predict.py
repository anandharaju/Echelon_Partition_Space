from utils import utils
import numpy as np
import pandas as pd
from sklearn import metrics
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.filter import filter_benign_fn_files, filter_malware_fp_files
from config import constants as cnst
from .predict_args import DefaultPredictArguments, Predict as pObj


def predict_byte(model, xfiles, args):
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict_generator(
        utils.data_generator(xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_byte_by_section(model, xfiles, q_sections, args):
    xlen = len(xfiles)
    pred_steps = xlen//args.batch_size if xlen % args.batch_size == 0 else xlen//args.batch_size + 1
    pred = model.predict_generator(
        utils.data_generator_by_section(q_sections, xfiles, np.ones(xfiles.shape), args.max_len, args.batch_size, shuffle=False),
        steps=pred_steps,
        verbose=args.verbose
        )
    return pred


def predict_by_features(model, fn_list, label, batch_size, verbose, features_to_drop=None):
    pred = model.predict_generator(
        utils.data_generator_by_features(fn_list, np.ones(fn_list.shape), batch_size, False, features_to_drop),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def predict_by_fusion(model, fn_list, label, batch_size, verbose):
    byte_sequence_max_len = model.input[0].shape[1]
    pred = model.predict_generator(
        utils.data_generator_by_fusion(fn_list, np.ones(fn_list.shape), byte_sequence_max_len, batch_size, shuffle=False),
        steps=len(fn_list),
        verbose=verbose
    )
    return pred


def calculate_prediction_metrics(predict_obj):
    predict_obj.ypred = (predict_obj.yprob >= (predict_obj.thd / 100)).astype(int)
    cm = metrics.confusion_matrix(predict_obj.ytrue, predict_obj.ypred, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    predict_obj.tpr = (tp / (tp + fn)) * 100
    predict_obj.fpr = (fp / (fp + tn)) * 100
    predict_obj.auc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred)
    predict_obj.rauc = metrics.roc_auc_score(predict_obj.ytrue, predict_obj.ypred, max_fpr=0.01)

    print("Threshold used for Prediction :", predict_obj.thd,
          "TPR: {:6.2f}".format(predict_obj.tpr),
          "\tFPR: {:6.2f}".format(predict_obj.fpr),
          "\tAUC: ", predict_obj.auc,
          "\tRst. AUC: ", predict_obj.rauc)
    return predict_obj


def select_decision_threshold(predict_obj):
    threshold = 0.0
    selected_threshold = 100
    temp_ypred = None
    TPR = None
    FPR = None
    while threshold <= 100.0:
        temp_ypred = (predict_obj.yprob >= (threshold / 100)).astype(int)
        cm = metrics.confusion_matrix(predict_obj.ytrue, temp_ypred, labels=[cnst.BENIGN, cnst.MALWARE])
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        TPR = (tp / (tp + fn)) * 100
        FPR = (fp / (fp + tn)) * 100
        if FPR <= predict_obj.target_fpr:
            selected_threshold = threshold
            # print("Selected Threshold: {:6.2f}".format(threshold), "TPR: {:6.2f}".format(TPR), "\tFPR: {:6.2f}".format(FPR))
            break
        else:
            threshold += 0.1

    print("Selected Threshold: {:6.2f}".format(selected_threshold), "TPR: {:6.2f}".format(TPR), "\tFPR: {:6.2f}".format(FPR))
    predict_obj.thd = selected_threshold
    predict_obj.ypred = temp_ypred
    predict_obj.tpr = TPR
    predict_obj.fpr = FPR
    return predict_obj


def get_bfn_mfp(pObj):
    prediction = (pObj.yprob > (pObj.thd / 100)).astype(int)

    if cnst.PERFORM_B2_BOOSTING:
        if pObj.boosting_upper_bound is None:
            fn_indices = np.all([pObj.ytrue.ravel() == cnst.MALWARE, prediction.ravel() == cnst.BENIGN], axis=0)
            pObj.boosting_upper_bound = np.min(pObj.yprob[fn_indices])
            print("Setting B2 boosting threshold:", pObj.boosting_upper_bound)

        # To filter the predicted Benign FN files from prediction results
        brow_indices = np.where(np.all([prediction == cnst.BENIGN, pObj.yprob >= pObj.boosting_upper_bound], axis=0))[0]
        pObj.xB1 = pObj.xtrue[brow_indices]
        pObj.yB1 = pObj.ytrue[brow_indices]
        pObj.yprobB1 = pObj.yprob[brow_indices]
        pObj.ypredB1 = prediction[brow_indices]

        # To filter the benign files that can be boosted directly to B2 set
        boosted_indices = np.where(np.all([prediction == cnst.BENIGN, pObj.yprob < pObj.boosting_upper_bound], axis=0))[0]
        fn_escaped_by_boosting = np.where(np.all([prediction.ravel() == cnst.BENIGN, pObj.yprob.ravel() < pObj.boosting_upper_bound, pObj.ytrue.ravel() == cnst.MALWARE], axis=0))[0]
        pObj.boosted_xB2 = pObj.xtrue[boosted_indices]
        pObj.boosted_yB2 = pObj.ytrue[boosted_indices]
        pObj.boosted_yprobB2 = pObj.yprob[boosted_indices]
        pObj.boosted_ypredB2 = prediction[boosted_indices]
        print("Number of files boosted to B2:", len(np.where(prediction == cnst.BENIGN)[0]), "-", len(brow_indices), "=", len(boosted_indices), "Boosting Bound:", pObj.boosting_upper_bound, "Escaped FNs:", len(fn_escaped_by_boosting))

    else:
        # To filter the predicted Benign FN files from prediction results
        brow_indices = np.where(prediction == cnst.BENIGN)[0]
        pObj.xB1 = pObj.xtrue[brow_indices]
        pObj.yB1 = pObj.ytrue[brow_indices]
        pObj.yprobB1 = pObj.yprob[brow_indices]
        pObj.ypredB1 = prediction[brow_indices]

    # print("\nPREDICT MODULE    Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(brow_indices),
    # len(np.where(pObj.yB1 == cnst.BENIGN)[0]), len(np.where(pObj.yB1 == cnst.MALWARE)[0])))

    mrow_indices = np.where(prediction == cnst.MALWARE)[0]
    pObj.xM1 = pObj.xtrue[mrow_indices]
    pObj.yM1 = pObj.ytrue[mrow_indices]
    pObj.yprobM1 = pObj.yprob[mrow_indices]
    pObj.ypredM1 = prediction[mrow_indices]

    return pObj

def predict_tier1(model_idx, pobj, fold_index):
    predict_args = DefaultPredictArguments()
    tier1_model = load_model(predict_args.model_path + cnst.TIER1_MODELS[model_idx] + "_" + str(fold_index) + ".h5")
    # model.summary()

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        pobj.yprob = predict_byte(tier1_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier1_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier1_model, pobj.xtrue, predict_args)

    if pobj.thd is None:
        pobj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
    else:
        pobj = calculate_prediction_metrics(pobj)

    pobj = get_bfn_mfp(pobj)
    return pobj


def predict_tier2(model_idx, pobj, fold_index):
    predict_args = DefaultPredictArguments()
    tier2_model = load_model(predict_args.model_path + cnst.TIER2_MODELS[model_idx] + "_" + str(fold_index) + ".h5")

    if cnst.EXECUTION_TYPE[model_idx] == cnst.BYTE:
        # pbs.trigger_predict_by_section()
        pobj.yprob = predict_byte_by_section(tier2_model, pobj.xtrue, pobj.q_sections, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FEATURISTIC:
        pobj.yprob = predict_by_features(tier2_model, pobj.xtrue, predict_args)
    elif cnst.EXECUTION_TYPE[model_idx] == cnst.FUSION:
        pobj.yprob = predict_by_fusion(tier2_model, pobj.xtrue, predict_args)

    if pobj.thd is None:
        pObj = select_decision_threshold(pobj)  # +++ returned pobj also includes ypred based on selected threshold
        # pObj = get_bfn_mfp(pObj)              # Enable if TIER-3 is added in future
        return pObj
    else:
        pObj = calculate_prediction_metrics(pobj)
        return pObj


def get_reconciled_tpr_fpr(yt1, yp1, yt2, yp2):
    cm1 = metrics.confusion_matrix(yt1, yp1, labels=[cnst.BENIGN, cnst.MALWARE])
    tn1 = cm1[0][0]
    fp1 = cm1[0][1]
    fn1 = cm1[1][0]
    tp1 = cm1[1][1]

    cm2 = metrics.confusion_matrix(yt2, yp2, labels=[cnst.BENIGN,cnst.MALWARE])
    tn2 = cm2[0][0]
    fp2 = cm2[0][1]
    fn2 = cm2[1][0]
    tp2 = cm2[1][1]

    tpr = ((tp1+tp2) / (tp1+tp2+fn1+fn2)) * 100
    fpr = ((fp1+fp2) / (fp1+fp2+tn1+tn2)) * 100

    return tpr, fpr


def get_tpr_fpr(yt, yp):
    cm = metrics.confusion_matrix(yt, yp, labels=[cnst.BENIGN, cnst.MALWARE])
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tpr = (tp / (tp+fn)) * 100
    fpr = (fp / (fp+tn)) * 100
    return tpr, fpr


def reconcile(pt1, pt2, cv_obj, fold_index):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   RECONCILING DATA  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # RECONCILE - xM1, yprobM1, xB1, pred_proba2
    print("BEFORE RECONCILIATION: [Total True Malwares:", np.sum(pt1.ytrue), "]",
          "TPs found:", np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.MALWARE], axis=0)),
          "B1 has (FNs):", np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.BENIGN], axis=0)))

    if cnst.PERFORM_B2_BOOSTING:
        print(np.shape(pt1.xM1), np.shape(pt1.boosted_xB2), np.shape(pt2.xtrue))
        xtruereconciled = np.concatenate((pt1.xM1, pt1.boosted_xB2, pt2.xtrue))  # pt2.xtrue contains xB1
        ytruereconciled = np.concatenate((pt1.yM1, pt1.boosted_yB2, pt2.ytrue))
        ypredreconciled = np.concatenate((pt1.ypredM1, pt1.boosted_ypredB2, pt2.ypred))
        yprobreconciled = np.concatenate((pt1.yprobM1, pt1.boosted_yprobB2, pt2.yprob))
    else:
        xtruereconciled = np.concatenate((pt1.xM1, pt2.xtrue))  # pt2.xtrue contains xB1
        ytruereconciled = np.concatenate((pt1.yM1, pt2.ytrue))
        ypredreconciled = np.concatenate((pt1.ypredM1, pt2.ypred))
        yprobreconciled = np.concatenate((pt1.yprobM1, pt2.yprob))

    print("AFTER RECONCILIATION :", "[M1+B1] => [", "M1+(M2+B2)", "+B2_Boosted" if cnst.PERFORM_B2_BOOSTING else "", "]", np.shape(xtruereconciled),
          "TPs found: [T2] =>", np.sum(np.all([pt2.ytrue.ravel() == cnst.MALWARE, pt2.ypred.ravel() == cnst.MALWARE], axis=0)),
          "[RECON TPs] =>", np.sum(np.all([pt2.ytrue.ravel() == cnst.MALWARE, pt2.ypred.ravel() == cnst.MALWARE], axis=0)) +
          np.sum(np.all([pt1.ytrue.ravel() == cnst.MALWARE, pt1.ypred.ravel() == cnst.MALWARE], axis=0)))

    reconciled_tpr = np.array([])
    reconciled_fpr = np.array([])
    tier1_tpr = np.array([])
    tier1_fpr = np.array([])
    probability_score = np.arange(0, 100.01, 0.1)
    for p in probability_score:
        rtpr, rfpr = None, None
        if cnst.PERFORM_B2_BOOSTING:
            ytrue_M1B2_Boosted = np.concatenate((pt1.yM1, pt1.boosted_yB2))
            yprob_M1B2_Boosted = np.concatenate((pt1.yprobM1, pt1.boosted_yprobB2))
            rtpr, rfpr = get_reconciled_tpr_fpr(ytrue_M1B2_Boosted, yprob_M1B2_Boosted > (p/100), pt2.ytrue, pt2.yprob > (p/100))
        else:
            rtpr, rfpr = get_reconciled_tpr_fpr(pt1.yM1, pt1.yprobM1 > (p/100), pt2.ytrue, pt2.yprob > (p/100))
        reconciled_tpr = np.append(reconciled_tpr, rtpr)
        reconciled_fpr = np.append(reconciled_fpr, rfpr)

        tpr1, fpr1 = get_tpr_fpr(pt1.ytrue, pt1.yprob > (p/100))
        tier1_tpr = np.append(tier1_tpr, tpr1)
        tier1_fpr = np.append(tier1_fpr, fpr1)

    cv_obj.t1_tpr_folds[fold_index] = tier1_tpr
    cv_obj.t1_fpr_folds[fold_index] = tier1_fpr
    cv_obj.recon_tpr_folds[fold_index] = reconciled_tpr
    cv_obj.recon_fpr_folds[fold_index] = reconciled_fpr

    cv_obj.t1_mean_tpr_auc = tier1_tpr if cv_obj.t1_mean_tpr_auc is None else np.sum([cv_obj.t1_mean_tpr_auc, tier1_tpr], axis=0) / 2
    cv_obj.t1_mean_fpr_auc = tier1_fpr if cv_obj.t1_mean_fpr_auc is None else np.sum([cv_obj.t1_mean_fpr_auc, tier1_fpr], axis=0) / 2
    cv_obj.recon_mean_tpr_auc = reconciled_tpr if cv_obj.recon_mean_tpr_auc is None else np.sum([cv_obj.recon_mean_tpr_auc, reconciled_tpr], axis=0) / 2
    cv_obj.recon_mean_fpr_auc = reconciled_fpr if cv_obj.recon_mean_fpr_auc is None else np.sum([cv_obj.recon_mean_fpr_auc, reconciled_fpr], axis=0) / 2

    tpr1, fpr1 = get_tpr_fpr(pt1.ytrue, pt1.ypred)
    rtpr, rfpr = None, None
    if cnst.PERFORM_B2_BOOSTING:
        ytrue_M1B2_Boosted = np.concatenate((pt1.yM1, pt1.boosted_yB2))
        ypred_M1B2_Boosted = np.concatenate((pt1.ypredM1, pt1.boosted_ypredB2))
        rtpr, rfpr = get_reconciled_tpr_fpr(ytrue_M1B2_Boosted, ypred_M1B2_Boosted, pt2.ytrue, pt2.ypred)
    else:
        rtpr, rfpr = get_reconciled_tpr_fpr(pt1.yM1, pt1.ypredM1, pt2.ytrue, pt2.ypred)
    print("FOLD:", fold_index+1, "TIER1 TPR:", tpr1, "FPR:", fpr1, "OVERALL TPR:", rtpr, "FPR:", rfpr)

    cv_obj.t1_tpr_list = np.append(cv_obj.t1_tpr_list, tpr1)
    cv_obj.t1_fpr_list = np.append(cv_obj.t1_fpr_list, fpr1)
    cv_obj.recon_tpr_list = np.append(cv_obj.recon_tpr_list, rtpr)
    cv_obj.recon_fpr_list = np.append(cv_obj.recon_fpr_list, rfpr)

    cv_obj.t1_mean_auc_score = np.append(cv_obj.t1_mean_auc_score, metrics.roc_auc_score(pt1.ytrue, pt1.yprob))
    cv_obj.t1_mean_auc_score_restricted = np.append(cv_obj.t1_mean_auc_score_restricted, metrics.roc_auc_score(pt1.ytrue, pt1.yprob, max_fpr=0.01))
    cv_obj.recon_mean_auc_score = np.append(cv_obj.recon_mean_auc_score, metrics.roc_auc_score(ytruereconciled, yprobreconciled))
    cv_obj.recon_mean_auc_score_restricted = np.append(cv_obj.recon_mean_auc_score_restricted, metrics.roc_auc_score(ytruereconciled, yprobreconciled, max_fpr=0.01))

    return cv_obj


def init(model_idx, thd1, boosting_upper_bound, thd2, q_sections, testdata, cv_obj, fold_index):
    # TIER-1 PREDICTION OVER TEST DATA
    print("\nPrediction on Testing Data - TIER1")
    predict_t1_test_data = pObj(cnst.TIER1, None, testdata.xdf.values, testdata.ydf.values)
    predict_t1_test_data.thd = thd1
    predict_t1_test_data.boosting_upper_bound = boosting_upper_bound
    predict_t1_test_data = predict_tier1(model_idx, predict_t1_test_data, fold_index)

    # print("TPR:", predict_t1_test_data.tpr, "FPR:", predict_t1_test_data.fpr)
    # plots.plot_auc(ytrain, pred_proba1, thd1, "tier1")

    test_b1datadf = pd.concat([pd.DataFrame(predict_t1_test_data.xB1), pd.DataFrame(predict_t1_test_data.yB1)], axis=1)
    test_b1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)
    test_m1datadf = pd.concat([pd.DataFrame(predict_t1_test_data.xM1), pd.DataFrame(predict_t1_test_data.yM1)], axis=1)
    test_m1datadf.to_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None, index=None)

    test_b1datadf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "b1_test_"+str(fold_index)+"_pkl.csv", header=None)
    predict_t1_test_data.xB1, predict_t1_test_data.yB1 = test_b1datadf.iloc[:, 0], test_b1datadf.iloc[:, 1]
    test_m1datadf = pd.read_csv(cnst.PROJECT_BASE_PATH + cnst.ESC + "data" + cnst.ESC + "m1_test_"+str(fold_index)+"_pkl.csv", header=None)
    predict_t1_test_data.xM1, predict_t1_test_data.yM1 = test_m1datadf.iloc[:, 0], test_m1datadf.iloc[:, 1]

    # TIER-2 PREDICTION
    print("Prediction on Testing Data - TIER2 [B1 data]")
    predict_t2_test_data = pObj(cnst.TIER2, None, predict_t1_test_data.xB1, predict_t1_test_data.yB1)
    predict_t2_test_data.thd = thd2
    predict_t2_test_data.q_sections = q_sections
    predict_t2_test_data = predict_tier2(model_idx, predict_t2_test_data, fold_index)
    # print("TPR:", predict_t2_test_data.tpr, "FPR:", predict_t2_test_data.fpr)

    # RECONCILIATION OF PREDICTION RESULTS FROM TIER - 1&2
    return reconcile(predict_t1_test_data, predict_t2_test_data, cv_obj, fold_index)


if __name__ == '__main__':
    print("PREDICT MAIN")
    '''print("Prediction on Testing Data")
    #import Predict as pObj
    testdata = pd.read_csv(cnst.PROJECT_BASE_PATH + 'small_pkl_1_1.csv', header=None)
    pObj_testdata = Predict(cnst.TIER1, cnst.TIER1_TARGET_FPR, testdata.iloc[:, 0].values, testdata.iloc[:, 1].values)
    pObj_testdata.thd = 67.1
    pObj_testdata = predict_tier1(0, pObj_testdata)  # TIER-1 prediction - on training data
    print("TPR:", pObj_testdata.tpr, "FPR:", pObj_testdata.fpr)'''


'''
def get_prediction_data(result_file):
    t1 = pd.read_csv(result_file, header=None)
    y1 = t1[1]
    p1 = t1[2]
    pv1 = t1[3]
    return y1, p1, pv1
'''