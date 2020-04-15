import os
from utils import utils
import numpy as np
from keras.models import Model
from keras.models import load_model
import pickle
from os.path import join
import config.constants as cnst
import plots.plots as plots
from plots import ati_plots
from predict.predict import predict_byte
from predict.predict_args import DefaultPredictArguments, Predict as pObj
from plots.ati_plots import feature_map_histogram
from .ati_args import SectionActivationDistribution
import pandas as pd


def find_qualified_sections(sd, trend, common_trend, support):
    btrend = trend.loc["BENIGN_ACTIVATION_MAGNITUDE"]
    mtrend = trend.loc["MALWARE_ACTIVATION_MAGNITUDE"]

    # Averaging based on respective benign and malware population
    btrend = btrend / sd.b1_b_truth_count
    mtrend = mtrend / sd.b1_m_truth_count

    btrend[btrend == 0] = 1
    mtrend[mtrend == 0] = 1

    activation_magnitude_gaps = mtrend / btrend
    q_criteria_by_percentiles = np.percentile(activation_magnitude_gaps, q=cnst.PERCENTILES)
    # q_criteria = [0]
    # q_criteria = np.append([0], q_criteria)
    # print("Q_criteria by Deciles:", q_criteria_by_percentiles, "\n")
    q_sections_by_q_criteria = {}
    for q_criterion in q_criteria_by_percentiles:
        q_sections_by_q_criteria[q_criterion] = trend.columns[activation_magnitude_gaps > q_criterion]
    return q_sections_by_q_criteria


def parse_pe_pkl(file_id, file, unprocessed):
    section_bounds = []
    file_byte_size = None
    max_section_end_offset = 0
    try:
        with open(file, 'rb') as pkl:
            fjson = pickle.load(pkl)
            file_byte_size = fjson['size_byte']
            pkl_sections = fjson["section_info"].keys()
            for pkl_section in pkl_sections:
                section_bounds.append(
                    (pkl_section,
                     fjson["section_info"][pkl_section]["section_bounds"]["start_offset"],
                     fjson["section_info"][pkl_section]["section_bounds"]["end_offset"]))
                if fjson["section_info"][pkl_section]["section_bounds"]["end_offset"] > max_section_end_offset:
                    max_section_end_offset = fjson["section_info"][pkl_section]["section_bounds"]["end_offset"]

            # Placeholder section "padding" - for activations in padding region
            if max_section_end_offset < fjson["size_byte"]:
                section_bounds.append((cnst.TAIL, max_section_end_offset + 1, fjson["size_byte"]))
            section_bounds.append((cnst.PADDING, fjson["size_byte"] + 1, cnst.MAX_FILE_SIZE_LIMIT))
    except Exception as e:
        print("parse failed . . . [FILE ID - ", file_id, "]  [", file, "] ", e)
        unprocessed += 1
    return section_bounds, unprocessed, file_byte_size


def get_feature_map(smodel, file):
    predict_args = DefaultPredictArguments()
    predict_args.verbose = cnst.ATI_PREDICT_VERBOSE
    prediction = predict_byte(smodel, np.array([file]), predict_args)
    raw_feature_map = prediction[0]
    return raw_feature_map


def map_act_to_sec(ftype, fmap, sbounds, sd):
    # section_support:      Information about how many samples in a given category has a section <Influence by presence>
    # activation_histogram: Information about total count of activations occurred in a given section for all samples
    #                       of given category <Influence by activation count>
    # activation_magnitude: Information about total sum of magnitude of activations occurred in a given section
    #                       for all samples of given category <Influence by activation strength>

    # fmap = fmap // 1  # print("FEATURE MAP ", len(feature_map), " : \n", feature_map)
    idx = np.argsort(fmap)[::-1][:len(fmap)]  # Sort activations in descending order -- Helpful to find top activations
    if sbounds is not None:

        for j in range(0, len(sbounds)):
            section = sbounds[j][0]
            sd.a_section_support[section] = (
                        sd.a_section_support[section] + 1) if section in sd.a_section_support.keys() else 1
            if ftype == cnst.BENIGN:
                sd.b_section_support[section] = (
                            sd.b_section_support[section] + 1) if section in sd.b_section_support.keys() else 1
                if section not in sd.m_section_support.keys():
                    sd.m_section_support[section] = 0
            else:
                if section not in sd.b_section_support.keys():
                    sd.b_section_support[section] = 0
                sd.m_section_support[section] = (
                            sd.m_section_support[section] + 1) if section in sd.m_section_support.keys() else 1

        for current_activation_window in range(0, len(fmap)):  # range(0, int(cnst.MAX_FILE_SIZE_LIMIT / cnst.CONV_STRIDE_SIZE)):
            section = None
            offset = idx[current_activation_window] * 500
            act_val = fmap[idx[current_activation_window]]

            ######################################################################################
            # Change for Pooling layer based Activation trend - Only Max activation is traced back
            if act_val == 0:
                continue
            ######################################################################################
            for j in range(0, len(sbounds)):
                cur_section = sbounds[j]
                if cur_section[1] <= offset <= cur_section[2]:
                    section = cur_section[0]
                    break

            if section is not None:
                # if "." not in section: section = "." + section #Same section's name with and without dot are different
                # Sum of Magnitude of Activations
                if section in sd.a_activation_magnitude.keys():
                    sd.a_activation_magnitude[section] += act_val
                    sd.a_activation_histogram[section] += 1
                    if ftype == cnst.BENIGN:
                        if sd.b_activation_magnitude[section] is None:
                            sd.b_activation_magnitude[section] = act_val
                            sd.b_activation_histogram[section] = 1
                        else:
                            sd.b_activation_magnitude[section] += act_val
                            sd.b_activation_histogram[section] += 1
                    else:
                        if sd.m_activation_magnitude[section] is None:
                            sd.m_activation_magnitude[section] = act_val
                            sd.m_activation_histogram[section] = 1
                        else:
                            sd.m_activation_magnitude[section] += act_val
                            sd.m_activation_histogram[section] += 1
                else:
                    sd.a_activation_magnitude[section] = act_val
                    sd.a_activation_histogram[section] = 1
                    if ftype == cnst.BENIGN:
                        sd.b_activation_magnitude[section] = act_val
                        sd.b_activation_histogram[section] = 1
                        sd.m_activation_magnitude[section] = None
                        sd.m_activation_histogram[section] = None
                    else:
                        sd.b_activation_magnitude[section] = None
                        sd.b_activation_histogram[section] = None
                        sd.m_activation_magnitude[section] = act_val
                        sd.m_activation_histogram[section] = 1
            else:
                # !!! VERIFY ALL OFFSET IS MATCHED AND CHECK FOR LEAKAGE !!!
                # print("No matching section found for OFFSET:", offset)
                sd.a_activation_magnitude[cnst.LEAK] += act_val
                sd.a_activation_histogram[cnst.LEAK] += 1
                if ftype == cnst.BENIGN:
                    sd.b_activation_magnitude[cnst.LEAK] += act_val
                    sd.b_activation_histogram[cnst.LEAK] += 1
                else:
                    sd.m_activation_magnitude[cnst.LEAK] += act_val
                    sd.m_activation_histogram[cnst.LEAK] += 1
    return sd


def process_files(args):
    unprocessed = 0
    max_activation_value = 0
    samplewise_feature_maps = []
    stunted_model = get_stunted_model(args)

    print("FMAP MODULE Total B1 [{0}]\tGroundTruth [{1}:{2}]".format(len(args.t2_y_train),
                                                                     len(np.where(args.t2_y_train == cnst.BENIGN)[0]),
                                                                     len(np.where(args.t2_y_train == cnst.MALWARE)[0])))
    sd = SectionActivationDistribution()
    sd.b1_count = len(args.t2_y_train)
    sd.b1_b_truth_count = len(np.where(args.t2_y_train == cnst.BENIGN)[0])
    sd.b1_m_truth_count = len(np.where(args.t2_y_train == cnst.MALWARE)[0])

    pObj_fmap = pObj(cnst.TIER1, None, args.t2_x_train, args.t2_y_train)
    for i in range(0, len(pObj_fmap.xtrue)):
        # print("File # ", i, "Max Activation Value:", max_activation_value)
        file = pObj_fmap.xtrue[i]
        file_type = pObj_fmap.ytrue[i]  # Using Ground Truth to get trend of actual benign and malware files
        if not os.path.exists(cnst.DATA_SOURCE_PATH + file):
            print(file, " does not exist. Skipping . . .")
            unprocessed += 1
            continue
        section_bounds, unprocessed, fsize = parse_pe_pkl(i, cnst.DATA_SOURCE_PATH + file, unprocessed)
        raw_feature_map = get_feature_map(stunted_model, file)
        # if len(np.shape(raw_feature_map)) == 1:
        #    feature_map = raw_feature_map.ravel()
        # else:

        if cnst.USE_POOLING_LAYER:
            try:
                pooled_max_2D_map = np.amax(raw_feature_map, axis=0)
                pooled_max_1D_map = np.sum(raw_feature_map == np.amax(raw_feature_map, axis=0), axis=1)[:np.min([cnst.MAX_FILE_CONVOLUTED_SIZE,int(fsize/cnst.CONV_STRIDE_SIZE)+2])]
                sd = map_act_to_sec(file_type, pooled_max_1D_map, section_bounds, sd)
            except Exception as e:
                print("$$$$$$$$", str(e))  # max_activation_value, raw_feature_map.ravel().size)
        else:
            feature_map = raw_feature_map.sum(axis=1).ravel()
            # feature_map_histogram(feature_map, prediction)
            try:
                if max_activation_value < feature_map.max():
                    max_activation_value = feature_map.max()
                    # print("max_activation_value: ", max_activation_value)
            except Exception as e:
                print("$$$$$$$$")  # max_activation_value, raw_feature_map.ravel().max(), raw_feature_map.ravel().size)
            samplewise_feature_maps.append(feature_map)
            sd = map_act_to_sec(file_type, feature_map, section_bounds, sd)

    return sd, max_activation_value

    # print(section_stat)
    # print("Unprocessed file count: ", unprocessed)

    # Find activation distribution
    # raw_arr = np.array(np.squeeze(temp_feature_map_list))
    # print(len(raw_arr), raw_arr.max())
    # raw_arr = raw_arr[raw_arr > 0.3]
    # print(len(raw_arr))
    # plt.hist(raw_arr, 10)#range(0, len(raw_arr)))
    # plt.show()

    '''for key in act.keys():
        # key = "."+key if "." not in key else key
        if key is not None and key != '' and key != '.padding':
            with open("BENIGN" if "benign" in section_stat_file else "MALWARE" + "_activation_" + key[1:] + ".csv", mode='a+') as f:
                f.write(str(act[key]))
    '''
    '''
    #overall_stat.append(section_stat)
    for x in pcs_keys:
        overall_stat_str += str(section_stat[x]) + ","
    overall_stat_str = overall_stat_str[:-1] + "\n"

    print("\n[Unprocessed Files : ", unprocessed, "]      Overall Stats: ", overall_stat_str)

    processed_file_count = len(fn_list) - unprocessed
    normalized_stats_str = str(section_stat["header"]/processed_file_count) + "," \
                           + str(section_stat["text"]/processed_file_count) + "," \
                           + str(section_stat["data"]/processed_file_count) + "," \
                           + str(section_stat["rsrc"]/processed_file_count) + "," \
                           + str(section_stat["pdata"]/processed_file_count) + "," \
                           + str(section_stat["rdata"]/processed_file_count) + "\n"
                           #+ str(section_stat["padding"]/processed_file_count) \

    print("Normalized Stats: ", normalized_stats_str)
    #plt.show()

    with open(section_stat_file, 'w+') as f:
        f.write(overall_stat_str)
        f.write("\n")
        f.write(normalized_stats_str)
    '''


def get_stunted_model(args):
    # limit gpu memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    complete_model = load_model(join(args.save_path, args.t1_model_name))
    # model.summary()
    # redefine model to output right after the sixth hidden layer
    # (ReLU activation layer after convolution - before max pooling)
    stunted_outputs = [complete_model.layers[x].output for x in [cnst.LAYER_NUM_TO_STUNT]]
    stunted_model = Model(inputs=complete_model.inputs, outputs=stunted_outputs)
    # stunted_model.summary()
    print("Model stunted upto ", stunted_outputs[0])
    return stunted_model


def save_activation_trend(sd):
    fmaps_trend = pd.DataFrame()
    fmaps_common_trend = pd.DataFrame()
    fmaps_section_support = pd.DataFrame()

    fmaps_trend["ACTIVATION / HISTOGRAM"] = ["ALL_ACTIVATION_MAGNITUDE", "BENIGN_ACTIVATION_MAGNITUDE",
                                             "MALWARE_ACTIVATION_MAGNITUDE", "HISTOGRAM_ALL", "HISTOGRAM_BENIGN",
                                             "HISTOGRAM_MALWARE"]
    fmaps_common_trend["COMMON"] = ["ALL_ACTIVATION_MAGNITUDE", "BENIGN_ACTIVATION_MAGNITUDE",
                                    "MALWARE_ACTIVATION_MAGNITUDE", "HISTOGRAM_ALL", "HISTOGRAM_BENIGN",
                                    "HISTOGRAM_MALWARE"]
    fmaps_section_support["SUPPORT"] = ["PRESENCE_IN_ALL", "PRESENCE_IN_BENIGN", "PRESENCE_IN_MALWARE",
                                        "SUPPORT_IN_ALL", "SUPPORT_IN_BENIGN", "SUPPORT_IN_MALWARE"]

    for key in sd.a_activation_histogram.keys():
        fmaps_trend[key] = [int(sd.a_activation_magnitude[key]) if sd.a_activation_magnitude[key] is not None else
                            sd.a_activation_magnitude[key],
                            int(sd.b_activation_magnitude[key]) if sd.b_activation_magnitude[key] is not None else
                            sd.b_activation_magnitude[key],
                            int(sd.m_activation_magnitude[key]) if sd.m_activation_magnitude[key] is not None else
                            sd.m_activation_magnitude[key],
                            int(sd.a_activation_histogram[key]) if sd.a_activation_histogram[key] is not None else
                            sd.a_activation_histogram[key],
                            int(sd.b_activation_histogram[key]) if sd.b_activation_histogram[key] is not None else
                            sd.b_activation_histogram[key],
                            int(sd.m_activation_histogram[key]) if sd.m_activation_histogram[key] is not None else
                            sd.m_activation_histogram[key]]

        if sd.b_activation_histogram[key] is not None and sd.m_activation_histogram[key] is not None:
            fmaps_common_trend[key] = [
                int(sd.a_activation_magnitude[key]) if sd.a_activation_magnitude[key] is not None else
                sd.a_activation_magnitude[key],
                int(sd.b_activation_magnitude[key]) if sd.b_activation_magnitude[key] is not None else
                sd.b_activation_magnitude[key],
                int(sd.m_activation_magnitude[key]) if sd.m_activation_magnitude[key] is not None else
                sd.m_activation_magnitude[key],
                int(sd.a_activation_histogram[key]) if sd.a_activation_histogram[key] is not None else
                sd.a_activation_histogram[key],
                int(sd.b_activation_histogram[key]) if sd.b_activation_histogram[key] is not None else
                sd.b_activation_histogram[key],
                int(sd.m_activation_histogram[key]) if sd.m_activation_histogram[key] is not None else
                sd.m_activation_histogram[key]]

    if sd.b1_count > 0 and sd.b1_b_truth_count > 0 and sd.b1_m_truth_count > 0:
        for key in sd.a_section_support.keys():
            fmaps_section_support[key] = [sd.a_section_support[key], sd.b_section_support[key],
                                          sd.m_section_support[key],
                                          "{:0.1f}%".format(sd.a_section_support[key] / sd.b1_count * 100),
                                          "{:0.1f}%".format(sd.b_section_support[key] / sd.b1_b_truth_count * 100),
                                          "{:0.1f}%".format(sd.m_section_support[key] / sd.b1_m_truth_count * 100)]

    fmaps_trend.fillna(-1, inplace=True)

    fmaps_trend.set_index('ACTIVATION / HISTOGRAM', inplace=True)
    fmaps_common_trend.set_index('COMMON', inplace=True)
    fmaps_section_support.set_index('SUPPORT', inplace=True)

    # Store activation trend identified
    fmaps_trend.to_csv(cnst.COMBINED_FEATURE_MAP_STATS_FILE, index=True)
    fmaps_common_trend.to_csv(cnst.COMMON_COMBINED_FEATURE_MAP_STATS_FILE, index=True)
    fmaps_section_support.to_csv(cnst.SECTION_SUPPORT, index=True)

    # Drop padding and leak information after saving - not useful for further processing
    fmaps_trend.drop([cnst.PADDING, cnst.LEAK], axis=1, inplace=True)
    fmaps_common_trend.drop([cnst.PADDING, cnst.LEAK], axis=1, inplace=True)
    fmaps_section_support.drop([cnst.PADDING], axis=1, inplace=True)
    return fmaps_trend, fmaps_common_trend, fmaps_section_support


def start_ati_process(args):
    print("\nATI - PROCESSING BENIGN AND MALWARE FILES\t\t", "B1 FILES COUNT:", np.shape(args.t2_y_train)[0])
    print("-----------------------------------------")
    sd, max_activation_value = process_files(args)
    # print("Final max_activation_value", max_activation_value)
    fmaps_trend, fmaps_common_trend, fmaps_section_support = save_activation_trend(sd)
    return sd, fmaps_trend, fmaps_common_trend, fmaps_section_support


def init(args):
    sd, trend, common_trend, support = start_ati_process(args)

    # select sections for Tier-2 based on identified activation trend
    q_sections_by_q_criteria = find_qualified_sections(sd, trend, common_trend, support)

    # select, drop = plots.save_stats_as_plot(fmaps, qualification_criteria)
    return q_sections_by_q_criteria  # select, drop


if __name__ == '__main__':
    # start_visualization_process(args)
    plots.save_stats_as_plot()

    # pe = pefile.PE("D:\\08_Dataset\\benign\\git-gui.exe")
    # parse_pe(0, "D:\\08_Dataset\\benign\\git-gui.exe", 204800, 0)
    # for section in pe.sections:
    #    print(section)
    # print(pe.OPTIONAL_HEADER, "\n", pe.NT_HEADERS, "\n", pe.FILE_HEADER, "\n", pe.RICH_HEADER, "\n", pe.DOS_HEADER,
    # \"\n", pe.__IMAGE_DOS_HEADER_format__, "\n", pe.header, "\n", "LENGTH", len(pe.header))

'''def display_edit_distance():
    sections = []
    top_sections = []
    malware_edit_distance = []
    print("\n   SECTION [EDIT DISTANCE SCORE]")
    df = pd.read_csv(combined_stat_file)
    df.set_index("type", inplace=True)
    for i in range(0, len(keys)):
        a = df.loc['FN'].values[i]
        b = df.loc['BENIGN'].values[i]
        c = df.loc['MALWARE'].values[i]
        dist1 = norm(a-b) // 1
        dist2 = norm(a-c) // 1
        print(keys[i], dist1, dist2, "[MALWARE]" if dist2 < dist1 else "[BENIGN]", dist1 - dist2)
        if dist2 < dist1:
            malware_edit_distance.append(dist1 - dist2)
            sections.append(keys[i])
    idx = np.argsort(malware_edit_distance)[::-1]
    for t in idx:
        print("%10s" % sections[t], "%20s" % str(malware_edit_distance[t]))
        top_sections.append(sections[t])
    return top_sections[:3]

    def ks(cutoff):
    from scipy import stats
    keys = ['header', 'text', 'data', 'rsrc', 'pdata', 'rdata']
    for key in keys:
        b = pd.read_csv('D:\\03_GitWorks\\echelon\\out\\result_multi\\benign.csv' + ".activation_" + key + ".csv", header=None)
        m = pd.read_csv('D:\\03_GitWorks\\echelon\\out\\result_multi\\malware.csv' + ".activation_" + key + ".csv", header=None)
        b = np.squeeze((b.get_values()))
        m = np.squeeze((m.get_values()))
        b = (b - b.min()) / (b.max() - b.min())
        m = (m - m.min()) / (m.max() - m.min())
        print(key, b.max(), len(b), len(b[b > cutoff]))
        print(key, m.max(), len(m), len(m[m > cutoff]))
        print("Section: ", key[:4], "\t\t", stats.ks_2samp(np.array(b), np.array(m)))
        plt.hist(b[b > cutoff], 100)
        plt.hist(m[m > cutoff], 100)
        plt.legend(['benign', 'malware'])
        plt.show()
        # break
    '''
