import os
import time
import pickle
import numpy as np
import argparse
import pandas as pd
from config import constants as cnst
from analyzers.parse_pe import parse_pe_section_data
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict


parser = argparse.ArgumentParser(description='ECHELON')
parser.add_argument('--max_len', type=int, default=512000)
parser.add_argument('--save_path', type=str, default='preprocessed_data.pkl')
parser.add_argument('csv', type=str)


def preprocess(file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    for fn in file_list:
        fpath = cnst.DATA_SOURCE_PATH + fn
        if not os.path.isfile(fpath):
            print(fpath, 'not exists')
        else:
            with open(fpath, 'rb') as f:
                # For reading image representation of byte data from pickle file
                fjson = pickle.load(f)
                data = fjson["whole_bytes"]  # image_256w
                corpus.append(data)

                # For reading a executable file directly
                # corpus.append(f.read())

    # corpus = [[byte for byte in doc] for doc in corpus]
    len_list = None  # [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, truncating='post', padding='post')  # , value=b'\x00')
    return seq, len_list


def preprocess_by_section(file_list, max_len, sections, section_map):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    if sections is None:
        print("No sections supplied to process. Check if Q-criterion based selection completed successfully.")

    corpus = []
    section_byte_map = OrderedDict.fromkeys(section_map, value=0)
    for fn in file_list:
        fpath = cnst.DATA_SOURCE_PATH + fn
        if not os.path.isfile(fpath):
            print(fpath, 'not exists')
        else:
            with open(fpath, 'rb') as f:
                # For reading image representation of section-wise byte data from pickle file
                fjson = pickle.load(f)
                keys = fjson["section_info"].keys()

                try:
                    # Update byte map with sections that are present in current file
                    for key in keys:
                        if key in section_map:
                            section_byte_map[key] = 1
                        #else:
                        #    print("Unknown Section in B1 samples:", key)
                            
                    byte_map = []
                    byte_map_input = section_byte_map.values()  # ordered dict preserves the order of byte map
                    for x in byte_map_input:
                        byte_map.append(x)
                        
                    combined = []
                    combined = np.concatenate([combined, byte_map, np.zeros(cnst.CONV_WINDOW_SIZE)])

                    for section in sections:
                        if section in keys:
                            combined = np.concatenate([combined, fjson["section_info"][section]["section_data"], np.zeros(cnst.CONV_WINDOW_SIZE)])

                    if cnst.TAIL in sections:
                        fsize = len(fjson["whole_bytes"])
                        sections_end = 0
                        for key in keys:
                            if fjson["section_info"][key]['section_bounds']["end_offset"] > sections_end:
                                sections_end = fjson["section_info"][key]['section_bounds']["end_offset"]
                        if sections_end < fsize - 1:
                            combined = np.concatenate([combined, fjson["whole_bytes"][sections_end:fsize], np.zeros(cnst.CONV_WINDOW_SIZE)])

                    corpus.append(combined)
                    if len(combined) > max_len:
                        print("[CAUTION: LOSS_OF_DATA] Section Byte Map + Sections : exceeded max sample length by "+str(len(combined)-max_len)+" bytes")

                except Exception as e:
                    print("Module: process_by_section. Error:", str(e))

    corpus = [[byte for byte in doc] for doc in corpus]
    len_list = [len(doc) for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
    return seq, len_list


if __name__ == '__main__':
    '''args = parser.parse_args()

    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    
    print('Preprocessing ...... this may take a while ...')
    st = time.time()
    processed_data = preprocess_by_section(fn_list, args.max_len, '.data')[0]
    print('Finished ...... %d sec' % int(time.time()-st))
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print('Preprocessed data store in', args.save_path)'''
    st = time.time()
    #get_offline_features()
    print("Time taken:", time.time() - st)
