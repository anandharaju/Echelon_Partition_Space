import os
import time
import pickle
import numpy as np
import argparse
import pandas as pd
from config import constants as cnst
from analyzers.parse_pe import parse_pe_section_data
from keras.preprocessing.sequence import pad_sequences


parser = argparse.ArgumentParser(description='ECHELON')
parser.add_argument('--max_len', type=int, default=512000)
parser.add_argument('--save_path', type=str, default='D:\\03_GitWorks\\echelon\\out\\preprocess\\preprocessed_data.pkl')
parser.add_argument('csv', type=str)


def preprocess(file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    for fn in file_list:
        if not os.path.isfile(fn):
            print(fn, 'not exists')
        else:
            with open(fn, 'rb') as f:
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


def preprocess_by_section(file_list, max_len, sections):
    '''
    Return processed data (ndarray) and original section length (list)
    '''
    corpus = []
    for fn in file_list:
        if not os.path.isfile(fn):
            print(fn, 'not exist')
        else:
            with open(fn, 'rb') as f:
                # For reading image representation of section-wise byte data from pickle file
                fjson = pickle.load(f)
                keys = fjson["section_info"].keys()
                combined = []
                for section in sections:
                    if section in keys:
                        combined = np.concatenate([combined, fjson["section_info"][section]["section_data"]])
                        combined = np.concatenate([combined, np.zeros(cnst.CONV_WINDOW_SIZE)])
                corpus.append(combined)
            # corpus.append(parse_pe_section_data(fn, section))

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
    get_offline_features()
    print("Time taken:", time.time() - st)
