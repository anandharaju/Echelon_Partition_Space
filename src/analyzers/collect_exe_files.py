import os, shutil
import pefile
import pandas as pd
from shutil import copyfile
import config.constants as cnst
import pickle


def partition_pkl_files(type, fold, files):
    partition_label = type + "_" + str(fold)
    # csv = pd.read_csv(csv_path, header=None)
    print("Total number of files:", len(files))
    partition_count = 1
    file_count = 1
    partition_data = {}
    cur_partition_size = 0

    for file in files:  # iloc[:, 0]:
        src_path = os.path.join("D:\\08_Dataset\\Internal\\mar2020\\pickle_files\\", file)
        src_file_size = os.stat(src_path).st_size

        if cur_partition_size > cnst.MAX_PARTITION_SIZE:
            partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"_p"+str(partition_count)+".pkl")
            with open(partition_path, "wb") as phandle:
                pickle.dump(partition_data, phandle)
            print("Created Partition", partition_label+"_p"+str(partition_count), "with", file_count-1, "files")
            file_count = 1
            partition_count += 1
            partition_data = {}
            cur_partition_size = 0

        with open(src_path, 'rb') as f:
            cur_pkl = pickle.load(f)
            partition_data[file[:-4]] = cur_pkl
            cur_partition_size += src_file_size
            file_count += 1

    if cur_partition_size > 0:
        partition_path = os.path.join(cnst.DATA_SOURCE_PATH, partition_label+"_p"+str(partition_count) + ".pkl")
        with open(partition_path, "wb") as phandle:
            pickle.dump(partition_data, phandle)
        print("Created Partition", partition_path, "with", file_count-1, "files")
        partition_count += 1
        partition_data = {}
        cur_partition_size = 0


def get_partition_data(type, fold):
    partition_label = type + "_" + str(fold)
    print("Loading partitioned data for Fold-"+str(fold+1)+". . .", partition_label)
    partition_count = 1
    partition_path = os.path.join(cnst.DATA_SOURCE_PATH + partition_label + "_p" + str(partition_count) + ".pkl")

    if not os.path.isfile(partition_path):
        print("Partition file ", partition_path, ' does not exist.')
    else:
        with open(partition_path, "rb") as pkl_handle:
            return pickle.load(pkl_handle)


def group_files_by_pkl_list():
    csv = pd.read_csv("D:\\03_GitWorks\\Project\\data\\xs_pkl.csv", header=None)
    dst_folder = "D:\\03_GitWorks\\Project\\data\\xs_pkl\\"
    for file in csv.iloc[:, 0]:
        src_path = os.path.join("D:\\08_Dataset\\Internal\\mar2020\\pickle_files\\", file)
        dst_Path = os.path.join(dst_folder, file)
        copyfile(src_path, dst_Path)
        # copyfile(src_path[:-4], dst_Path[:-4])


def copy_files(src_path, dst_path, ext, max_size):
    total_count = 0
    total_size = 0
    unprocessed = 0
    dst_dir = dst_path
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for src_dir, dirs, files in os.walk(src_path):
        for file_ in files:
            if total_count >= max_files: break
            try:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)

                #For Benign
                #if not fnmatch.fnmatch(src_file, ext):
                #    continue

                #For Malware
                #if fnmatch.fnmatch(src_file, ext):
                    #continue

                src_file_size = os.stat(src_file).st_size
                if src_file_size > max_size:
                    continue

                try:
                    # check if file can be processed by pefile module
                    pe = pefile.PE(src_file)
                    if pe._PE__warnings is not None and len(pe._PE__warnings) > 0 \
                            and pe._PE__warnings[0] == 'Invalid section 0. Contents are null-bytes.':
                        raise Exception(pe._PE__warnings[0]+" "+pe._PE__warnings[1])
                    for item in pe.sections:
                        # Check if all sections are parse-able without error
                        _ = item.Name.rstrip(b'\x00').decode("utf-8").strip()
                        _ = item.get_data()
                except Exception as e:
                    unprocessed += 1
                    print("parse failed . . . [ Unprocessed Count: ", str(unprocessed), "] [ Error: " + str(e)
                          + " ] [ FILE ID - ", src_file, "] ")
                    continue

                shutil.copy(src_file, dst_dir)
                print(total_count, "      ", src_file, dst_file)
            except Exception as e1:
                print("Copy failed ", src_file)
            total_count += 1
            total_size += src_file_size
    return total_count, total_size


if __name__ == '__main__':
    src_path = "D:\\08_Dataset\\VirusTotal\\repo\\all"
    dst_path = "D:\\08_Dataset\\aug24_malware\\"

    ext = '*.exe'
    max_size = 512000  # bytes 500KB
    max_files = 110000
    # total_count, total_size = copy_files(src_path, dst_path, ext, max_size)

    # group_files_by_pkl_list()
    for fold in range(0, 5):
        partition_pkl_files("master_train_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_train_"+str(fold)+"_pkl.csv")
        partition_pkl_files("master_val_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_val_"+str(fold)+"_pkl.csv")
        partition_pkl_files("master_test_"+str(fold), "D:\\03_GitWorks\\Project\\data\\master_test_"+str(fold)+"_pkl.csv")
    print("\nCompleted.")

