import os, shutil
import pefile
import pandas as pd
from shutil import copyfile
import config.constants as cnst


def group_files_by_pkl_list():
    csv = pd.read_csv("D:\\03_GitWorks\\April15\\data\\ds1_20.csv", header=None)
    dst_folder = "D:\\00_SFU\\ds1_20_raw\\"
    for file in csv.iloc[:, 0]:
        src_path = os.path.join(cnst.DATA_SOURCE_PATH, file)
        dst_Path = os.path.join(dst_folder, file)
        copyfile(src_path[:-4], dst_Path[:-4])


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

    group_files_by_pkl_list()
    print("\nCompleted.")

