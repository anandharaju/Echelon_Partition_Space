import time
import os
import glob
from config import constants as cnst
from config.echelon_meta import EchelonMeta
import core.generate_train_predict as gtp


def clean_files():
    project_path = cnst.PROJECT_BASE_PATH
    for model_file in glob.glob(project_path + cnst.ESC + "model" + cnst.ESC + "echelon_byte*"):
        os.remove(model_file)
    for img_file in glob.glob(project_path + + cnst.ESC + "out" + cnst.ESC + "imgs" + cnst.ESC + "*.png"):
        os.remove(img_file)


def main():
    metaObj = EchelonMeta()
    # metaObj.project_details()
    # utils.initiate_tensorboard_logging(cnst.TENSORBOARD_LOG_PATH)              # -- TENSOR BOARD LOGGING
    tst = time.time()
    ###################################################################################################################
    # ********************            SET PARAMETERS FOR TRAINING & TESTING             *******************************
    ###################################################################################################################

    # clean_files()

    model = 0  # model index
    metaObj.run_setup()
    gtp.train_predict(model, cnst.ALL_FILE)
    metaObj.run_setup()
    exit()

    '''cust_data = ['small_pkl_6_1.csv', 'small_pkl_7_1.csv', 'small_pkl_8_1.csv', 'small_pkl_9_1.csv', 'small_pkl_10_1.csv']
    for file in cust_data:
        count = 3
        path = cnst.PROJECT_BASE_PATH + file
        while count > 0:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", file)
            gtp.train_predict(MODEL, path)
            count -= 1'''
    ###################################################################################################################


if __name__ == '__main__':
    main()
