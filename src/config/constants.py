# 42 :Answer to the Ultimate Question of Life, the Universe, and Everything
# ~ The Hitchhiker's Guide to the Galaxy
RANDOM_SEED = 42
VERBOSE_0 = 0
VERBOSE_1 = 1
TIER1_PRETRAINED_MODEL = "ember_malconv.h5"
TIER2_PRETRAINED_MODEL = "ember_malconv.h5"
BENIGN = 0
MALWARE = 1
T1_TRAIN_BATCH_SIZE = 100
T2_TRAIN_BATCH_SIZE = 100
PREDICT_BATCH_SIZE = 200

EPOCHS = 1
PREDICT_VERBOSE = VERBOSE_1
RESUME = True
PERFORM_B2_BOOSTING = False

LINUX_ENV = False
ESC = "/" if LINUX_ENV else "\\"

# TIER-1
TIER1 = "TIER1"
TIER1_EPOCHS = EPOCHS
TIER1_TARGET_FPR = 0.5
SKIP_TIER1_TRAINING = True
SKIP_ATI_PROCESSING = True
USE_PRETRAINED_FOR_TIER1 = False
T1_VERBOSE = VERBOSE_1

# TIER-2
TIER2 = "TIER2"
TIER2_EPOCHS = EPOCHS
TIER2_TARGET_FPR = 0
USE_PRETRAINED_FOR_TIER2 = False
T2_VERBOSE = VERBOSE_1

# CROSS VALIDATION
CV_FOLDS = 5
INITIAL_FOLD = 0
RANDOMIZE = False  # Set Random seed for True
TEST_SET_SIZE = 0.0001

#  DATA SOURCE
MAX_FILE_SIZE_LIMIT = 2**20  # 204800
MAX_FILE_COUNT_LIMIT = None
CONV_WINDOW_SIZE = 500
CONV_STRIDE_SIZE = 500
PROJECT_BASE_PATH = '/home/vivo/Echelon_Linux' if LINUX_ENV else 'D:\\03_GitWorks\\Huawei_Echelon_Reboot'
DATA_SOURCE_PATH = '/home/vivo/pickle_files/' if LINUX_ENV else 'D:\\08_Dataset\\Internal\\mar2020\\pickle_files\\'
ALL_FILE = PROJECT_BASE_PATH  + ESC + 'data' + ESC + 'balanced_pkl.csv'  # small_pkl_1_1.csv'
BENIGN_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_benign_pkl.csv'
MALWARE_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'medium_malware_pkl.csv'
TRAINING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'training.csv'
TESTING_FILE = PROJECT_BASE_PATH + ESC + 'data' + ESC + 'testing.csv'
GENERATE_BENIGN_MALWARE_FILES = False
CHECK_FILE_SIZE = False
#PATHS = ['D:\\08_Dataset\\Huawei_DS\\data\\mal\\00', 'D:\\08_Dataset\\Huawei_DS\\data\\mal\\01', 'D:\\08_Dataset\\Huawei_DS\\data\\clean\\00', 'D:\\08_Dataset\\Huawei_DS\\data\\clean\\01']
TIER1_MODELS = ['echelon_byte', 'echelon_featuristic', 'echelon_fusion']
TIER2_MODELS = ['echelon_byte_2', 'echelon_featuristic_2', 'echelon_fusion_2']
EXECUTION_TYPE = ['BYTE', 'FEATURISTIC', 'FUSION']
PLOT_TITLE = ['Byte Sequence', 'Superficial Features', 'MalFusion']
BYTE = 'BYTE'
FEATURISTIC = 'FEATURISTIC'
FUSION = 'FUSION'

TENSORBOARD_LOG_PATH = PROJECT_BASE_PATH + ESC + "log" + ESC + "tensorboard" + ESC
PLOT_PATH = PROJECT_BASE_PATH + ESC + "out" + ESC + "imgs" + ESC

SAVE_PATH = PROJECT_BASE_PATH  + ESC + 'model' + ESC   # help='Directory to save model and log'
MODEL_PATH = PROJECT_BASE_PATH + ESC + 'model' + ESC  # help="model to resume"


# #####################################################################################################################
# FEATURE MAP VISUALIZATION
# #####################################################################################################################
LAYER_NUM_TO_STUNT = 4 # 6 for echelon
PERCENTILES = [90] #[50, 75, 95]

COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats.csv'
COMMON_COMBINED_FEATURE_MAP_STATS_FILE = PROJECT_BASE_PATH + ESC + 'out' + ESC + 'result' + ESC + 'combined_stats_common.csv'
SECTION_SUPPORT = PROJECT_BASE_PATH + ESC + "out" + ESC + "result" + ESC + "section_support_by_samples.csv"

PADDING = "PADDING"
LEAK = "SECTIONLESS"

'''
# SECTION_STATS_HEADER = 'type,header,text,data,rsrc,pdata,rdata,padding\n'
SECTION_STATS_HEADER = 'type,header,text,data,pdata,rsrc,rdata,edata,idata,bss,reloc,debug,sdata,xdata,hdata,xdata,' \
                       'npdata,itext,apiset,qtmetad,textbss,its,extjmp,cdata,detourd,cfguard,guids,sdbid,extrel,' \
                       'ndata,detourc,shared,rodata,gfids,didata,pr0,tls,imrsiv,stab,mrdata,sxdata,orpc,c2r,nep,' \
                       'shdata,srdata,didat,stabstr,bldvar,isoapis\n'

PCS = ['.header', '.text', '.data', '.pdata', '.rsrc', '.rdata', '.edata', '.idata', '.bss', '.reloc', '.debug', '.sdata', '.xdata', '.hdata', '.xdata', '.npdata', '.itext', '.apiset', '.qtmetad', '.textbss', '.its', '.extjmp', '.cdata', '.detourd', '.cfguard', '.guids', '.sdbid', '.extrel', '.ndata', '.detourc', '.shared', '.rodata', '.gfids', '.didata', '.pr0', '.tls', '.imrsiv', '.stab', '.mrdata', '.sxdata', '.orpc', '.c2r', '.nep', '.shdata', '.srdata', '.didat', '.stabstr', '.bldvar', '.isoapis']
PCS_KEYS = ['header', 'text', 'data', 'pdata', 'rsrc', 'rdata', 'edata', 'idata', 'bss', 'reloc', 'debug', 'sdata', 'xdata', 'hdata', 'xdata', 'npdata', 'itext', 'apiset', 'qtmetad', 'textbss', 'its', 'extjmp', 'cdata', 'detourd', 'cfguard', 'guids', 'sdbid', 'extrel', 'ndata', 'detourc', 'shared', 'rodata', 'gfids', 'didata', 'pr0', 'tls', 'imrsiv', 'stab', 'mrdata', 'sxdata', 'orpc', 'c2r', 'nep', 'shdata', 'srdata', 'didat', 'stabstr', 'bldvar', 'isoapis']
'''
'''
FILES = [  # 'D:\\03_GitWorks\\echelon\\out\\result\\FP.csv',
           # 'D:\\03_GitWorks\\echelon\\out\\result\\FN.csv',
         PROJECT_BASE_PATH+'\\out\\result\\benign.csv',
         PROJECT_BASE_PATH+'\\out\\result\\malware.csv'
]
'''
'''
SECTION_STAT_FILES = [  # "D:\\03_GitWorks\\echelon\\out\\result\\FP_section_stats.csv",
                        # "D:\\03_GitWorks\\echelon\\out\\result\\FN_section_stats.csv",
                      PROJECT_BASE_PATH+"\\out\\result\\benign_section_stats.csv",
                      PROJECT_BASE_PATH+"\\out\\result\\malware_section_stats.csv"
]
'''

'''
file_type = ['BENIGN', 'MALWARE']  # , 'FP', 'FN']
plot_file = "D:\\03_GitWorks\\echelon\\out\\PE_Section_Statistics.png"
plot_file_no_fp = "D:\\03_GitWorks\\echelon\\out\\PE_Section_Statistics_NO_fp.png"
intuition1 = "D:\\03_GitWorks\\echelon\\out\\intuition1.png"
intuition2 = "D:\\03_GitWorks\\echelon\\out\\x.png"
'''
