import numpy as np
import matplotlib.pyplot as plt
from config import constants as cnst

def plot_auc(pObj):
    plt.clf()
    # AUC PLOT
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    # plt.plot([0.01, 0.01], [0, 1], 'black', linestyle=':', label="Target FPR")
    # plt.plot([0, 0.022], [0.9, 0.9], 'black', linestyle='-.', label="Target TPR")

    plt.xlabel("FPR %", fontsize=6)
    plt.ylabel("TPR %", fontsize=6)
    plt.xticks(np.arange(0, 6, 1), fontsize=6)
    plt.yticks(np.arange(0, 101, 10), fontsize=6)
    plt.xlim(0, 6)
    plt.ylim(0, 110)
    # plt.title(plot_title, fontdict = {'fontsize' : 6})
    plt.plot(fpr, tpr, lw=2, alpha=.8, label="AUC")
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig(cnst.PROJECT_BASE_PATH+ cnst.ESC + "out" + cnst.ESC + "imgs"+cnst.ESC+"class_ratio_1.png", bbox_inches='tight')
    #plt.show()

    plt.clf()
    # AUC PLOT
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    # plt.plot([0.01, 0.01], [0, 1], 'black', linestyle=':', label="Target FPR")
    # plt.plot([0, 0.022], [0.9, 0.9], 'black', linestyle='-.', label="Target TPR")

    plt.xlabel("Class Ratio", fontsize=6)
    plt.ylabel("TPR and FPR (%)", fontsize=6)
    plt.xticks(np.arange(0, len(xticklabels), 1), fontsize=6, labels=xticklabels)
    plt.yticks(np.arange(0, 101, 10), fontsize=6)
    plt.xlim(0, len(xticklabels))
    plt.ylim(0, 101)
    plt.title("CLASS RATIO BASED TPR vs FPR", fontdict={'fontsize': 6})
    plt.plot(np.arange(0, len(fpr), 1), fpr, lw=2, alpha=.8, label="FPR")
    plt.plot(np.arange(0, len(tpr), 1), tpr, lw=2, alpha=.8, label="TPR")
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig(cnst.PROJECT_BASE_PATH+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"class_ratio_2.png", bbox_inches='tight')
    #plt.show()

    plt.clf()
    # AUC PLOT
    dpi = 300
    figsize = (4, 2.7)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    plt.xlabel("Class Ratio [Benign : Malware]", fontsize=6)
    plt.ylabel("AUC / Restricted AUC %", fontsize=6)
    plt.xticks(np.arange(0, len(xticklabels), 1), fontsize=6, labels=xticklabels)
    plt.yticks(np.arange(0.5, 1.1, 0.05), fontsize=6)
    plt.xlim(0, len(xticklabels))
    plt.ylim(0.55, 1)
    plt.title("CLASS RATIO BASED AUC & RESTRICTED AUC", fontdict={'fontsize': 6})
    plt.plot(np.arange(0, len(auc), 1), auc, lw=2, alpha=.8, label="AUC")
    plt.plot(np.arange(0, len(rauc), 1), rauc, lw=2, alpha=.8, label="Restricted AUC")
    plt.legend(loc=1, prop={'size': 4})
    plt.savefig(cnst.PROJECT_BASE_PATH+cnst.ESC+"out"+cnst.ESC+"imgs"+cnst.ESC+"class_ratio_3.png", bbox_inches='tight')
    #plt.show()


xticklabels = ["1:1", "2:1", "3:1", "4:1", "5:1", "6:1", "7:1", "8:1", "9:1", "10:1"]
tpr = [83.16, 75.33, 72.66, 85.33, 89.50, 75.00, 83.66, 87.10]
fpr = [1.66, 3.08, 2.50, 2.37, 3.10, 2.47, 2.90, 2.41]
auc = [0.93, 0.85, 0.85, 0.91, 0.92, 0.85, 0.90, 0.92]
rauc = [0.66, 0.56, 0.57, 0.58, 0.57, 0.57, 0.57, 0.58]
plot_auc(None)
