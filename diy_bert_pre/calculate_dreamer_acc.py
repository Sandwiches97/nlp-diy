import numpy as np
import pandas as pd
import glob


def get_max_acc(path:str):
    name = glob.glob((path))
    record_test, record_val = [], []
    for it in name:
        if it.__contains__('dataframe_best'):
            continue
        fdata = pd.read_csv(it)
        f_test_list = 0
        for seed in range(9):
            seed *= 100
            f_test = fdata["test_acc_list_seed_%d"%seed]
            f_test_list+=f_test
        f_test_list /= 9
        test_max = max(f_test_list)
        record_test.append(test_max)
    acc = np.array(record_test)
    return np.mean(acc)

# def get_epoch_acc(path:str, epoch:int):
#     name = glob.glob((path))
#     epoch %= 1000
#     record_test, record_val = [], []
#     for it in name:
#         if it.__contains__('dataframe_best'):
#             continue
#         fdata = pd.read_csv(it)
#         f_train = fdata["train_acc_list"]
#         f_test = fdata["test_acc_list"]
#         f_val = fdata["val_acc_list"]
#         test_epoch = f_test[epoch]
#         val_epoch = f_val[epoch]
#         record_test.append(test_epoch)
#         record_val.append(val_epoch)
#
#     acc = (np.array(record_val) + np.array(record_test)) / 2
#     return np.mean(acc)

# def get_acc_under_max_val(path:str):
#     name = glob.glob((path))
#     record_test, record_val = [], []
#     for it in name:
#         if it.__contains__('dataframe_best'):
#             continue
#         fdata = pd.read_csv(it)
#         f_train = fdata["train_acc_list"]
#         f_test = fdata["test_acc_list"]
#         f_val = fdata["val_acc_list"]
#         test_max = max(f_test)
#         val_max = max(f_val)
#
#         val_idx_list = np.where(f_val == val_max)[0]
#         test_idx_list = np.where(f_test == test_max)[0]
#         new_val_idx_list = np.array([f_test[k] for k in val_idx_list])
#         new_test_idx_list = np.array([f_val[k] for k in test_idx_list])
#
#         record_test.append(np.max(new_test_idx_list))
#         record_val.append(np.max(new_val_idx_list))
#
#     acc = (np.array(record_val)+np.array(record_test))/2
#     return np.mean(record_val), np.mean(record_test)


path = 'E:\FangC\SourceCode\pytorch\snn_net\dreamer\C3B1/subject_dependent_ScoreDominance/*.csv'
                                                    # subject_dependent_ScoreArousal;
                                                    # subject_dependent_ScoreValence ;
                                                    # subject_dependent_ScoreDominance
a = get_max_acc(path)
print(a)