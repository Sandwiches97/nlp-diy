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
        for seed in range(10):
            seed *= 100
            f_test = fdata["test_acc_list_seed_%d"%seed]
            f_test_list+=f_test
        f_test_list /= 9
        test_max = max(f_test_list)
        record_test.append(test_max)
    acc = np.array(record_test)
    return np.mean(acc)



path = 'E:\FangC\SourceCode\pytorch\snn_net\deap\C3B1/subject_dependent_label_v/*.csv'
                                                    # subject_dependent_label_a;
                                                    # subject_dependent_label_v ;
                                                    # subject_dependent_label_d
a = get_max_acc(path)
print(a)