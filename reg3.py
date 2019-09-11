import scipy.stats
import pandas as pd

from collections import OrderedDict

from reg1 import df2a, df2b
from reg1 import linear_reg_test, polynomial_reg_test, convert_data_and_label_to_train
from reg2 import read_excel2


def cal_coef(train_df, label_df):
    SPM = scipy.stats.spearmanr
    res = OrderedDict()
    for col in train_df.columns:
        coef, p = (SPM(train_df[col], label_df))
        res[col] = (coef, p)
    return res


work_excel = 'work2.xlsx'
# read excel and join multiple rows into one row
df1 = read_excel2(work_excel)
# del df1['x7'], df1['x8']


if __name__ == "__main__":
    # train data
    X, Y = convert_data_and_label_to_train(df1, df2a)
    result_dict = cal_coef(X, Y)
    for k, v in result_dict.items():
        print(k, v)

    a, b = scipy.stats.spearmanr(X, Y)
    print(a, b)
