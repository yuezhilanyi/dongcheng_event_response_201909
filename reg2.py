import pandas as pd
import pickle

from reg1 import df2a, df2b
from reg1 import linear_reg_test, polynomial_reg_test, convert_data_and_label_to_train


def read_excel2(file_path, sheetname=0):
    """

    :param file_path:
    :param sheetname:
    :param columns:
    :return:
    """
    # read excel and join multiple rows into one row
    res = pd.read_excel(file_path, sheetname=sheetname)
    res.columns = ['no', 'area', 'year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                   'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    del res['no'], res['year']

    return res


work_excel = 'work2.xlsx'
# read excel and join multiple rows into one row
df1 = read_excel2(work_excel)


if __name__ == "__main__":
    # train data
    X, Y = convert_data_and_label_to_train(df1, df2a)
    # test data
    test_out = read_excel2('test2_2.xlsx')
    test_index = test_out['area']
    del test_out['area']

    linear_reg_test(X, Y, test_out, test_index)
    _, model = polynomial_reg_test(X, Y, test_out, test_index)
    pickle.dump(model, open('polynomial.sav', 'wb'))
