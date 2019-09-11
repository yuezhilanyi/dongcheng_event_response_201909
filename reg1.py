import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def read_excel(file_path, sheetname=0, columns=None):
    """

    :param file_path:
    :param sheetname:
    :param columns:
    :return:
    """
    # read excel and join multiple rows into one row
    if columns is None:
        columns = ['area', 'x1', 'x2', 'x10', 'x11', 'x12', 'x13', 'x15']
    res = pd.read_excel(file_path, sheetname=sheetname)
    res.columns = ['no', 'area', 'year', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                   'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']

    # remove columns containing too many N/A
    res = res[columns]
    # res = res.fillna(0)

    # https://stackoverflow.com/questions/51901068/how-to-combine-multiple-rows-into-a-single-row-with-python-pandas-based-on-the-v
    res_out = res.set_index(['area', res.groupby(['area']).cumcount() + 1]).unstack().sort_index(level=1, axis=1)
    res_out.columns = res_out.columns.map('{0[0]}_{0[1]}'.format)
    res_out = res_out.reset_index()
    return res_out


# work_excel = 'work.xlsx'
work_excel = 'work_without_prc.xlsx'
# read excel and join multiple rows into one row
df1 = read_excel(work_excel)

# read label
df2 = pd.read_excel('label.xlsx')
df2a = df2.iloc[:, :2]
df2b = df2.iloc[:, 2:]
df2a.columns = ['area', 'mean_value']
df2b.columns = ['area', 'mean_value']


def convert_data_and_label_to_train(data_df, label_df):
    df = pd.merge(data_df, label_df, on='area')
    train_label = df['mean_value']
    train_data = df.drop('mean_value', 1)
    del train_data['area']
    return train_data, train_label


def linear_reg_test(train_data, train_label, test_data, index):
    """
    线性回归
    """
    lr = LinearRegression(normalize=True)
    lr.fit(train_data, train_label)

    print(lr.intercept_, '\n', lr.coef_)

    y1 = lr.predict(test_data)
    y1 = pd.Series(y1)
    y1.index = index
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options
        print(y1)
    return y1, lr


def polynomial_reg_test(train_data, train_label, test_data, index):
    """
    多项式回归
    """
    quadratic_featurizer = PolynomialFeatures(degree=2)
    X_train_quadratic = quadratic_featurizer.fit_transform(train_data)
    regressor_quadratic = LinearRegression(normalize=True)
    regressor_quadratic.fit(X_train_quadratic, train_label)

    print(regressor_quadratic.intercept_, '\n', regressor_quadratic.coef_)

    test_data = quadratic_featurizer.fit_transform(test_data)
    y1 = regressor_quadratic.predict(test_data)
    y1 = pd.Series(y1)
    y1.index = index
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options
        print(y1)
    return y1, regressor_quadratic


if __name__ == "__main__":
    # train data
    X, Y = convert_data_and_label_to_train(df1, df2a)
    # test data
    test_out = read_excel('test.xlsx')
    test_index = test_out['area']
    del test_out['area']

    linear_reg_test(X, Y, test_out, test_index)
    polynomial_reg_test(X, Y, test_out, test_index)
