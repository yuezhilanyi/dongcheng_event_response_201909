import os
import scipy.stats

import numpy as np
import pandas as pd

from collections import OrderedDict
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


STREET_CODE_MAP = {
    "东华门": 110101001,
    "景山": 110101002,
    "交道口": 110101003,
    "安定门": 110101004,
    "北新桥": 110101005,
    "东四": 110101006,
    "朝阳门": 110101007,
    "建国门": 110101008,
    "东直门": 110101009,
    "和平里": 110101010,
    "前门": 110101011,
    "崇外": 110101012,
    "东花市": 110101013,
    "龙潭": 110101014,
    "体育馆": 110101015,
    "天坛": 110101016,
    "永定门外": 110101017,
}


def read_source(file_path, sheetname=0):
    """
    读取源文件并简化，最后以结案日期为索引，生成一张新表用作训练和测试
    :param file_path: str, pandas pickle or excel file
    :param sheetname: int
    :return: pd.DataFrame
    """
    # read into dataframe
    if file_path.endswith("npy"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith("xlsx") or file_path.endswith("xls"):
        df = pd.read_excel(file_path, sheet_name=sheetname)
        # df.to_pickle('../event_data.npy')
    else:
        raise Exception("Only 'npy' or 'xlsx' or 'xls' supported, input file type: {}".format(file_path.split('.'[-1])))
    return df


def get_gt(dataframe, somewhere, someday):
    """
    根据时间和地点, 获取某个指数
    :param dataframe: pd.DataFrame
    :param somewhere: 街道, str
    :param someday: 日期, TimeSeries
    :return: 评价指数, float
    """
    somewhere = STREET_CODE_MAP[somewhere]  # 名称转编码(指数表格用的是编码)
    iso_someday = int(someday.isoformat().replace('-', ''))
    dataframe = dataframe.xs(iso_someday)
    res = dataframe[somewhere]
    return res


def new_df_until_someday(dataframe, somewhere, someday, write_path=''):
    # TODO: 仅仅筛选某日及之前未完成的
    day_start = pd.Timestamp(datetime.combine(someday, datetime.min.time()))
    day_end = pd.Timestamp(datetime.combine(someday, datetime.max.time()))
    # F1 - 某地所有案件
    dataframe = dataframe[dataframe["街道"] == somewhere]

    # F2.1 - 监督员自行处理的案件
    dataframe_self = dataframe[dataframe["问题来源"] == "监督员自行处理"]
    if len(dataframe_self) != 0:  # 防止 UserWarning: Boolean Series key will be reindexed to match DataFrame index
        dataframe_self = dataframe_self[dataframe["上报时间"].dt.date == someday]  # 某日自行处理案件, 默认为当天完成(结案时间不正确)
    # F2.2 - 剩余案件 (均具有处置截止时间, 处置结束时间, 结案时间, 立案时间等信息)
    dataframe = dataframe[dataframe["问题来源"] != "监督员自行处理"]

    # F3 (<-F2.2) - 考核日之前的所有案件
    dataframe = dataframe[dataframe["上报时间"] <= day_end]

    # F4 (<-F3) - 截至考核日0点未结案案件
    dataframe = dataframe[-(dataframe["处置结束时间"] <= day_start)]

    # basename = int(day_end.isoformat().replace('-', '')[:8])
    basename = int(someday.isoformat().replace('-', ''))
    if write_path:
        if not os.path.isdir(write_path):
            os.mkdir(write_path)
        try:
            dataframe.to_excel(os.path.join(write_path, '{}_{}.xlsx'.format(somewhere, basename)))
        except PermissionError as e:
            print(e)

    return dataframe, dataframe_self


def finished_before_someday(df, someday):
    # 必须包含的列: 统计日, 上报时间, 处置截止时间, 处置结束时间
    day_start = pd.Timestamp(datetime.combine(someday, datetime.min.time()))
    day_end = pd.Timestamp(datetime.combine(someday, datetime.max.time()))

    def func_less_than_day_end(x): return day_end if x > day_end else x

    df["当日处置结束时间"] = df["处置结束时间"].apply(func_less_than_day_end)

    return df[df["处置截止时间"] > df["当日处置结束时间"]], df[df["处置截止时间"] <= df["当日处置结束时间"]]


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


def cal_coef(train_df, label_df):
    SPM = scipy.stats.spearmanr
    res = OrderedDict()
    for col in train_df.columns:
        if train_df[col].any():
            coef, p = (SPM(train_df[col], label_df))
        else:
            coef, p = -1, -1
        res[col] = (coef, p)
    return res


def regression_test(input_file_path):
    # regression
    df2 = pd.read_excel(input_file_path)
    try:
        Y = df2["原指标"]
        X = df2.drop(["Unnamed: 0", "日期", "街道", "原指标"], axis=1)
    except KeyError:
        Y = df2["('原指标', '')"]
        X = df2.drop(["Unnamed: 0", "('日期', '')", "('街道', '')", "('原指标', '')"], axis=1)  # 两级索引
    linear_reg_test(X, Y, X, df2.index)
    polynomial_reg_test(X, Y, X, df2.index)

    # 计算相关系数
    result_dict = cal_coef(X, Y)
    for k, v in result_dict.items():
        print(k, v)

    a, b = scipy.stats.spearmanr(X, Y)
    print(a, b)


def index_analysis(input_path):
    df = pd.read_excel(input_path)
    df = df.iloc[:, 1:]  # remove 日期
    arr = df.to_numpy()
    uni = np.unique(arr)
    arr = arr / uni[0] if uni[0] != 0 else arr / uni[1]
    bins = np.bincount(arr.flatten().astype('uint16'))
    return uni, bins
