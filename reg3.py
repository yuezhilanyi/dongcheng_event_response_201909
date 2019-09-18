import os
import tqdm

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta


def read_source(file_path, sheetname=0):
    """
    读取源文件并简化，最后以结案日期为索引，生成一张新表用作训练和测试
    :param file_path:
    :param sheetname:
    :return:
    """
    # read into dataframe
    if file_path.endswith("npy"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith("xlsx") or file_path.endswith("xls"):
        df = pd.read_excel(file_path, sheetname=sheetname)
        # df.to_pickle('../event_data.npy')
    else:
        raise Exception("Only 'npy' or 'xlsx' or 'xls' supported, input file type: {}".format(file_path.split('.'[-1])))

    # filter with ["问题类型"] == "事件"
    df = df[df["问题类型"] == "事件"]
    # filter with ['当前阶段'] == '[作废]'
    df = df[df['当前阶段'] != '[作废]']  # 根据2019.9.18与网格中心考评处的沟通, 不考虑作废案件

    # keep only useful columns, to reduce too many dimensions
    df = df[['案件号', '问题来源', '问题类型', '大类名称', '街道', '上报时间', '当前阶段', '处置截止时间','处置结束时间',
             '结案时间', '立案时间', '强制结案时间']]
    """
    '案件号', '问题来源', '问题类型', '大类名称', '街道', '上报时间', '当前阶段', '延期类型', '案件标识', '处置截止时间', 
    '处置结束时间', '强结限制', '案件类型', '结案时间', '首次派遣时间', '受理开始时间', '案件阶段名称', '受理时间', 
    '立案时间', '强制结案时间'
    """
    return df


def convert_to_new_dataframe(df):
    # # add some fields
    df["结案日期"] = df["结案时间"].dt.date
    df_gt = pd.read_excel("../ZS222.xlsx")
    df_gt.set_index(df_gt["日期"], inplace=True)

    # new dataframe
    index = df["结案日期"].value_counts().index
    index = sorted(index)

    def get_gt(dataframe_gt, somewhere, someday):
        iso_someday = int(someday.isoformat().replace('-', ''))
        dataframe_gt = dataframe_gt.xs(iso_someday)
        gt = dataframe_gt[somewhere]
        return gt

    def dataframe2list(dataframe, somewhere, someday):
        """

        :param dataframe: pandas dataframe
        :param somewhere: str,
        :param someday: datetime.date
        :return: list consist of str, float and int values
        """
        # F1 - 某地所有案件
        dataframe = dataframe[dataframe["街道"] == somewhere]

        # F2.1 - 监督员自行处理的案件
        df_self = dataframe[dataframe["问题来源"] == "监督员自行处理"]
        df_self = df_self[dataframe["上报时间"].dt.date == someday]  # 某日自行处理案件, 默认为当天完成(结案时间不正确)
        n1 = len(df_self)  # 自行处理案件总数
        # F2.2 - 剩余案件 (均具有处置截止时间, 处置结束时间, 结案时间, 立案时间等信息)
        dataframe = dataframe[dataframe["问题来源"] != "监督员自行处理"]

        # F3 (<-F2.2) - 考核日之前的所有案件
        dataframe = dataframe[dataframe["上报时间"] <= someday]
        # F3.1 非考核日之前结案
        dataframe = dataframe[-dataframe["结案日期"] < someday]

        n2 = len(dataframe)  # 非自行处理案件总数
        dti = 0  # disposal time index

        t1, t2p, t2a = None, None, None

        for _, row in dataframe.iterrows():
            t1 = (row["立案时间"] - row["上报时间"]).seconds
            t2p = (row["处置截止时间"] - row["立案时间"]).seconds
            t2a = (row["处置结束时间"] - row["立案时间"]).seconds
        return someday, somewhere, n1, n2, t1, t2p, t2a

    df["立案耗时"] = df["立案时间"].subtract(df["上报时间"])
    df["处置预计耗时"] = df["处置截止时间"].subtract(df["立案时间"])
    df["处置实际耗时"] = df["处置结束时间"].subtract(df["立案时间"])

    return df


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
    # define source file path
    # source_file = '../queryResult_2019-09-10_145030_processed.xlsx'
    source_file = '../event_data.npy'
    df1 = read_source(source_file)
    # df1.to_pickle('../event_data_simplified.npy')

    df2 = convert_to_new_dataframe(df1)
    df2.to_excel('../ndf.xlsx')

    # dm = DataModel(df1)
    # df1 = dm.cal_all()
    # df1.to_pickle('../one_line.npy')
    # df1.to_excel('../one_line.xlsx')
