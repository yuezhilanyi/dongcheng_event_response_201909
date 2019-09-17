import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
from tqdm import trange


class DataModel(object):
    def __init__(self, dataframe):
        """

        :param dataframe: DataFrame
        """
        self.df = dataframe
        self.someday = None  # datetime.date for index
        self.index = None

    def count_all_events_of_someday(self):
        """
        统计当日所有事件的数量（当天新增）
        :return:
        """
        c1 = self.df[self.df["结案时间"].dt.date == self.someday]  # 当日结案的所有事件
        c2 = self.df[self.df["上报时间"].dt.date <= self.someday]
        c2 = c2[pd.isnull(c2["结案时间"])]  # 截至当日，已上报但是没有结案的事件
        c = len(c1) + len(c2)
        return c, len(c1), len(c2)

    def count_obsolete_events_of_someday(self):
        """
        以上报时间为依据，统计当日作废条目的数量
        :return: int, number of obsolete events of the dataframe
        """
        # https://stackoverflow.com/questions/28133018/convert-pandas-series-to-datetime-in-a-dataframe
        c = self.df[self.df["上报时间"].dt.date == self.someday]  # 查询日当天
        c = c[c['当前阶段'] == '[作废]']  # 作废
        return len(c)

    def cal_non_obsolete_events(self):
        """
        以下四种情况：
        有结案时间（查询时间）
            a. 提前或按期完成, 结案时间（查询时间）早于处置截止时间。
            b. 延期完成, 结案时间（查询时间）晚于处置截止时间。
        无结案时间（上报时间早于查询日24点）
            有处置截止时间：
                c. 进行中
            无处置截止时间：
                d. 超期未完成
        :return:  int list
        """
        c = self.df[self.df["结案时间"].dt.date == self.someday]  # 查询日当天
        c = c[c['当前阶段'] != '[作废]']  # 未作废

        # 有结案时间
        c1 = c[-pd.isnull(c["结案时间"])]
        # a = c1[c1['结案时间'] < c1['处置截止时间'] + timedelta(days=1)]
        a = c1[c1['结案时间'] <= c1['处置截止时间']]
        wa = 1
        b = c1[c1['结案时间'] > c1['处置截止时间']]
        wb = 1

        # 无结案时间
        # TODO: 这里都为0(因为前边使用了结案时间做筛选), 需要优化
        c2 = c[pd.isnull(c["结案时间"])]
        c = c2[-pd.isnull(c["处置截止时间"])]  # 有处置截止时间
        wc = 1
        d = c2[pd.isnull(c["处置截止时间"])]  # 无处置截止时间
        wd = 1

        return len(a), wa, len(b), wb, len(c), wc, len(d), wd

    def cal_all(self):
        # get all '结案时间' list
        index = self.df['结案时间'].dt.date.value_counts().index
        self.index = sorted(index)
        # new dataframe group by "结案时间"
        # TODO: add rows that has no events, May Day, Sprint Festival, etc.
        res = pd.DataFrame(index=self.index, columns=['作废数量', '截至当日总计', '截至当日结案总计', '截至当日未结案总计',
                                                      '按期完成数量', '按期权重',
                                                      '延期完成数量',  '延期权重',
                                                      '超期未完成数量', '超期权重',
                                                      '进行中数量', '进行中权重',
                                                      ])  # 均为当天数据
        for i in trange(len(index)):
            day = index[i]
            assert day is not None
            self.someday = day
            obsolete = self.count_obsolete_events_of_someday()
            total, t1, t2 = self.count_all_events_of_someday()
            c1, w1, c2, w2, c3, w3, c4, w4 = self.cal_non_obsolete_events()
            res.iloc[i] = [obsolete, total, t1, t2,
                           c1, w1, c2, w2, c3, w3, c4, w4]
        return res


def read_source(file_path, output_path, sheetname=0, save_to_npy=True, save_to_excel=False):
    """
    读取源文件并简化，最后以结案日期为索引，生成一张新表用作训练和测试
    :param file_path:
    :param output_path:
    :param sheetname:
    :param save_to_npy:
    :param save_to_excel:
    :return:
    """
    # read into dataframe
    if file_path.endswith("npy"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith("xlsx") or file_path.endswith("xls"):
        df = pd.read_excel(file_path, sheetname=sheetname)
    else:
        raise Exception("Only 'npy' or 'xlsx' or 'xls' supported, input file type: {}".format(file_path.split('.'[-1])))

    # filter with ["问题类型"] == "事件"
    df = df[df["问题类型"] == "事件"]

    # keep only useful columns, to reduce too many dimensions
    df = df[["处置用时", "上报时间", "问题类型", "案件类型", "小类名称", "街道", "当前阶段", "强结限制", "延期类型",
             "问题状态", "发核查数", "处置截止时间", "结案时间", "作废时间"]]

    # add some fields
    df["结案日期"] = df["结案时间"].dt.date
    df["处置限制时长"] = df["处置截止时间"].subtract(df["上报时间"])
    df["实际花费时长"] = df["结案时间"].subtract(df["上报时间"])

    output_path = os.path.splitext(output_path)[0]
    if save_to_npy:
        df.to_pickle(output_path + '.npy')
    if save_to_excel:
        df.to_excel(output_path + '.xlsx')

    return df


def convert_to_new_dataframe(res):
    # filter with ['当前阶段'] != '[作废]', for there is no '结案时间' meanwhile
    # TODO: 对某一日数据，统计作废事件数量（按上报时间）后，去除作废事件（没有结案时间），再进行其他指标的计算
    res = res[['当前阶段'] != '[作废]']


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
    source_file = '../event_data.npy'
    df1 = read_source(source_file, output_path='../event_data_simplified')

    dm = DataModel(df1)
    df1 = dm.cal_all()
    df1.to_pickle('../one_line.npy')
    df1.to_excel('../one_line.xlsx')
