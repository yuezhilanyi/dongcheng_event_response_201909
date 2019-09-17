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

    # keep only useful columns, to reduce too many dimensions
    df = df[["问题类型", "街道", "上报时间", "当前阶段", "处置时限", "处置用时", "结案时间", "处置超时倍数",
             "是否自处置", "强制结案数", "实际办结数", "准办结数", "返工数", "超时结案数"]]
    """ 其他可能用到的字段
    问题类型	小类名称	街道	上报时间	问题描述	当前阶段	处置时限	延期类型	处置截止时间	处置结束时间	问题状态	
    结案类型	强结限制	案件类型	处置用时	捆绑处置用时	结案时间	受理开始时间	案件阶段名称	是否自处置	处置超时倍数	
    强制结案数	实际办结数	准办结数	返工数	超时结案数	受理时间	受理数	按时督查数	按时受理数	按时立案数	
    按时处置数	按时派遣数	按时核查数	按时结案数	立案时间	立案数	处置数	处置开始时间	核查时间	结案数	
    强制结案时间	当前阶段标识	NUM
    """

    return df


def convert_to_new_dataframe(df):
    # # add some fields
    df["结案日期"] = df["结案时间"].dt.date

    # new dataframe
    index = df["结案日期"].value_counts().index
    index = sorted(index)

    def dataframe2list(dataframe, somewhere, someday):
        """

        :param dataframe: pandas dataframe
        :param somewhere: str,
        :param someday: datetime.date
        :return: list consist of str, float and int values
        """
        dataframe = dataframe[dataframe["街道"] == somewhere]
        dataframe1 = dataframe[dataframe["结案日期"] == someday]
        # TODO: 添加未结案的并加权
        dataframe2 = dataframe[pd.isnull(dataframe["结案日期"])]
        dataframe2 = dataframe2[dataframe2["上报时间"] <= someday]

        merge = dataframe1.append(dataframe2)

        total = len(merge)  # 案件总数
        dti = 0  # disposal time index
        t7 = len(dataframe2)  # "未结案总数"
        t8 = len(merge[merge['当前阶段'] == '[作废]'])  # "作废案件总数"
        t1, t2, t3, t4, t5, t6, = 0, 0, 0, 0, 0, 0  # "自处置总数", "强制结案总数", "实际办结总数",
        # "准办结总数", "返工总数", "超时结案总数"
        for _, row in merge.iterrows():
            total += 1
            if pd.isnull(row["处置时限"]):  # 某些没有处置时限, 比如作废的
                dti += 0
            else:
                if pd.isnull(row["结案日期"]):  # 未结案的
                    # TODO: 有些未结案,但是已处置,有处置用时,比如 NO 10614
                    # https://stackoverflow.com/questions/1937622/convert-date-to-datetime-in-python/1937636
                    deadline = datetime.combine(someday, datetime.max.time())
                    used_time = deadline - row["上报时间"]
                    days = used_time.days
                    # https://stackoverflow.com/questions/31283001/get-total-number-of-hours-from-a-pandas-timedelta
                    hours = used_time / pd.Timedelta(hours=1) - days * 24
                    if days == 0:  # 当天
                        row["处置用时"] = hours
                    else:
                        row["处置用时"] = days * 9 + (hours - 9)
                # TODO: 要求对于尚未结案的,需要及时更新处置超时倍数,否则会发生结案后处置及时指数反常的情况
                dti += row["处置用时"] / row["处置时限"] * (row["处置超时倍数"] + 1)  # 处置超时倍数取值(0-3), 所以需要+1
            t1 += row["是否自处置"]
            t2 += row["强制结案数"]
            t3 += row["实际办结数"]
            t4 += row["准办结数"]
            t5 += row["返工数"]
            t6 += row["超时结案数"]
        return someday, somewhere, total, dti / total, t1, t2, t3, t4, t5, t6, t7, t8

    lst = []
    for i in tqdm.trange(len(index)):
        day = index[i]
        assert day is not None
        for area in df["街道"].value_counts().index:
            # tqdm.tqdm.write(area)
            result = dataframe2list(df, area, day)
            lst.append(result)

    res = pd.DataFrame(lst, columns=["日期", "街道",
                                     "案件总数", "处置及时指数", "自处置总数", "强制结案总数", "实际办结总数",
                                     "准办结总数", "返工总数", "超时结案总数", "未结案总数", "作废案件总数"])

    return res


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
    source_file = '../event_data_simplified.npy'
    df1 = read_source(source_file)
    # df1.to_pickle('../event_data_simplified.npy')

    df2 = convert_to_new_dataframe(df1)
    df2.to_csv('../ndf.csv')

    # dm = DataModel(df1)
    # df1 = dm.cal_all()
    # df1.to_pickle('../one_line.npy')
    # df1.to_excel('../one_line.xlsx')
