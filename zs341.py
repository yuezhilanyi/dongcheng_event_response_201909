"""
四级指数编号 - ZS341
四级指数 - 服务需求指数
指数说明 - 劳动关系纠纷、社会福利、社会保障、社会事业等的需求分布
计算方式 - 网格内劳动关系纠纷、社会福利、社会保障、社会事业等相关诉求案件的数量
评价方式 - 负向评价
权重 - 1
预警阈值 - max(10%）
分级规则 - 四分之一数，中位数，四分之三数
备注 - 无
"""
import tqdm

import pandas as pd

from utils import read_source, new_df_until_someday, get_gt
from utils import linear_reg_test, polynomial_reg_test, cal_coef, spearmanr


def dataframe_preprocess(file_path, sheetname=0):
    """
    读取源文件并简化
    :param file_path:
    :param sheetname:
    :return:
    """
    df = read_source(file_path, sheetname=sheetname)

    # 一次筛选
    # 筛选后大类名称: 服务项目, 劳动与社会保障(包含劳动关系与纠纷, 社会福利与保障), 矛盾纠纷, 社会事业, 特殊行业监管
    df = df[df["问题类型"] == "社会服务管理"]
    # 二次筛选
    # TODO: 做两次分析, 分别为包含二次筛选和不包含
    # df = df[(df["小类名称"] == "劳动关系与纠纷") |
    #         (df["小类名称"] == "社会福利与保障") |
    #         (df["大类名称"] == "社会事业")]
    # filter with ['当前阶段'] == '[作废]'
    df = df[df['当前阶段'] != '[作废]']  # 根据2019.9.18与网格中心考评处的沟通, 不考虑作废案件

    # keep only useful columns, to reduce too many dimensions
    df = df[['问题类型', '大类名称', '小类名称', '街道', '上报时间', '当前阶段', '处置截止时间', '处置结束时间']]

    return df


def convert_to_new_dataframe(srs_df, write_path=''):
    # 读取文件
    srs_df["上报日期"] = srs_df["上报时间"].dt.date
    df_gt = pd.read_excel("../source_data/ZS341 - 服务需求指数.xlsx")
    df_gt.set_index(df_gt["日期"], inplace=True)

    index = srs_df["上报日期"].value_counts().index
    index = sorted(index)

    lst = []
    # for i in tqdm.trange(30):
    for i in tqdm.trange(len(index)):
        day = index[i]
        assert day is not None

        for area in ["东华门", "景山", "交道口", "安定门", "北新桥", "东四", "朝阳门", "建国门", "东直门", "和平里",
                     "前门", "崇外", "东花市", "龙潭", "体育馆", "天坛", "永定门外"]:  # 和网格代码顺序一致, 方便后续观察对比
            # tqdm.tqdm.write(area)
            df, df_self = new_df_until_someday(srs_df, area, day, write_path=write_path)
            df = df[-(df["处置结束时间"] < pd.Timestamp(day))]

            # 获取原指标
            gt = get_gt(df_gt, area, day)

            # 如果 dataframe 为空, 跳至下一个
            if len(df) == 0:
                lst.append([day, area, 0, len(df_self), 0, 0, 0, 0, 0, 0, gt])
                continue

            # 截至统计日, 所有未完成的案件
            n1 = len(df)  # 社会服务管理案件总数
            n1a = len(df[df["小类名称"] == "劳动关系与纠纷"])  # 劳动关系与纠纷
            n1b = len(df[df["小类名称"] == "社会福利与保障"])  # 社会福利与保障
            n1c = len(df[df["大类名称"] == "社会事业"])  # 社会事业
            # 统计日当日, 未完成的案件
            df = df[df["上报时间"].dt.date == day]  # 当天案件总数
            n2 = len(df[df["上报时间"].dt.date == day])  # 当天案件总数
            n2a = len(df[df["小类名称"] == "劳动关系与纠纷"])  # 劳动关系与纠纷
            n2b = len(df[df["小类名称"] == "社会福利与保障"])  # 社会福利与保障
            n2c = len(df[df["大类名称"] == "社会事业"])  # 社会事业

            lst.append([day, area, n1, n1a, n1b, n1c, n2, n2a, n2b, n2c, gt])

    res = pd.DataFrame(lst, columns=["日期", "街道",
                                     "社会服务管理案件总数", "劳动关系与纠纷(总计)", "社会福利与保障(总计)", "社会事业(总计)",
                                     "当天案件总数", "劳动关系与纠纷(当日)", "社会福利与保障(当日)", "社会事业(当日)", "原指标"])

    return res


if __name__ == "__main__":
    # source_file = '../queryResult_2019-09-10_145030_zs341.xlsx'
    # source_file = '../queryResult_2019-09-10_145030.npy'
    # df1 = dataframe_preprocess(source_file)
    #
    # df2 = convert_to_new_dataframe(df1, write_path='../tmp_zs341')
    # df2.to_excel('../zs341_20190923.xlsx')

    # regression
    df2 = pd.read_excel('../zs341_20190923.xlsx')
    df2 = df2[df2["原指标"] != 0]
    Y = df2["原指标"]
    X = df2.drop(["Unnamed: 0", "日期", "街道", "原指标"], axis=1)
    linear_reg_test(X, Y, X, df2.index)
    polynomial_reg_test(X, Y, X, df2.index)

    # 计算相关系数
    result_dict = cal_coef(X, Y)
    for k, v in result_dict.items():
        print(k, v)

    a, b = spearmanr(X, Y)
    print(a, b)
