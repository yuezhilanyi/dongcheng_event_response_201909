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

            # 如果 dataframe 为空, 跳至下一个
            if len(df) == 0:
                lst.append([day, area, 0, len(df_self), 0, 0, 0, 0])
                continue

            # 开始统计
            n1 = len(df)  # 社会服务管理案件总数
            n2 = len(df[df["上报时间"].dt.date == day])  # 当天案件总数
            n3 = len(df[df["小类名称"] == "劳动关系与纠纷"])  # 劳动关系与纠纷
            n4 = len(df[df["小类名称"] == "社会福利与保障"])  # 社会福利与保障
            n5 = len(df[df["大类名称"] == "社会事业"])  # 社会事业
            gt = get_gt(df_gt, area, day)

            lst.append([day, area, n1, n2, n3, n4, n5, gt])

    res = pd.DataFrame(lst, columns=["日期", "街道",
                                     "社会服务管理案件总数", "当天案件总数",
                                     "劳动关系与纠纷", "社会福利与保障", "社会事业", "原指标"])

    return res


def cal_index(input_excel):
    """
    根据"强制结案总数", "计划内耗时总长(分钟)", "计划外耗时总长(分钟)"三个指标得到基础分;
    根据"自行处理案件总数", 奖励一定分数;
    根据"强制结案总数", 扣除一定分数;
    "其他案件总数"不作为打分依据(一定程度上,已经在"计划内(外)耗时总长中得到体现);
    "立案耗时总长(分钟)"不作为打分依据, 因本指标为"处置效能指数", 不牵涉从上报到立案
    :return:
    """
    df = pd.read_excel(input_excel)
    # TODO: 增加当天内完成案件的权重(比如自行处理案件因当天完成, 相比第二天完成的案件, 少了一晚上的执行分数)
    w1 = 60  # 自行处理案件总数, 认为w1分钟为完成一个自行处理案件所需的平均时间
    w2 = 0  # 其他案件总数, 不考评
    w3 = 0  # 强制结案总数, 暂不考评(没有好的思路, 且强制结案会同时生成一个新的案件)
    w4 = 0  # 立案耗时总长(分钟), 不考评
    w5 = 1  # 计划内耗时总长(分钟)
    w6 = -1  # 计划外耗时总长(分钟)
    df['新评分'] = (
                        df["自行处理案件总数"] * w1 +
                        df["其他案件总数"] * w2 +
                        df["强制结案总数"] * w3 +
                        df["立案耗时总长(分钟)"] * w4 +
                        df["计划内耗时总长(分钟)"] * w5 +
                        df["计划外耗时总长(分钟)"] * w6
                ) / 1000
    return df


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
    index = df2.index
    Y = df2["原指标"]
    X = df2.drop(["Unnamed: 0", "日期", "街道", "原指标"], axis=1)
    linear_reg_test(X, Y, X, index)
    polynomial_reg_test(X, Y, X, index)

    # 计算相关系数
    result_dict = cal_coef(X, Y)
    for k, v in result_dict.items():
        print(k, v)

    a, b = spearmanr(X, Y)
    print(a, b)