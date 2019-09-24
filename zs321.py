"""
四级指数编号 - ZS321
四级指数 - 环境问题指数
指数说明 - 道路不洁、破损、地锁、绿地脏乱等环境案件的分布情况
计算方式 - 网格内道路不洁、破损、绿地脏乱等环境案件的数量
评价方式 - 负向评价
权重 - 1
预警阈值 - max(10%）
分级规则 - 四分之一数，中位数，四分之三数
备注 - 无
"""
import tqdm

import pandas as pd

from utils import read_source, new_df_until_someday, get_gt
from utils import regression_test


def dataframe_preprocess(file_path, sheetname=0):
    """
    读取源文件并简化
    :param file_path:
    :param sheetname:
    :return:
    """
    df = read_source(file_path, sheetname=sheetname)

    # 一次筛选
    df = df[df['当前阶段'] != '[作废]']  # 根据2019.9.18与网格中心考评处的沟通, 不考虑作废案件

    # keep only useful columns, to reduce too many dimensions
    df = df[['问题来源', '问题类型', '大类名称', '小类名称', '小类明细', '微类名称',
             '街道', '上报时间', '当前阶段', '处置截止时间', '处置结束时间']]

    # 二次筛选
    df = df.loc[(df["问题类型"] == "事件") & (df["大类名称"] == "市容环境")]
    return df


def convert_to_new_dataframe(srs_path, gt_path, write_path=''):
    # 读取文件
    srs_df = dataframe_preprocess(srs_path)
    srs_df["上报日期"] = srs_df["上报时间"].dt.date
    df_gt = pd.read_excel(gt_path)
    df_gt.set_index(df_gt["日期"], inplace=True)

    index = srs_df["上报日期"].value_counts().index
    index = sorted(index)

    res = pd.DataFrame()

    j = 0
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

            # 截至统计日, 所有未完成的案件
            n1 = len(df)  # 案件总数
            # 统计日当日, 未完成的案件
            n2 = len(df[df["上报时间"].dt.date == day])  # 当天案件总数
            # 自行处理的案件总数
            n3 = len(df_self)
            # 分类统计
            s = df.groupby("小类名称").count()["上报时间"]
            s["案件总数"] = n1
            s["当天案件总数"] = n2
            s["自行处理案件总数"] = n3
            s["日期"] = day
            s["街道"] = area
            s["原指标"] = gt
            res[j] = s
            j += 1

    res.fillna(0, inplace=True)
    return res.T


if __name__ == "__main__":
    # source_file = '../queryResult_2019-09-10_145030_zs341.xlsx'
    source_file = '../queryResult_2019-09-10_145030.npy'
    gt_file = "../source_data/ZS321 - 环境问题指数.xlsx"

    df2 = convert_to_new_dataframe(source_file, gt_file, write_path='../tmp_zs321')
    df2.to_excel('../zs321_20190923.xlsx')

    # regression
    regression_test('../zs321_20190923.xlsx')
