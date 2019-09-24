"""
四级指数编号 - ZS322
四级指数 - 废弃物指数
指数说明 - 废弃家具、施工废料、生活垃圾等环境案件的分布情况
计算方式 - 网格内废弃家具、施工废料、生活垃圾等环境案件的数量
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
    df = df[df['当前阶段'] != '[作废]']  # 根据2019.9.18与网格中心考评处的沟通, 不考虑作废案件

    # keep only useful columns, to reduce too many dimensions
    df = df[['问题类型', '大类名称', '小类名称', '小类明细', '微类名称',
             '街道', '上报时间', '当前阶段', '处置截止时间', '处置结束时间']]

    # 二次筛选
    # "垃圾不落地"(大类名称), "垃圾分类"(大类名称) 中不包含废弃物
    attr_filter = "施工废弃料|废弃家具|生活垃圾|施工废料"
    df = df.loc[
        df["大类名称"].str.contains(attr_filter, na=False) |
        df["小类名称"].str.contains(attr_filter, na=False) |
        df["小类明细"].str.contains(attr_filter, na=False)
    ]

    return df


def convert_to_new_dataframe(srs_path, gt_path, write_path=''):
    # 读取文件
    srs_df = dataframe_preprocess(srs_path)
    srs_df["上报日期"] = srs_df["上报时间"].dt.date
    df_gt = pd.read_excel(gt_path)
    df_gt.set_index(df_gt["日期"], inplace=True)

    index = srs_df["上报日期"].value_counts().index
    index = sorted(index)

    lst = []
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

            # 过滤字符串
            f1 = "施工废弃料|施工废料"
            f2 = "废弃家具"
            f3 = "生活垃圾"

            # 截至统计日, 所有未完成的案件
            def func_count(str_filter):
                return len(df.loc[df["大类名称"].str.contains(str_filter, na=False) |
                                  df["小类名称"].str.contains(str_filter, na=False) |
                                  df["小类明细"].str.contains(str_filter, na=False)])
            n1 = len(df)  # 案件总数
            n1a = func_count(f1)  # "施工废弃料|施工废料"
            n1b = func_count(f2)  # "废弃家具"
            n1c = func_count(f3)  # "生活垃圾"
            # 统计日当日, 未完成的案件
            df = df[df["上报时间"].dt.date == day]  # 当天案件总数
            n2 = len(df[df["上报时间"].dt.date == day])  # 当天案件总数
            n2a = func_count(f1)  # "施工废弃料|施工废料"
            n2b = func_count(f2)  # "废弃家具"
            n2c = func_count(f3)  # "生活垃圾"

            lst.append([day, area, n1, n1a, n1b, n1c, n2, n2a, n2b, n2c, gt])

    res = pd.DataFrame(lst, columns=["日期", "街道",
                                     "案件总数", "施工废弃料(总计)", "废弃家具(总计)", "生活垃圾(总计)",
                                     "当天案件总数", "施工废弃料(当日)", "废弃家具(当日)", "生活垃圾(当日)", "原指标"])

    return res


if __name__ == "__main__":
    # source_file = '../queryResult_2019-09-10_145030_zs341.xlsx'
    source_file = '../queryResult_2019-09-10_145030.npy'
    gt_file = "../source_data/ZS322 - 废弃物指数.xlsx"

    df2 = convert_to_new_dataframe(source_file, gt_file, write_path='../tmp_zs322')
    df2.to_excel('../zs322_20190923.xlsx')

    # regression
    df2 = pd.read_excel('../zs322_20190923.xlsx')
    # df2 = df2[df2["原指标"] != 0]
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
