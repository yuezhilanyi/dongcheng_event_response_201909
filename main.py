import os
import tqdm

import pandas as pd
import numpy as np
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
    df = df[['案件号', '问题来源', '问题类型', '大类名称', '街道', '上报时间', '当前阶段', '处置截止时间', '处置结束时间',
             '结案时间', '立案时间', '强制结案时间']]

    # 时间类
    df["立案耗时"] = df["立案时间"].subtract(df["上报时间"])
    df["处置预计耗时"] = df["处置截止时间"].subtract(df["立案时间"])
    # 时间转分钟

    # https://blog.csdn.net/liudinglong1989/article/details/78728683
    def f(x): return int(x / timedelta(minutes=1)) if isinstance(x, timedelta) else 0

    df["立案耗时"] = df["立案耗时"].apply(f)
    df["处置预计耗时"] = df["处置预计耗时"].apply(f)

    # 立案耗时加权
    bins = [0, 60, 360, 1440, 4320, 999999]  # minutes [0-1h, 1h-6h, 6h-1d, 1d-3d, >3d]
    # bins = [timedelta(hours=0), timedelta(hours=1), timedelta(hours=6),
    #         timedelta(hours=24), timedelta(hours=72), timedelta(days=365)]  # minutes [0-1h, 1h-6h, 6h-1d, 1d-3d, >3d]
    df["W立案"] = pd.cut(df["立案耗时"], bins, labels=[1, 0, -1, -2, -3])
    # 强结案加权
    df["W强结"] = -pd.isnull(df["强制结案时间"])

    return df


def convert_to_new_dataframe(srs_df):
    # # add some fields
    srs_df["结案日期"] = srs_df["结案时间"].dt.date
    df_gt = pd.read_excel("../ZS222.xlsx")
    df_gt.set_index(df_gt["日期"], inplace=True)

    # new dataframe
    index = srs_df["结案日期"].value_counts().index
    index = sorted(index)

    def get_gt(dataframe_gt, somewhere, someday):
        iso_someday = int(someday.isoformat().replace('-', ''))
        dataframe_gt = dataframe_gt.xs(iso_someday)
        gt = dataframe_gt[somewhere]
        return gt

    def new_df_until_someday(dataframe, somewhere, someday):
        deadline = pd.Timestamp(datetime.combine(someday, datetime.max.time()))
        # F1 - 某地所有案件
        dataframe = dataframe[dataframe["街道"] == somewhere]

        # F2.1 - 监督员自行处理的案件
        dataframe_self = dataframe[dataframe["问题来源"] == "监督员自行处理"]
        dataframe_self = dataframe_self[dataframe["上报时间"].dt.date == someday]  # 某日自行处理案件, 默认为当天完成(结案时间不正确)
        # F2.2 - 剩余案件 (均具有处置截止时间, 处置结束时间, 结案时间, 立案时间等信息)
        dataframe = dataframe[dataframe["问题来源"] != "监督员自行处理"]

        # F3 (<-F2.2) - 考核日之前的所有案件
        dataframe = dataframe[dataframe["上报时间"] <= deadline]

        basename = int(deadline.isoformat().replace('-', '')[:8])
        dataframe.to_excel('../tmp/{}_{}.xlsx'.format(somewhere, basename))

        return dataframe, dataframe_self

    # for i in tqdm.trange(len(index)):
    lst = []
    for i in tqdm.trange(30):
        day = index[i]
        day_start = pd.Timestamp(datetime.combine(day, datetime.min.time()))
        day_end = pd.Timestamp(datetime.combine(day, datetime.max.time()))
        assert day is not None

        def func_less_than_day_end(x): return day_end if x > day_end else x

        def func_dt2mins(x): return int(x / timedelta(minutes=1)) if isinstance(x, timedelta) else 0

        def func_less_than_day_start(x): return day_start if x < day_start else x  # 取立案时间和零点较晚的

        # for area in df["街道"].value_counts().index:
        for area in ["东华门", "景山", "交道口", "安定门", "北新桥", "东四", "朝阳门", "建国门", "东直门", "和平里",
                     "前门", "崇外", "东花市", "龙潭", "体育馆", "天坛", "永定门外"]:  # 和网格代码顺序一致, 方便后续观察对比
            # tqdm.tqdm.write(area)
            df, df_self = new_df_until_someday(srs_df, area, day)
            df = df[-(df["处置结束时间"] < pd.Timestamp(day))]

            # 如果 dataframe 为空, 跳至下一个
            if len(df) == 0:
                lst.append([day, area, 0, len(df_self), 0, 0, 0, 0])
                continue

            # TODO: 漏洞 - 2017东城管字537978

            # 非考核日之前结案, 说明考核日当天未结案, 替换处置结束时间为当天24点
            df["处置结束时间"] = df["处置结束时间"].apply(func_less_than_day_end)

            # 截至当天, 处置实际耗时总计
            df["处置实际耗时"] = df["处置结束时间"].subtract(df["立案时间"])
            df["处置实际耗时"] = df["处置实际耗时"].apply(func_dt2mins)
            df["处置实际耗时"] = np.where(df["处置结束时间"] < df["立案时间"], timedelta(days=0) / timedelta(minutes=1), df["处置实际耗时"])
            # 处置耗时百分比
            df["处置耗时百分比"] = df["处置实际耗时"] / df["处置预计耗时"] * 100

            # 当天处置实际耗时
            df["当天案件开始时间"] = df["立案时间"].apply(func_less_than_day_start)
            df["当天处置实际耗时"] = df["处置结束时间"].subtract(df["当天案件开始时间"])
            df["当天处置实际耗时"] = df["当天处置实际耗时"].apply(func_dt2mins)
            df["当天处置实际耗时"] = np.where(df["处置结束时间"] < df["当天案件开始时间"], timedelta(days=0) / timedelta(minutes=1),
                                      df["当天处置实际耗时"])
            # 1439 -> 1440
            def func_1439to1440(x): return 1440 if x == 1439 else x
            df["当天处置实际耗时"] = df["当天处置实际耗时"].apply(func_1439to1440)
            # # 计划内耗时
            df["处置截止时间"] = df["处置截止时间"].apply(func_less_than_day_end)
            df["当天计划内耗时"] = df["处置截止时间"].subtract(df["当天案件开始时间"])

            def f4(x):
                if x <= timedelta(days=0):
                    return timedelta(days=0)
                elif timedelta(days=0) <= x <= timedelta(days=1):
                    return x
                elif timedelta(days=1) < x:
                    return timedelta(days=1)
                else:
                    return timedelta(days=0)
            df["当天计划内耗时"] = df["当天计划内耗时"].apply(f4)
            df["当天计划内耗时"] = df["当天计划内耗时"].apply(func_dt2mins)
            # http://www.it1352.com/605416.html
            df["当天计划内耗时"] = np.where(df["处置截止时间"] > df["处置结束时间"], df["当天处置实际耗时"], df["当天计划内耗时"])
            df["当天计划内耗时"] = df["当天计划内耗时"].apply(func_1439to1440)
            # # 计划外耗时
            df["当天计划外耗时"] = df["当天处置实际耗时"].subtract(df["当天计划内耗时"])

            iso_someday = int(day_end.isoformat().replace('-', '')[:8])
            try:
                df.to_excel('../tmp2/{}_{}.xlsx'.format(area, iso_someday))
            except PermissionError as e:
                print(e)

            # 开始统计
            n1 = len(df_self)  # 自行处理的数量
            n2 = len(df)  # 非自行处理的数量
            n3 = df["W强结"].sum()  # 强结数量
            c1 = df["立案耗时"].sum()  # 立案耗时总长(分钟)
            c2 = df["当天计划内耗时"].sum()  # 计划内耗时总长(分钟)
            c3 = df["当天计划外耗时"].sum()  # 计划外耗时总长(分钟)
            gt = get_gt(df_gt, area, day)

            lst.append([day, area, n1, n2, n3, c1, c2, c3, gt])

    res = pd.DataFrame(lst, columns=["日期", "街道",
                                     "自行处理案件总数", "其他案件总数", "强制结案总数",
                                     "立案耗时总长(分钟)", "计划内耗时总长(分钟)", "计划外耗时总长(分钟)",
                                     "原评分"])

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
    # source_file = '../event_data.npy'
    # df1 = read_source(source_file)

    # df2 = convert_to_new_dataframe(df1)
    # df2.to_excel('../ndf20190919.xlsx')

    df3_file_path = '../ndf20190919.xlsx'
    df3 = cal_index(df3_file_path)
    df3.to_excel(df3_file_path)

