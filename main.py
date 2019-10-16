"""
主程序，传参确定需要计算的数据文件路径以及指数。
尽量保持各子函数的结构不变。
"""
import sys
import argparse
import importlib

import pandas as pd

import zs222
import zs232
import zs321
import zs322
import utils

INDEX_GT_PATHS = {
    "zs222": "../source_data/ZS222 - 处置效能指数.xlsx",
    "zs232": "../source_data/ZS232 - 服务效能指数.xlsx",
    "zs321": "../source_data/ZS321 - 环境问题指数.xlsx",
    "zs322": "../source_data/ZS322 - 废弃物指数.xlsx",
    "zs341": "../source_data/ZS341 - 服务需求指数.xlsx",
    "zs342": "../source_data/ZS342 - 矛盾纠纷指数.xlsx",
}

MODEL_PATHS = {
    "zs222": "./models/zs222.sav",
    "zs232": "./models/zs232.sav",
    "zs321": "./models/zs321_poly.sav",
    "zs322": "./models/zs322_poly.sav",
    "zs341": "./models/zs341.sav",
    "zs342": "./models/zs342.sav",
}

"""
zs222: 生成新指数，生成后缩放至原范围（缩放参数由全年数据计算得出）
zs232: 生成新指数，生成后缩放至原范围（缩放参数由全年数据计算得出）
zs321: 调用全年数据训练的模型进行计算，然后缩放至原范围（缩放参数由全年数据计算得出）
zs322: 调用全年数据训练的模型进行计算，然后缩放至原范围（缩放参数由全年数据计算得出）
zs341: 生成新指数，生成后缩放至原范围（缩放参数由全年数据计算得出）
zs342: 生成新指数，生成后缩放至原范围（缩放参数由全年数据计算得出）
"""


class Processing(object):
    def __init__(self, source_file_path, write_daily_data_to_disk=True):
        self.source_file_path = source_file_path
        self.write_daily_data_to_disk = write_daily_data_to_disk
        self.allowed_ics = ['zs222', 'zs232', 'zs321', 'zs322', 'zs341', 'zs342']

    def cal_zs222(self):
        """

        :return: pd.DataFrame (日期，街道，评分)
        """
        ic = 'zs222'
        source_file = self.source_file_path
        write_path = '../tmp_{}'.format(ic) if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS[ic]

        df = zs222.convert_to_new_dataframe(source_file, gt_file, write_path=write_path)

        df2_file_path = '../{}_20190923.xlsx'.format(ic)
        df2 = zs222.cal_index(df2_file_path)
        df2 = df2[["日期", "街道", "新评分"]]
        df2.to_excel(df2_file_path)

        return df2

    def cal_zs232(self):
        """

        :return: pd.DataFrame (日期，街道，评分)
        """
        ic = "zs232"
        source_file = self.source_file_path
        write_path = '../tmp_zs232' if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS["zs232"]

        df2 = zs232.convert_to_new_dataframe(source_file, gt_file, write_path=write_path)
        df2.to_excel('../zs232_20190923.xlsx')

        df3_file_path = '../zs232_20190923.xlsx'
        df3 = zs232.cal_index(df3_file_path)
        df3 = df3[["日期", "街道", "新评分"]]
        df3.to_excel(df3_file_path)

        return df3

    def cal_zs321(self):
        """

        :return: pd.Series
        """
        source_file = self.source_file_path
        write_path = '../tmp_zs321' if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS["zs321"]  # TODO: 新数据没有历史评分

        df2 = zs321.convert_to_new_dataframe(source_file, gt_file, write_path=write_path)

        # cal
        X = df2.drop(["Unnamed: 0", "日期", "街道", "原指标"], axis=1)
        y = utils.model_predict(X, MODEL_PATHS['zs321'])

        df3 = df2.assign(score=y)
        df3 = df3.loc[["日期", "街道", 'score']]
        df3 = df3.rename(columns={"score": "新评分"})

        df3.to_excel('../zs321_20190923.xlsx')

        return df3

    def cal_zs322(self):
        source_file = self.source_file_path
        write_path = '../tmp_zs322' if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS["zs322"]

        df2 = zs322.convert_to_new_dataframe(source_file, gt_file, write_path=write_path)
        df2.to_excel('../zs322_20190923.xlsx')

        # cal
        X = df2.drop(["Unnamed: 0", "日期", "街道", "原指标"], axis=1)
        y = utils.model_predict(X, MODEL_PATHS['zs322'])

        return y

    def cal_zs34x(self, ic=None):
        """

        :param ic: index code, zs341 or zs342
        :return:  pd.DataFrame
        """
        daily_data_write_path = '../tmp_{}'.format(ic) if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS[ic]
        summary_data_write_path = './output/{}.xlsx'.format(ic)

        # TODO: 将两个 convert_to_new_dataframe 合并为一个
        zs34x = importlib.import_module(ic)
        df = zs34x.convert_to_new_dataframe(self.source_file_path, gt_file, write_path=daily_data_write_path)

        # cal
        df = df.drop(["('日期', '')", "('街道', '')", "('原指标', '')"], axis=1)
        y = utils.model_predict(df, MODEL_PATHS[ic])

        df = df.assign(score=y)
        df.to_excel(summary_data_write_path)

        return df

    def cal_x(self, ic):
        """

        :param ic:  index code
        :return:  pd.Series, pd.DataFrame
        """
        print("该功能暂时只用于测试!!!")
        assert ic in self.allowed_ics, "Mode must be in {}, not {}".format(self.allowed_ics, ic)
        mod = importlib.import_module(ic)

        daily_data_write_path = '../tmp_{}'.format(ic) if self.write_daily_data_to_disk else ''
        gt_file = INDEX_GT_PATHS[ic]
        summary_data_write_path = './output/{}.xlsx'.format(ic)

        # TODO: 将两个 convert_to_new_dataframe 合并为一个
        df = mod.convert_to_new_dataframe(self.source_file_path, gt_file, write_path=daily_data_write_path)

        if ic in ['zs321', 'zs322']:
            # cal
            try:
                df = df.drop(["日期", "街道", "原指标"], axis=1)
            except KeyError:
                df = df.drop(["('日期', '')", "('街道', '')", "('原指标', '')"], axis=1)
            # TODO: complete model file
            y = utils.model_predict(df, MODEL_PATHS[ic])
        else:
            if ic in ['zs341', 'zs342']:
                """
                基础分100， 按期完成 +10 / d， 延期完成 -5 /d
                """
                w1 = 10
                w2 = -5
                y = 100 + \
                    df[[c for c in df if '按时完成' in c]].iloc[:, 0] * w1 + \
                    df[[c for c in df if '延期完成' in c]].iloc[:, 0] * w2
            elif ic in ['zs222', 'zs232']:
                """
                根据"强制结案总数", "计划内耗时总长(分钟)", "计划外耗时总长(分钟)"三个指标得到基础分;
                根据"自行处理案件总数", 奖励一定分数;
                根据"强制结案总数", 扣除一定分数;
                "其他案件总数"不作为打分依据(一定程度上,已经在"计划内(外)耗时总长中得到体现);
                "立案耗时总长(分钟)"不作为打分依据, 因本指标为"处置效能指数", 不牵涉从上报到立案
                """
                # TODO: 增加当天内完成案件的权重(比如自行处理案件因当天完成, 相比第二天完成的案件, 少了一晚上的执行分数)
                w1 = 60  # 自行处理案件总数, 认为w1分钟为完成一个自行处理案件所需的平均时间
                w2 = 0  # 其他案件总数, 不考评
                w3 = 0  # 强制结案总数, 暂不考评(没有好的思路, 且强制结案会同时生成一个新的案件)
                w4 = 0  # 立案耗时总长(分钟), 不考评
                w5 = 1  # 计划内耗时总长(分钟)
                w6 = -1  # 计划外耗时总长(分钟)
                y = (df["自行处理案件总数"] * w1 +
                     df["其他案件总数"] * w2 +
                     df["强制结案总数"] * w3 +
                     df["立案耗时总长(分钟)"] * w4 +
                     df["计划内耗时总长(分钟)"] * w5 +
                     df["计划外耗时总长(分钟)"] * w6
                     ) / 1000
            else:
                raise IndexError("不可能出现的错误。")

        df = df.assign(score=y)
        df.to_excel(summary_data_write_path)
        return y, df

    def cal_all(self, save_path):
        # new dataframe
        df = pd.DataFrame(columns=self.allowed_ics)
        # calculate
        for ic in self.allowed_ics:
            df[ic], _ = self.cal_x(ic)
        # save to disk
        df.to_excel(save_path)
        return df


def main(args):
    p = Processing(args.source_file_path, write_daily_data_to_disk=True)
    if args.mode == 'all':
        p.cal_all(args.output_path)
    else:
        p.cal_x(args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file_path', type=str, help='source file to be processed')
    parser.add_argument('output_path', type=str, help='result file to be saved')
    parser.add_argument('--mode', choices=['all', 'zs222', 'zs232', 'zs321', 'zs322', 'zs341', 'zs342'],
                        default='all')
    args = parser.parse_args()

    main(args)
