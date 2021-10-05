import pandas as pd
import csv
import numpy as np
import os
import torch
code_num = 758
trade_day = 1702
fts = 10
def load_EOD_data():
    f = open(r'../data/result(758)_label.csv')
    df = pd.read_csv(f, header=None)
    data = df.iloc[:, 0:10].values
    eod_data = data.reshape(code_num, trade_day, fts)
    data_label = df.iloc[:, 3] # 将取label的列转为取收盘价！第4列
    # 数据集为：开盘价，最高，最低，收盘价，成交量，成交额；还有5日，10日，20日，30日均线
    data_label = round(data_label / data_label.shift(1), 4).values  # 获取价格相对向量yt
    print('data_label_2', )  # data_label
    ground_truth = data_label.reshape(code_num, trade_day)  # 获取价格相对向量yt

    return eod_data, ground_truth

def get_batch(eod_data, gt_data, offset, seq_len):

    return eod_data[:, offset:offset + seq_len, :], \
           gt_data[:, offset + seq_len]


def get_industry_adj():
    adj = np.mat(pd.read_csv(open(r'../data/industry_adj(758).csv'), header=None))
    return adj
