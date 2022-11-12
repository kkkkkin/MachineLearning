import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle as pkl
from math import sqrt
import datetime
import time
import copy
import warnings

warnings.filterwarnings("ignore")

type2eng = {'旅游景点;公园': 'Attraction;Park', '教育培训;高等院校': 'Education;School', '购物;购物中心': 'Shopping;Mall',
            '医疗;综合医院': 'Medical;Hospital', '运动健身;体育场馆': 'Sport;Stadium', '旅游景点;文物古迹': 'Attraction;Monument',
            '旅游景点;风景区': 'Attraction;Scenic', '交通设施;火车站': 'Transportation;Train', '交通设施;长途汽车站': 'Transportation;Bus',
            '旅游景点;植物园': 'Attraction;Arboretum', '旅游景点;游乐园': 'Attraction;Amusement', '旅游景点;水族馆': 'Attraction;Aquarium',
            '旅游景点;动物园': 'Attraction;Zoo', '交通设施;飞机场': 'Transportation;Airport'}

# %% md

## load data

# %%

# 997 areas data
area_info = pd.read_csv('./data/area_passenger_info.csv',
                        names=['id', 'name', 'type', 'center_x', 'center_y', 'grid_x', 'grid_y', 'area'])
area_flow = pd.read_csv('./data/area_passenger_index.csv', names=['id', 'date_hour', 'flow'])
area_info['type'] = area_info[['type']].apply(lambda x: type2eng[x['type']], axis=1)
area_info['type1'], area_info['type2'] = area_info['type'].str.split(';', 1).str
area_info.drop(['type'], axis=1)
area_info.head()


# %% md

## 1. merge area_info and area_flow

# %%

def merge_area_info_flow(df, dates):
    df = copy.deepcopy(df)
    cols = []
    for date in dates:
        date_hour = int(date.strftime("%Y%m%d%H"))
        col = date.strftime("%m%d%H")
        cols.append(col)
        flows = area_flow[area_flow['date_hour'] == date_hour]['flow'].tolist()
        df[col] = flows
    return df, cols


def hour_normalize(row, cols):
    res = []
    for x in cols:
        res.append(row[x] / row[x[:4]] if row[x[:4]] > 0 else 0)
    return pd.Series(res)


def add_mean_cols(df, dates):
    df = copy.deepcopy(df)
    hour_cols, day_cols = [], []

    for date in dates:
        day = date.strftime("%m%d")
        day_cols.append(day)
        cols = [x for x in date_cols if x[:4] == day]
        df[day] = df[cols].mean(axis=1)

    for hour in range(24):
        hour = str(hour).zfill(2)
        hour_cols += [hour + '0', hour + '1', hour + '2']

        cols0 = [x for x in date_cols[-6 * 24:-24] if x[-2:] == hour]
        df[hour + '0'] = df.apply(hour_normalize, cols=cols0, axis=1).mean(axis=1)
        cols1 = [x for x in date_cols[-7 * 24:-6 * 24] + date_cols[-24:] if x[-2:] == hour]
        df[hour + '1'] = df.apply(hour_normalize, cols=cols1, axis=1).mean(axis=1)
        cols2 = [x for x in date_cols[-7 * 24:] if x[-2:] == hour]

    return df, hour_cols, day_cols


# %%

# start = datetime.datetime(2020,1,26,0)
# stop = datetime.datetime(2020,2,9,0)
start = datetime.datetime(2020, 2, 2, 0)
stop = datetime.datetime(2020, 2, 16, 0)

dates = [start + datetime.timedelta(hours=x) for x in range((stop - start).days * 24)]
area_df, date_cols = merge_area_info_flow(area_info, dates)

dates = [start + datetime.timedelta(days=x) for x in range((stop - start).days)]
area_df, hour_cols, day_cols = add_mean_cols(area_df, dates)

# %% md

## 2. hour-aware growth index

# %%

'''
 define three level for growth index
'''
hour_list_0 = ['00', '01', '02', '03', '04', '05', '06', '23']
hour_list_1 = ['07', '08', '09', '19', '20', '21', '22']
hour_list_2 = ['10', '11', '12', '13', '14', '15', '16', '17', '18']
cols_0_0 = [x for x in date_cols[-7 * 24:] if x[-2:] in hour_list_0]
cols_0_1 = [x for x in date_cols[:-7 * 24] if x[-2:] in hour_list_0]
cols_1_0 = [x for x in date_cols[-7 * 24:] if x[-2:] in hour_list_1]
cols_1_1 = [x for x in date_cols[:-7 * 24] if x[-2:] in hour_list_1]
cols_2_0 = [x for x in date_cols[-7 * 24:] if x[-2:] in hour_list_2]
cols_2_1 = [x for x in date_cols[:-7 * 24] if x[-2:] in hour_list_2]
growth_dict = {k: 'growth_0' for k in hour_list_0}
growth_dict.update({k: 'growth_1' for k in hour_list_1})
growth_dict.update({k: 'growth_2' for k in hour_list_2})


# %% md

## 3. Rule application

# %%

def rule_predict(row):
    k, res = 0, []
    base = row['base_2']

    alpha_list = row['alpha']
    if alpha_list[0] > 0:
        base = 0.45 * base / alpha_list[0] + 0.55 * row['base_1']

    for alpha in alpha_list:

        if k == 0 or k == 6:
            hour_colsx = [x for x in hour_cols if x[-1] == '1']
        else:
            hour_colsx = [x for x in hour_cols if x[-1] == '0']
        k += 1

        for hc in hour_colsx[:8]:
            beta = row[hc]
            res.append(row[growth_dict[hc[:2]]] * base * sqrt(alpha) * beta)

        for hc in hour_colsx[8:]:
            beta = row[hc]
            if k == 6 and alpha < 1.02:
                beta = 1.02 * beta
            if k == 7 and alpha < 1.02:
                beta = 1.02 * beta
            if k == 6 and alpha < 1:
                res.append(row[growth_dict[hc[:2]]] * base * sqrt(alpha) * beta)
            elif k == 7 and alpha < 1:
                res.append(row[growth_dict[hc[:2]]] * base * sqrt(alpha) * beta)
            else:
                res.append(row[growth_dict[hc[:2]]] * base * alpha * beta)

    return res


def merge_alpha(row, w=[0.5, 0.5]):
    res = []
    for x, y in zip(row['week0_alpha'], row['week1_alpha']):
        res.append(x * w[0] + y * w[1])
    return res


# %%

# recent two weeks
for i in range(2):
    one_week = day_cols[i * 7:(i + 1) * 7]
    area_df['week' + str(i) + '_mean'] = area_df[one_week].mean(axis=1)
    area_df['week' + str(i) + '_alpha'] = area_df.apply(
        lambda x: [x[j] / x['week' + str(i) + '_mean'] for j in one_week], axis=1)

w = [0, 1.0]
# 3-1 alpha computing
area_df['alpha'] = area_df.apply(merge_alpha, w=w, axis=1)

# 3-2 beta computing
hour_cols0 = [x for x in hour_cols if x[-1] == '0']
area_df[hour_cols0] = area_df.apply(lambda x: x[hour_cols0] / sum(x[hour_cols0]) if sum(x[hour_cols0]) > 0 else 0,
                                    axis=1)
hour_cols1 = [x for x in hour_cols if x[-1] == '1']
area_df[hour_cols1] = area_df.apply(lambda x: x[hour_cols1] / sum(x[hour_cols1]) if sum(x[hour_cols1]) > 0 else 0,
                                    axis=1)

recent_num = 3
# 3-3 base computing
with open('./cache/base_lgb.pkl', 'rb') as f:
    day_flow_lgb = pkl.load(f)
area_df['base_1'] = area_df.apply(
    lambda x: sum([x[day_cols[-i]] / x['alpha'][-i] if x['alpha'][-i] > 0 else 0 for i in range(1, recent_num + 1)]),
    axis=1)
area_df['base_1'] = area_df['base_1'] * 24 / recent_num
area_df['base_2'] = day_flow_lgb * 24

# 3-4 growth computing
area_df['mean_0_0'] = area_df[cols_0_0].mean(axis=1)
area_df['mean_0_1'] = area_df[cols_0_1].mean(axis=1)
area_df['mean_1_0'] = area_df[cols_1_0].mean(axis=1)
area_df['mean_1_1'] = area_df[cols_1_1].mean(axis=1)
area_df['mean_2_0'] = area_df[cols_2_0].mean(axis=1)
area_df['mean_2_1'] = area_df[cols_2_1].mean(axis=1)
area_df['growth_0'] = np.power(1 + (area_df['mean_0_0'] - area_df['mean_0_1']) / area_df['mean_0_1'], 0.5).replace(
    [np.inf], np.nan).fillna(1)
area_df['growth_1'] = np.power(1 + (area_df['mean_1_0'] - area_df['mean_1_1']) / area_df['mean_1_1'], 0.5).replace(
    [np.inf], np.nan).fillna(1)
area_df['growth_2'] = np.power(1 + (area_df['mean_2_0'] - area_df['mean_2_1']) / area_df['mean_2_1'], 0.5).replace(
    [np.inf], np.nan).fillna(1)

gi_dict1 = dict(area_df[area_df['week1_mean'] > 10].groupby(['type1']).mean()['growth_0'])
gi_dict2 = dict(area_df[area_df['week1_mean'] > 0].groupby(['type2']).mean()['growth_0'])
area_df['growth_0_type1'] = area_df.apply(lambda x: gi_dict1[x['type1']] if x['type1'] in gi_dict1 else x['growth_0'],
                                          axis=1)
area_df['growth_0_type2'] = area_df.apply(lambda x: gi_dict2[x['type2']] if x['type2'] in gi_dict2 else x['growth_0'],
                                          axis=1)
area_df['growth_0'] = 0.3 * area_df['growth_0_type1'] + 0.4 * area_df['growth_0_type2'] + 0.3 * area_df['growth_0']

gi_dict1 = dict(area_df[area_df['week1_mean'] > 10].groupby(['type1']).mean()['growth_1'])
gi_dict2 = dict(area_df[area_df['week1_mean'] > 0].groupby(['type2']).mean()['growth_1'])
area_df['growth_1_type1'] = area_df.apply(lambda x: gi_dict1[x['type1']] if x['type1'] in gi_dict1 else x['growth_1'],
                                          axis=1)
area_df['growth_1_type2'] = area_df.apply(lambda x: gi_dict2[x['type2']] if x['type2'] in gi_dict2 else x['growth_1'],
                                          axis=1)
area_df['growth_1'] = 0.3 * area_df['growth_1_type1'] + 0.4 * area_df['growth_1_type2'] + 0.3 * area_df['growth_1']

gi_dict1 = dict(area_df[area_df['week1_mean'] > 10].groupby(['type1']).mean()['growth_2'])
gi_dict2 = dict(area_df[area_df['week1_mean'] > 0].groupby(['type2']).mean()['growth_2'])
area_df['growth_2_type1'] = area_df.apply(lambda x: gi_dict1[x['type1']] if x['type1'] in gi_dict1 else x['growth_2'],
                                          axis=1)
area_df['growth_2_type2'] = area_df.apply(lambda x: gi_dict2[x['type2']] if x['type2'] in gi_dict2 else x['growth_2'],
                                          axis=1)
area_df['growth_2'] = 0.3 * area_df['growth_2_type1'] + 0.4 * area_df['growth_2_type2'] + 0.3 * area_df['growth_2']

area_df['growth_index'] = np.power(1 + (area_df['week1_mean'] - area_df['week0_mean']) / area_df['week0_mean'], 0.5)
gi_dict1 = dict(area_df.groupby(['type1']).mean()['growth_index'])
gi_dict2 = dict(area_df.groupby(['type2']).mean()['growth_index'])
area_df['growth_index_type1'] = area_df.apply(
    lambda x: gi_dict1[x['type1']] if x['type1'] in gi_dict1 else x['growth_index'], axis=1)
area_df['growth_index_type2'] = area_df.apply(
    lambda x: gi_dict2[x['type2']] if x['type2'] in gi_dict2 else x['growth_index'], axis=1)
area_df['growth_index'] = 0.3 * area_df['growth_index_type1'] + 0.4 * area_df['growth_index_type2'] + 0.3 * area_df[
    'growth_index']


# %%

def new_hour_list(a):
    a = np.array(list(map(int, a)))
    a = np.hstack([a, a + 24])
    return a


# 3-5 apply rule model
pred_flow_rule = area_df.apply(rule_predict, axis=1)
pred_flow_rule = np.array([x for x in pred_flow_rule])

hour_0 = new_hour_list(hour_list_0)
hour_1 = new_hour_list(hour_list_1)
hour_2 = new_hour_list(hour_list_2)

growth = np.ones((997, 24 * 2))
growth[:, hour_0] = np.tile(np.array(area_df['growth_0']).reshape(-1, 1), hour_0.shape)
growth[:, hour_1] = np.tile(np.array(area_df['growth_1']).reshape(-1, 1), hour_1.shape)
growth[:, hour_2] = np.tile(np.array(area_df['growth_2']).reshape(-1, 1), hour_2.shape)

add_2days = pred_flow_rule[:, :24 * 2] * growth
pred_flow_rule = np.concatenate((pred_flow_rule, add_2days), axis=-1).flatten()
pred_flow_rule.shape


# 3-6 save
with open('./cache/pred_flow_rule.pkl', 'wb') as f:
    pkl.dump(pred_flow_rule, f)
with open('./cache/growth_index.pkl', 'wb') as f:
    pkl.dump(area_df['growth_index'], f)
