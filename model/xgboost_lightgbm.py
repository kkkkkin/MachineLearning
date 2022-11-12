import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle as pkl
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
area_info['type'] = area_info[['type']].apply(lambda x: type2eng[x['type']], axis=1)
area_flow = pd.read_csv('./data/area_passenger_index.csv', names=['id', 'date_hour', 'flow'])
weather = pd.read_csv('./data/weather_data.csv', names=['date', 'weekday', 'high', 'low', 'weather'])

# %%

area_flow['date_hour'] = pd.to_datetime(area_flow['date_hour'], format='%Y%m%d%H')
area_flow['day'] = area_flow['date_hour'].dt.day.apply(lambda x: x - 17 if x >= 17 else x + 14)
area_flow['hour'] = area_flow['date_hour'].dt.hour
area_flow.head()

# %%

dict_wea = {'中雪': 2, '多云': 0, '小雨': 2, '小雪': 2, '晴': 0, '暴雪': 2, '阴': 1, '雨夹雪': 2, '雾': 1, '霾': 1}

weather['weather'] = weather['weather'].apply(lambda x: x if '~' in x else x + '~' + x)
weather['wea1'], weather['wea2'] = weather['weather'].str.split('~', 1).str
weather_df = weather.drop(['weather'], axis=1)
weather_df['wea1_encoded'] = weather_df['wea1'].map(dict_wea)
weather_df['wea2_encoded'] = weather_df['wea2'].map(dict_wea)
weather_df['wea_encoded'] = weather_df.apply(lambda x: max(x['wea1_encoded'], x['wea2_encoded']), axis=1)
weather_df['day'] = range(weather_df.shape[0])
weather_df['weekend'] = weather_df['weekday'].apply(lambda x: 1 if x > 5 else 0)

weather_df.drop(['date', 'wea2_encoded', 'wea1_encoded', 'wea1', 'wea2'], axis=1, inplace=True)
weather_df.head()


# %% md

## tree model: xgoost & lightgbm

# %%

def training_xgboost(train_data, train_y):
    model_xgb = xgb.XGBRegressor(max_depth=5
                                 , learning_rate=0.14
                                 , n_estimators=2000
                                 , n_jobs=-1)

    t1 = time.time()
    model_xgb.fit(train_data, train_y)
    print('training time:', str(int(time.time() - t1)) + 's, ', end='')
    return model_xgb


def training_lightgbm(train_data, train_y):
    model_lgb = lgb.LGBMRegressor(num_leaves=20
                                  , max_depth=5
                                  , learning_rate=0.14
                                  , n_estimators=2000
                                  , n_jobs=-1)

    t1 = time.time()
    model_lgb.fit(train_data, train_y)
    print('training time:', str(int(time.time() - t1)) + 's, ', end='')
    return model_lgb


# %% md

## feature extraction

# %%

# 时序特征提取
def feature_extraction(flow_data_in, split_day, area_embed=[], weekday=False, model='xgboost'):
    flow_data_in = pd.merge(flow_data_in, weather_df, on='day')
    flow_data_in = pd.merge(flow_data_in, area_info[['id', 'area', 'type']], on='id')

    # graph embedding feature
    if len(area_embed) != 0:
        flow_data_in = pd.merge(flow_data_in, area_embed, on='id')

    # if only weekday or not
    if weekday:
        flow_data_in = flow_data_in[flow_data_in['weekend'] == 0]
        if split_day >= 30 and model == 'xgboost':
            flow_data_in = flow_data_in[flow_data_in['day'] != 28]  # notice: it's a bug !

    # graph embedding feature
    flow_data_in['flow_1db'] = [0] * 1 + flow_data_in['flow'][:-1].tolist()
    flow_data_in['flow_2db'] = [0] * 2 + flow_data_in['flow'][:-2].tolist()
    flow_data_in['flow_3db'] = [0] * 3 + flow_data_in['flow'][:-3].tolist()
    flow_data_in['flow_3dba'] = flow_data_in[['flow_1db', 'flow_2db', 'flow_3db']].mean(axis=1)

    # yesterday feature
    flow_data_in['wea_1db'] = [0] + flow_data_in['wea_encoded'][:-1].tolist()
    flow_data_in['low_1db'] = [0] + flow_data_in['low'][:-1].tolist()
    flow_data_in['week_1db'] = [0] + flow_data_in['weekend'][:-1].tolist()

    # # type & id encode
    dict_type = dict(flow_data_in[flow_data_in['day'] < split_day].groupby(['type']).mean()['flow'])
    dict_id = dict(flow_data_in[flow_data_in['day'] < split_day].groupby(['id']).mean()['flow'])
    flow_data_in['type'] = flow_data_in['type'].map(dict_type)
    flow_data_in['id'] = flow_data_in['id'].map(dict_id)

    flow_data_out = flow_data_in[flow_data_in['day'] > 15]
    flow_data_out.drop(['weekday', 'high'], axis=1, inplace=True)

    return flow_data_out


# %% md

## 1. base flow for RULE

# %%

'''
1-1 validation
'''
flow_data = area_flow.groupby(['id', 'day']).agg(['mean'])['flow'].reset_index()
flow_data = flow_data.rename(columns={'mean': 'flow'})
flow_data = feature_extraction(flow_data, 23)

valid_data = flow_data[flow_data['day'] >= 23]
train_data = flow_data[flow_data['day'] < 23]
train_y = train_data['flow']

train_data.drop(['flow', 'flow_3db', 'flow_2db'], axis=1, inplace=True)
print(train_data.shape, train_y.shape, valid_data.shape)

# %%

'''
1-1 validation
'''
day_data = valid_data[valid_data['day'] == 23]
day_y = day_data['flow']
day_data.drop(['flow', 'flow_2db', 'flow_3db'], axis=1, inplace=True)

# xgboost
model_xgb_forbase_val = training_xgboost(train_data, train_y)
day_flow = model_xgb_forbase_val.predict(day_data)
day_flow[day_flow < 0] = 0
print('xgboost validation score: ', 1 / (mean_squared_error(day_y, day_flow) ** 0.5 + 1))

# lightgbm
model_lgb_forbase_val = training_lightgbm(train_data, train_y)
day_flow = model_lgb_forbase_val.predict(day_data)
day_flow[day_flow < 0] = 0
print('lightgbm validation score: ', 1 / (mean_squared_error(day_y, day_flow) ** 0.5 + 1))

# %%

'''
1-2 prediction
'''
test_data = pd.read_csv('./data/test_submit_example.csv', names=['id', 'date', 'flow'])
test_data['date'] = pd.to_datetime(test_data['date'], format='%Y%m%d%H')
test_data['day'] = test_data['date'].dt.day.apply(lambda x: x + 14)
test_data.drop(['date'], axis=1, inplace=True)

area_flow_all = pd.concat([area_flow[['id', 'day', 'flow']], test_data])
flow_data = area_flow_all.groupby(['id', 'day']).agg(['mean'])['flow'].reset_index()
flow_data = flow_data.rename(columns={'mean': 'flow'})
flow_data = feature_extraction(flow_data, 30)

test_data = flow_data[flow_data['day'] >= 30]
train_data = flow_data[flow_data['day'] < 30]
train_y = train_data['flow']
train_data.drop(['flow', 'flow_3db', 'flow_2db'], axis=1, inplace=True)

print(train_data.shape, train_y.shape, test_data.shape)

# %%

'''
1-2 prediction
'''
day_data = test_data[test_data['day'] == 30]
day_y = day_data['flow']
day_data.drop(['flow', 'flow_2db', 'flow_3db'], axis=1, inplace=True)

# xgboost
model_xgb_forbase = training_xgboost(train_data, train_y)
day_flow_xgb = model_xgb_forbase.predict(day_data)
day_flow_xgb[day_flow_xgb < 0] = 0

# lightgbm
model_lgb_forbase = training_lightgbm(train_data, train_y)
day_flow_lgb = model_lgb_forbase.predict(day_data)
day_flow_lgb[day_flow_lgb < 0] = 0

# %%

'''
1-3 save the result of base
'''
# base_lgb.pkl 机器学习模型的基础人群密度预测结果
with open('./cache/base_lgb.pkl', 'wb') as f:
    pkl.dump(day_flow_lgb, f)


# %%


# %% md

## 2. hour-level flow prediction

# %%

def hour_level_flow_prediction(model='lightgbm', mode='valid', weekday=False):
    '''
    mode: valid or predict
    '''

    pred_res = {}
    for hour in range(24):
        with open('./cache/area_embedding_1/area_flow_hour_node2vec_embed_es16_wl5_' + str(hour) + '.pkl', 'rb') as f:
            hour_embed = pkl.load(f)
        if model == 'xgboost' or weekday == False:
            with open('./cache/area_embedding_0/area_flow_hour_embed_' + str(hour) + '.pkl', 'rb') as f:
                hour_embed = pkl.load(f)

        hour_embed = {int(k) + 1: v for k, v in hour_embed.items()}
        area_embed = pd.DataFrame.from_dict(hour_embed, orient='index').reset_index().rename(columns={'index': 'id'})

        if mode == 'valid':
            split_day = 23
            flow_data = area_flow[area_flow['hour'] == hour][['id', 'flow', 'day']]
        if mode == 'predict':
            split_day = 30
            test_data = pd.read_csv('./data/test_submit_example.csv', names=['id', 'date', 'flow'])
            test_data['date'] = pd.to_datetime(test_data['date'], format='%Y%m%d%H')
            test_data['day'] = test_data['date'].dt.day.apply(lambda x: x + 14)
            test_data['hour'] = test_data['date'].dt.hour
            test_data = test_data[test_data['hour'] == hour][['id', 'flow', 'day']]
            area_flow_1 = area_flow[area_flow['hour'] == hour][['id', 'flow', 'day']]
            flow_data = pd.concat([area_flow_1, test_data]).sort_values(['id', 'day'])

        predict_len = 5 if weekday else 7
        split_day = split_day + 1 if weekday else split_day
        flow_data = feature_extraction(flow_data, split_day, area_embed, weekday, model)

        test_data = flow_data[flow_data['day'] >= split_day]
        train_data = flow_data[flow_data['day'] < split_day]
        train_y = train_data['flow']
        train_data.drop(['flow', 'flow_3db', 'flow_2db'], axis=1, inplace=True)

        if model == 'lightgbm':
            model_tree = training_lightgbm(train_data, train_y)
        if model == 'xgboost':
            model_tree = training_xgboost(train_data, train_y)

        scores = []
        pred_flow = []
        for day in range(split_day, split_day + predict_len):
            day_data = test_data[test_data['day'] == day]
            if len(day_data) == 0:
                continue
            day_data['flow_3dba'] = day_data[['flow_1db', 'flow_2db', 'flow_3db']].mean(axis=1)
            day_y = day_data['flow']
            day_data.drop(['flow', 'flow_2db', 'flow_3db'], axis=1, inplace=True)
            day_flow = model_tree.predict(day_data)

            day_flow[day_flow < 0] = 0
            pred_flow.append(day_flow)

            if mode == 'valid':
                scores.append(mean_squared_error(day_y, day_flow))

            if day < split_day + predict_len - 1:
                test_data.loc[test_data['day'] == day + 1, 'flow_3db'] = test_data[test_data['day'] == day][
                    'flow_2db'].tolist()
                test_data.loc[test_data['day'] == day + 1, 'flow_2db'] = test_data[test_data['day'] == day][
                    'flow_1db'].tolist()
                test_data.loc[test_data['day'] == day + 1, 'flow_1db'] = day_flow

        pred_res[hour] = {'pred_flow': np.array(pred_flow)}
        if mode == 'valid':
            scores = np.array(scores)
            print(hour, round(1 / (scores.mean() ** 0.5 + 1), 6))
            pred_res[hour]['score'] = scores

    return pred_res


# %%

def validation_score(pred_res, predict_len=7):
    day_s = [0] * predict_len
    for k, v in pred_res.items():
        for i in range(predict_len):
            day_s[i] += v['score'][i]

    for i in range(predict_len):
        print('day' + str(i), 1 / ((day_s[i] / 24) ** 0.5 + 1))


# %%


# %%

'''
2-1 lightgbm validation
'''
valid_res_lgb = hour_level_flow_prediction(model='lightgbm', mode='valid')
validation_score(valid_res_lgb, 7)

# %%

'''
2-1 xgboost validation
'''
valid_res_xgb = hour_level_flow_prediction(model='xgboost', mode='valid')
validation_score(valid_res_xgb, 7)

# %%


# %%

'''
2-2 lightgbm prediction
'''
pred_res_lgb = hour_level_flow_prediction(model='lightgbm', mode='predict')
pred_flow_hour_level_lgb = np.array([pred_res_lgb[h]['pred_flow'] for h in range(24)])
pred_flow_hour_level_lgb.shape

# %%

'''
2-2 xgboost prediction
'''
pred_res_xgb = hour_level_flow_prediction(model='xgboost', mode='predict')
pred_flow_hour_level_xgb = np.array([pred_res_xgb[h]['pred_flow'] for h in range(24)])
pred_flow_hour_level_xgb.shape

# %%

'''
2-3 save the result of hour-level prediction
'''
with open('./cache/pred_flow_hour_level_lgb.pkl', 'wb') as f:
    pkl.dump(pred_flow_hour_level_lgb, f)
with open('./cache/pred_flow_hour_level_xgb.pkl', 'wb') as f:
    pkl.dump(pred_flow_hour_level_xgb, f)

# %% md

## 3. hour-level flow prediction for weekday

# %%

'''
3-1 lightgbm validation for weekday
'''
weekday_valid_res_lgb = hour_level_flow_prediction(model='lightgbm', mode='valid', weekday=True)
validation_score(weekday_valid_res_lgb, 5)

# %%

'''
3-1 xgboost validation for weekday
'''
weekday_valid_res_xgb = hour_level_flow_prediction(model='xgboost', mode='valid', weekday=True)
validation_score(weekday_valid_res_xgb, 5)

# %%


# %%

'''
3-2 lightgbm prediction for weekday
'''
weekday_pred_res_lgb = hour_level_flow_prediction(model='lightgbm', mode='predict', weekday=True)
weekday_pred_flow_hour_level_lgb = np.array([weekday_pred_res_lgb[h]['pred_flow'] for h in range(24)])
weekday_pred_flow_hour_level_lgb.shape

# %%

'''
3-2 xgboost prediction for weekday
'''
weekday_pred_res_xgb = hour_level_flow_prediction(model='xgboost', mode='predict', weekday=True)
weekday_pred_flow_hour_level_xgb = np.array([weekday_pred_res_xgb[h]['pred_flow'] for h in range(24)])
weekday_pred_flow_hour_level_xgb.shape

# %%

'''
3-3 save the result of hour-level prediction
'''
with open('./cache/weekday_pred_flow_hour_level_lgb.pkl', 'wb') as f:
    pkl.dump(weekday_pred_flow_hour_level_lgb, f)
with open('./cache/weekday_pred_flow_hour_level_xgb.pkl', 'wb') as f:
    pkl.dump(weekday_pred_flow_hour_level_xgb, f)