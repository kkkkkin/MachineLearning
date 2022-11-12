import pickle as pkl
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


type2eng = {'旅游景点;公园': 'Attraction;Park', '教育培训;高等院校': 'Education;School', '购物;购物中心': 'Shopping;Mall',
            '医疗;综合医院': 'Medical;Hospital', '运动健身;体育场馆': 'Sport;Stadium', '旅游景点;文物古迹': 'Attraction;Monument',
            '旅游景点;风景区': 'Attraction;Scenic', '交通设施;火车站': 'Transportation;Train', '交通设施;长途汽车站': 'Transportation;Bus',
            '旅游景点;植物园': 'Attraction;Arboretum', '旅游景点;游乐园': 'Attraction;Amusement', '旅游景点;水族馆': 'Attraction;Aquarium',
            '旅游景点;动物园': 'Attraction;Zoo', '交通设施;飞机场': 'Transportation;Airport'}

# %%

area_info = pd.read_csv('../data/area_passenger_info.csv',
                        names=['id', 'name', 'type', 'center_x', 'center_y', 'grid_x', 'grid_y', 'area'])
area_info['type'] = area_info[['type']].apply(lambda x: type2eng[x['type']], axis=1)
area_info['type1'], area_info['type2'] = area_info['type'].str.split(';', 1).str
test_df = pd.read_csv('../data/test_submit_example.csv', names=['id', 'date', 'flow'])

with open('./cache/pred_flow_rule.pkl', 'rb') as f:
    pred_flow_rule = pkl.load(f)
with open('./cache/growth_index.pkl', 'rb') as f:
    growth_index = pkl.load(f)

with open('./cache/pred_flow_hour_level_lgb.pkl', 'rb') as f:
    pred_flow_hour_level_lgb = pkl.load(f)
with open('./cache/pred_flow_hour_level_xgb.pkl', 'rb') as f:
    pred_flow_hour_level_xgb = pkl.load(f)

with open('./cache/weekday_pred_flow_hour_level_lgb.pkl', 'rb') as f:
    weekday_pred_flow_hour_level_lgb = pkl.load(f)
with open('./cache/weekday_pred_flow_hour_level_xgb.pkl', 'rb') as f:
    weekday_pred_flow_hour_level_xgb = pkl.load(f)

# %%

test_df['flow'] = pred_flow_rule

# 1 hour-level fusion
oneday_flow = 0.5 * pred_flow_hour_level_lgb[:, 0, :].T + 0.5 * pred_flow_hour_level_xgb[:, 0, :].T

oneday_idx = test_df['date'].astype(str).str.contains('20200216')
flow_0216 = (oneday_flow.T * growth_index.tolist()).T
test_df.loc[oneday_idx, 'flow'] = 0.3 * test_df.loc[oneday_idx, 'flow'] + 0.7 * flow_0216.flatten()

oneday_idx = test_df['date'].astype(str).str.contains('20200223')
flow_0223 = (flow_0216.T * growth_index.tolist()).T
test_df.loc[oneday_idx, 'flow'] = 0.3 * test_df.loc[oneday_idx, 'flow'] + 0.7 * flow_0223.flatten()

# 2 hour-level of weekday fusion
w1, w2 = 0.7, 0.3
trans_idxs = [x - 1 for x in area_info[(area_info['type1'] != 'Transportation')]['id'].tolist()]
growth_index_weekday = growth_index * 1.05

for d in range(5):
    oneday_idx = test_df['date'].astype(str).str.contains(str(d + 20200217))
    oneday_flow = 0.5 * weekday_pred_flow_hour_level_lgb[:, d, :].T + 0.5 * weekday_pred_flow_hour_level_xgb[:, d, :].T
    oneday_flow = (oneday_flow.T * growth_index_weekday.tolist()).T
    oneday_flow[trans_idxs] = test_df.loc[oneday_idx, 'flow'].values.reshape(997, 24)[trans_idxs]
    test_df.loc[oneday_idx, 'flow'] = w1 * test_df.loc[oneday_idx, 'flow'] + w2 * oneday_flow.flatten()

# %%

# 3 write
test_df.to_csv('./output/test_submission_final.csv', header=0, index=0)

# %%

for i in range(20200216, 20200225):
    print(i, test_df[(test_df['date'] / 100).astype(int) == i].mean()['flow'])

# %%

test_df[test_df['date'] == 2020021712].head(30)
