import os
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# coding: utf-8


work_dir = './'
output_dir = work_dir + 'train_files/'
all_train_aiq = 'df_airQuality_201701_201804.csv'
all_train_grid_weather = 'df_grid_201701_201804.csv'
test_grid_weather = 'df_grid_20180501_20180502.csv'


holiday = ['2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29',
               '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02', '2017-04-02',
               '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
               '2017-05-28', '2017-05-29', '2017-05-30', '2017-10-01', '2017-10-02',
               '2017-10-03', '2017-10-04', '2017-10-05', '2017-10-06', '2017-10-07',
               '2017-10-08', '2017-12-30', '2017-12-31',
               '2018-01-01', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18',
               '2018-02-19', '2018-02-20', '2018-02-21', '2018-04-05', '2018-04-06',
               '2018-04-07', '2018-04-29', '2018-04-30', '2018-05-01', '2018-06-16',
               '2018-06-17', '2018-06-18']

work = ['2017-01-22', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30',
            '2018-02-11', '2018-02-24', '2018-04-08', '2018-04-28']
rest_first_day = ['2017-01-27', '2017-02-05', '2017-04-02', '2017-05-28', '2017-10-01', '2018-02-15', '2018-02-25',
                      '2018-04-05', '2018-04-29']
rest_last_day = ['2017-01-02', '2017-01-21', '2017-02-02', '2017-02-05', '2017-04-04', '2017-05-01', '2017-05-30',
                     '2018-01-01', '2018-02-21', '2018-04-07', '2018-05-01']
work_first_day = ['2017-01-03', '2017-01-22', '2017-02-03', '2017-04-05', '2017-05-02', '2017-05-31', '2018-01-02',
                     '2018-02-11', '2018-02-22', '2018-04-08', '2018-05-02']
work_last_day = ['2017-01-26', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2018-02-14', '2018-02-24',
                     '2018-04-04', '2018-04-28']

not_rest_first_day = ['2017-01-28', '2017-02-04', '2017-04-01', '2017-05-27', '2017-09-30', '2017-10-07',
                          '2018-02-17',
                          '2018-02-24', '2018-04-07', '2018-04-28']
not_rest_last_day = ['2017-01-01', '2017-01-22', '2017-02-29', '2017-04-02', '2017-04-30', '2017-05-28',
                         '2017-10-01'
                         '2017-12-31', '2018-02-11', '2018-02-18', '2018-04-08', '2018-04-29']
not_work_first_day = ['2017-01-02', '2017-01-23', '2017-01-30', '2017-04-03', '2017-05-01', '2017-05-29',
                         '2017-10-02',
                         '2018-01-01', '2018-02-12', '2018-02-19', '2018-04-09', '2018-04-30']
not_work_last_day = ['2017-01-27', '2017-02-03', '2017-03-31', '2017-05-26', '2017-09-29', '2017-10-06',
                         '2018-02-16',
                         '2018-02-23', '2018-04-04', '2018-04-06', '2018-04-27']


station_grid ={'miyunshuiku_aq': 'beijing_grid_414', 'tiantan_aq': 'beijing_grid_303', 'yizhuang_aq': 'beijing_grid_323',
           'pingchang_aq': 'beijing_grid_264', 'zhiwuyuan_aq': 'beijing_grid_262', 'qianmen_aq': 'beijing_grid_303',
           'pinggu_aq': 'beijing_grid_452', 'beibuxinqu_aq': 'beijing_grid_263', 'shunyi_aq': 'beijing_grid_368',
           'tongzhou_aq': 'beijing_grid_366', 'yungang_aq': 'beijing_grid_239', 'yufa_aq': 'beijing_grid_278',
           'wanshouxigong_aq': 'beijing_grid_303', 'mentougou_aq': 'beijing_grid_240',
           'dingling_aq': 'beijing_grid_265',
           'donggaocun_aq': 'beijing_grid_452', 'nongzhanguan_aq': 'beijing_grid_324', 'liulihe_aq': 'beijing_grid_216',
           'xizhimenbei_aq': 'beijing_grid_283', 'fangshan_aq': 'beijing_grid_238', 'nansanhuan_aq': 'beijing_grid_303',
           'huairou_aq': 'beijing_grid_349', 'dongsi_aq': 'beijing_grid_303', 'badaling_aq': 'beijing_grid_224',
           'yanqin_aq': 'beijing_grid_225', 'gucheng_aq': 'beijing_grid_261', 'fengtaihuayuan_aq': 'beijing_grid_282',
           'wanliu_aq': 'beijing_grid_283', 'yongledian_aq': 'beijing_grid_385', 'aotizhongxin_aq': 'beijing_grid_304',
           'dongsihuan_aq': 'beijing_grid_324', 'daxing_aq': 'beijing_grid_301', 'miyun_aq': 'beijing_grid_392',
           'guanyuan_aq': 'beijing_grid_282', 'yongdingmennei_aq': 'beijing_grid_303'
           }

st_type = {'dongsi_aq': 	'Urban Stations',
          'tiantan_aq': 	'Urban Stations',
          'guanyuan_aq': 	'Urban Stations',
          'wanshouxigong_aq': 	'Urban Stations',
          'aotizhongxin_aq': 	'Urban Stations',
          'nongzhanguan_aq': 	'Urban Stations',
          'wanliu_aq': 	'Urban Stations',
          'beibuxinqu_aq': 	'Urban Stations',
          'zhiwuyuan_aq': 	'Urban Stations',
          'fengtaihuayuan_aq': 	'Urban Stations',
          'yungang_aq': 	'Urban Stations',
          'gucheng_aq': 	'Urban Stations',
          'fangshan_aq': 	'Suburban Stations',
          'daxing_aq': 	'Suburban Stations',
          'yizhuang_aq': 	'Suburban Stations',
          'tongzhou_aq': 	'Suburban Stations',
          'shunyi_aq': 	'Suburban Stations',
          'pingchang_aq': 	'Suburban Stations',
          'mentougou_aq': 	'Suburban Stations',
          'pinggu_aq': 	'Suburban Stations',
          'huairou_aq': 	'Suburban Stations',
          'miyun_aq': 	'Suburban Stations',
          'yanqin_aq': 	'Suburban Stations',
          'dingling_aq':       	'Other Stations',
          'badaling_aq':       	'Other Stations',
          'miyunshuiku_aq':    	'Other Stations',
          'donggaocun_aq':     	'Other Stations',
          'yongledian_aq':     	'Other Stations',
          'yufa_aq':           	'Other Stations',
          'liulihe_aq':        	'Other Stations',
          'qianmen_aq':        	'Stations Near Traffic',
          'yongdingmennei_aq': 	'Stations Near Traffic',
          'xizhimenbei_aq':    	'Stations Near Traffic',
          'nansanhuan_aq':     	'Stations Near Traffic',
          'dongsihuan_aq':     	'Stations Near Traffic'}

def smape_value(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(
        np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


def preprocessing_train_data():
    airQuality_201701_201801 = 'airQuality_201701-201801.csv'
    df_airQuality_201701_201801 = pd.read_csv(work_dir + airQuality_201701_201801)
    del df_airQuality_201701_201801['CO']
    del df_airQuality_201701_201801['NO2']
    del df_airQuality_201701_201801['SO2']

    aiqQuality_201804 = 'aiqQuality_201804.csv'
    df_aiqQuality_201804 = pd.read_csv(work_dir + aiqQuality_201804)
    del df_aiqQuality_201804['CO_Concentration']
    del df_aiqQuality_201804['NO2_Concentration']
    del df_aiqQuality_201804['SO2_Concentration']
    del df_aiqQuality_201804['id']
    df_aiqQuality_201804.columns = df_airQuality_201701_201801.columns

    airQuality_201802_201803 = 'airQuality_201802-201803.csv'
    df_airQuality_201802_201803 = pd.read_csv(work_dir + airQuality_201802_201803)
    del df_airQuality_201802_201803['CO']
    del df_airQuality_201802_201803['NO2']
    del df_airQuality_201802_201803['SO2']
    frames = [df_airQuality_201701_201801, df_airQuality_201802_201803, df_aiqQuality_201804]
    df_airQuality_201701_201804 = pd.concat(frames)
    df_airQuality_201701_201804.rename(columns={'stationId': 'station_id', 'utc_time': 'time'}, inplace=True)
    df_airQuality_201701_201804.to_csv(output_dir + 'df_airQuality_201701_201804.csv', index=None)

    df_grid_201701_201803 = pd.read_csv(work_dir + 'gridWeather_201701-201803.csv')
    del df_grid_201701_201803['longitude']
    del df_grid_201701_201803['latitude']
    df_grid_201701_201803 = df_grid_201701_201803.drop_duplicates(["stationName", "utc_time"])

    df_grid_201804 = pd.read_csv(work_dir + 'gridWeather_201804.csv')
    del df_grid_201804['id']
    del df_grid_201804['weather']

    df_grid_201804 = df_grid_201804.drop_duplicates(["station_id", "time"])
    df_grid_201701_201803.columns = df_grid_201804.columns

    df_grid_201701_201803.columns = df_grid_201804.columns
    frames = [df_grid_201701_201803, df_grid_201804]
    df_grid_201701_201804 = pd.concat(frames)
    df_grid_201701_201804.to_csv(output_dir + 'df_grid_201701_201804.csv', index=None)


def preprocessing_test_data():
    df_grid_20180501_20180502 = pd.read_csv(work_dir + 'gridWeather_20180501-20180502.csv')
    del df_grid_20180501_20180502['id']
    del df_grid_20180501_20180502['weather']
    df_grid_20180501_20180502 = df_grid_20180501_20180502.drop_duplicates(["station_id", "time"])
    df_grid_20180501_20180502.head()
    df_grid_20180501_20180502.to_csv(output_dir + 'df_grid_20180501_20180502.csv', index=None)


def get_grid_weather_data(grid_id, n_station_name):
   df_grid_201701_201804 = pd.read_csv(output_dir + 'df_grid_201701_201804.csv')
   df_grid_id_201701_201804 = df_grid_201701_201804[df_grid_201701_201804['station_id'] == grid_id]
   df_grid_id_201701_201804.to_csv(output_dir + 'df_' + grid_id +'_201701_201804.csv',index= None)
   df_grid_id_201701_201804.to_csv(output_dir + 'df_' + n_station_name +'_201701_201804.csv',index= None)
   return df_grid_id_201701_201804
   

def get_grid_aiq_data(n_station_name):
    df_airQuality_201701_201804 = pd.read_csv(output_dir + 'df_airQuality_201701_201804.csv')
    df_grid_id_aiq_201701_201804 = df_airQuality_201701_201804[df_airQuality_201701_201804['station_id'] == n_station_name]
    df_grid_id_aiq_201701_201804.to_csv(output_dir + 'df_airQuality_201701_201804_' + n_station_name +'.csv',index= None)
    return df_grid_id_aiq_201701_201804


def join_weather_and_aiq(grid_id, n_station_name):
    weather = get_grid_weather_data(grid_id, n_station_name)
    aiq = get_grid_aiq_data(n_station_name)
    df_join_data = aiq.set_index('time').join(weather.set_index('time'), how='left', lsuffix='_left', rsuffix='_right')
    del df_join_data['station_id_right']
    return df_join_data


def is_holiday(time):
    date = time[:10]
    # print(date)
    if (date in holiday):
        return 1
    return 0


def is_work(time, week):
    date = time[:10]
    # print(date)
    if date in holiday or ((week >= 5 and (date not in work))):
        return 0
    else:
        return 1


def is_rest_last_day(time, week):
    date = time[:10]
    if (week == 6 and (not date in not_rest_last_day)) or (date in rest_last_day):
        return 1
    return 0


def is_work_firt_day(time, week):
    date = time[:10]
    if (week == 0 and (not date in not_work_first_day)) or (date in work_first_day):
        return 1
    return 0


def is_work_last_day(time, week):
    date = time[:10]
    if (week == 4 and (not date in not_work_last_day)) or (date in work_last_day):
        return 1
    return 0


def is_rest_first_day(time, week):
    date = time[:10]
    if (week == 5 and (not date in not_rest_first_day)) or (date in rest_first_day):
        return 1
    return 0


def add_features_time_type(df_join_data, station_type):
    df_join_data['station_type'] = station_type
    df_join_data.index = pd.DatetimeIndex(df_join_data.index)
    df_join_data['time'] = df_join_data.index.map(lambda x: str(x))
    df_join_data['time_week'] = df_join_data.index.map(lambda x: x.weekday)
    df_join_data['time_year'] = df_join_data.index.map(lambda x: x.year)
    df_join_data['time_month'] = df_join_data.index.map(lambda x: x.month)
    df_join_data['time_day'] = df_join_data.index.map(lambda x: x.day)
    df_join_data['time_hour'] = df_join_data.index.map(lambda x: x.hour)
    df_join_data['holiday'] = df_join_data.apply(lambda x :is_holiday(x['time']), axis = 1)
    df_join_data['work'] = df_join_data.apply(lambda x: is_work(x['time'], x['time_week']), axis = 1)
    df_join_data['work_firt_day'] = df_join_data.apply(lambda x: is_work_firt_day(x.time, x.time_week), axis = 1)
    df_join_data['rest_last_day'] = df_join_data.apply(lambda x: is_rest_last_day(x.time, x.time_week), axis = 1)
    df_join_data['work_last_day'] = df_join_data.apply(lambda x: is_work_last_day(x.time, x.time_week), axis = 1)
    df_join_data['rest_first_day'] = df_join_data.apply(lambda x: is_rest_first_day(x.time, x.time_week), axis = 1)
    return df_join_data


def fill_null_by_mean(df_join_data):
    df_join_data['PM2.5'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['PM2.5'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['PM10'] = df_join_data.groupby(['time_year', 'time_month' , 'time_day'])['PM10'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['O3'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['O3'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['temperature'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['temperature'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['pressure'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['pressure'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['humidity'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['humidity'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['wind_speed'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['wind_speed'].transform(lambda x: x.fillna(x.mean()))
    df_join_data['wind_direction'] = df_join_data.groupby(['time_year', 'time_month',  'time_day'])['wind_direction'].transform(lambda x: x.fillna(x.mean()))

    return df_join_data


def dropna_last(df_join_data):
    df_join_data = df_join_data.dropna(axis=0, how='any')
    return df_join_data

def add_stastistic_features(df_join_data):
    df_join_data['max_PM25_all'] = np.max(df_join_data['PM2.5'])
    df_join_data['mean_PM25_all'] = np.mean(df_join_data['PM2.5'])
    df_join_data['median_PM25_all'] = np.median(df_join_data['PM2.5'])
    df_join_data['sum_PM25_all'] = np.sum(df_join_data['PM2.5'])
    df_join_data['min_PM25_all'] = np.min(df_join_data['PM2.5'])
    df_join_data['var_PM25_all'] = np.var(df_join_data['PM2.5'])
    df_join_data['std_PM25_all'] = np.std(df_join_data['PM2.5'])
    return df_join_data


def get_grid_weather_data_test(grid_id, n_station_name):
   df_grid_20180501_20180502 = pd.read_csv(output_dir + 'df_grid_20180501_20180502.csv')
   df_grid_id_20180501_20180502 = df_grid_20180501_20180502[df_grid_20180501_20180502['station_id'] == grid_id]
   df_grid_id_20180501_20180502.to_csv(output_dir + 'df_' + grid_id +'20180501_20180502.csv',index= None)
   df_grid_id_20180501_20180502.to_csv(output_dir + 'df_' + n_station_name +'20180501_20180502.csv',index= None)
   df_grid_id_20180501_20180502 = df_grid_id_20180501_20180502.set_index('time')
   return df_grid_id_20180501_20180502


def grid_to_station_test_data(df_grid, station_name):
    del df_grid['station_id']
    df_grid['station_id_left'] = station_name
    df_grid.to_csv(output_dir + station_name +'_df_20180501_20180502_' + station_name +'.csv',index=None)
    return df_grid


def prepreocessing_first():
    if not os.path.exists(output_dir + all_train_aiq) or \
            not os.path.exists(output_dir + all_train_grid_weather):
        preprocessing_train_data()

    if not os.path.exists(output_dir + test_grid_weather):
        preprocessing_test_data()




def generate_traning_data_csv():
    """
    generate_traning_data_csv for every staion
    :return:
    """

    for station_name in station_grid.keys():
        df_station_name_tran_data = join_weather_and_aiq(station_grid[station_name], station_name)
        df_station_name_tran_data = add_features_time_type(df_station_name_tran_data, st_type[station_name])
        df_station_name_tran_data = fill_null_by_mean(df_station_name_tran_data)
        df_station_name_tran_data = dropna_last(df_station_name_tran_data)
        df_station_name_tran_data = add_stastistic_features(df_station_name_tran_data)
        df_station_name_tran_data.to_csv(output_dir + station_name + '_df_train_data_201701_201804.csv', index=None)


def generate_testing_data_csv():
    """
    generate_traning_data_csv for every staion
    :return:
    """

    for station_name in station_grid.keys():

        df_station_name_tran_data=pd.read_csv(output_dir + station_name + '_df_train_data_201701_201804.csv')
        df_station = get_grid_weather_data_test(station_grid[station_name], station_name)
        df_station = add_features_time_type(df_station, st_type[station_name])
        add_features = ['max_PM25_all',
                        'mean_PM25_all',
                        'median_PM25_all',
                        'sum_PM25_all',
                        'min_PM25_all',
                        'var_PM25_all',
                        'std_PM25_all']
        for feature in add_features:
            df_station[feature] = df_station_name_tran_data[feature][0]
        grid_to_station_test_data(df_station, station_name)



def merge_train_data():
    frames = []
    for station_name in station_grid.keys():
        df_station_name_train = pd.read_csv(output_dir + station_name + '_df_train_data_201701_201804.csv')
        frames.append(df_station_name_train)
    all_df_train = pd.concat(frames)
    all_df_train = all_df_train.sort_values(['time'])
    all_df_train.to_csv(output_dir + 'df_train_data_201701_201804_all.csv', index =None )

def merge_test_data():
    frames = []
    for station_name in station_grid.keys():
        df_station_name_test = pd.read_csv(output_dir + station_name +'_df_20180501_20180502_' + station_name +'.csv')
        frames.append(df_station_name_test)
    all_df_test = pd.concat(frames)
    all_df_test=all_df_test.sort_values(['station_id_left', 'time'])
    all_df_test.to_csv(output_dir + 'df_test_data_20180501_20180502_all.csv', index =None )



def number_hour(station_id, time_day, time_hour):
    num=0
    if(time_day==2):
       num += (24+time_hour)
    else:
       num += time_hour
    return station_id + '#' +str(num)


def prepare_for_train(train_file_name, to_test_file_name):
    df_train_data = pd.read_csv( train_file_name)
    del df_train_data['time']
    train_data = pd.get_dummies(df_train_data)
    # train_data.head()

    features = train_data.columns.values.tolist()
    labels =['PM2.5', 'PM10', 'O3']
    for i in labels:
        features.remove(i)
    x_train = train_data[features]
    x_train.head()
    y_train = train_data[labels]

    df_test_data = pd.read_csv(to_test_file_name )
    del df_test_data['time']
    df_test_id = pd.DataFrame()
    df_test_id['test_id'] = df_test_data.apply(
        lambda x: number_hour(x.station_id_left, x.time_day, x.time_hour), axis=1)
    df_test_id.to_csv('test_id.csv', index = None)
    x_test = pd.get_dummies(df_test_data)

    return x_train, y_train, x_test, df_test_id


model_param = {'lr': 0.01, 'depth': 10, 'tree': 5000, 'leaf': 400, 'sample': 0.9, 'seed': 3}
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': {'l2', 'l1'},
    'max_depth': model_param['depth'],
    'num_leaves': model_param['leaf'],
    'min_data_in_leaf': 20,
    'learning_rate': model_param['lr'],
    'feature_fraction': 1,
    'bagging_fraction': model_param['sample'],
    'bagging_freq': 1,
    'bagging_seed': model_param['seed'],
    'verbose': 0
}


def train_lgb_PM25_and_Predict(label_name, time_window):
    x_train_all, y_train_all, x_test, df_test_id = prepare_for_train(output_dir + 'df_train_data_201701_201804_all.csv',
                                                                     output_dir + 'df_test_data_20180501_20180502_all.csv')
    del x_train_all['wind_direction']
    del x_train_all['wind_speed']
    del x_test['wind_direction']
    del x_test['wind_speed']
    y_train = y_train_all[0:-840 * time_window]
    x_train = x_train_all[0:-840 * time_window]
    y_validation = y_train_all[-840 * time_window:]
    x_validation = x_train_all[-840 * time_window:]
    lgb_train = lgb.Dataset(x_train, y_train[label_name])
    lgb_eval = lgb.Dataset(x_validation, y_validation[label_name], reference=lgb_train)
    gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=model_param['tree'],
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
    filename = 'best_gbm_pm25.pkl' + str(time_window)
    pickle.dump(gbm, open(filename, 'wb'))
    # pre_y_pm25_v = gbm.predict(x_validation)
    # v_score = smape_value(y_validation[label_name].values, pre_y_pm25_v)
    pre_y_test_pm25= gbm.predict(x_test)
    # print('smape', v_score)
    return pre_y_test_pm25



def train_lgb_PM10_and_Predict(label_name, time_window):
    x_train_all, y_train_all, x_test, df_test_id = prepare_for_train(output_dir + 'df_train_data_201701_201804_all.csv',
                                                                     output_dir + 'df_test_data_20180501_20180502_all.csv')
    del x_train_all['wind_direction']
    del x_train_all['wind_speed']
    del x_test['wind_direction']
    del x_test['wind_speed']
    y_train = y_train_all[0:-840 * time_window]
    x_train = x_train_all[0:-840 * time_window]
    y_validation = y_train_all[-840 * time_window:]
    x_validation = x_train_all[-840 * time_window:]
    lgb_train = lgb.Dataset(x_train, y_train[label_name])
    lgb_eval = lgb.Dataset(x_validation, y_validation[label_name], reference=lgb_train)
    gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=model_param['tree'],
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
    filename = 'best_gbm_pm10.pkl' + str(time_window)
    pickle.dump(gbm, open(filename, 'wb'))
    # pre_y_pm10_v = gbm.predict(x_validation)
    # v_score = smape_value(y_validation[label_name].values, pre_y_pm10_v)
    pre_y_test_pm10= gbm.predict(x_test)
    # print('smape', v_score)
    return pre_y_test_pm10


def train_lgb_O3_and_Predict(label_name, time_window):
    x_train_all, y_train_all, x_test, df_test_id = prepare_for_train(output_dir + 'df_train_data_201701_201804_all.csv',
                                                                     output_dir + 'df_test_data_20180501_20180502_all.csv')
    del x_train_all['wind_speed']
    del x_test['wind_speed']
    y_train = y_train_all[0:-840 * time_window]
    x_train = x_train_all[0:-840 * time_window]
    y_validation = y_train_all[-840 * time_window:]
    x_validation = x_train_all[-840 * time_window:]
    lgb_train = lgb.Dataset(x_train, y_train[label_name])
    lgb_eval = lgb.Dataset(x_validation, y_validation[label_name], reference=lgb_train)
    gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=model_param['tree'],
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
    filename = 'best_gbm_O3.pkl' + str(time_window)
    pickle.dump(gbm, open(filename, 'wb'))
    # pre_y_O3_v = gbm.predict(x_validation)
    # v_score = smape_value(y_validation[label_name].values, pre_y_O3_v)
    pre_y_test_O3= gbm.predict(x_test)
    # print('smape', v_score)
    return pre_y_test_O3



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        work_dir = sys.argv[1]
        output_dir = work_dir + 'train_files/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prepreocessing_first()
    generate_traning_data_csv()
    generate_testing_data_csv()
    merge_train_data()
    merge_test_data()
    pre_y_test_pm25 = train_lgb_PM25_and_Predict('PM2.5', 10)
    pre_y_test_pm10 = train_lgb_PM10_and_Predict('PM10', 2)
    pre_y_test_O3 = train_lgb_O3_and_Predict('O3', 1)
    df_test_id = pd.read_csv('test_id.csv')
    df_result_pre = pd.DataFrame(pre_y_test_pm25, columns=['PM2.5'])
    df_result_pre['PM10'] = pre_y_test_pm10
    df_result_pre['O3'] = pre_y_test_O3
    df_result_pre['test_id'] = df_test_id['test_id']
    df_pridect_result = df_result_pre.reindex(columns=['test_id', 'PM2.5', 'PM10', 'O3'])
    df_pridect_result.to_csv('submition.csv', index=None)









