

from flask import Flask,render_template,request,flash,redirect,url_for
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import csv
import datetime
import tempfile
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV,cross_val_score
import joblib

# 隨機停止的措施
import time
import random

# 路徑要再調整
app=Flask(__name__,template_folder='templates', static_folder='static')




# -----------  匯入必要資料  -------------
game_all = pd.read_csv('game_full.csv')
game_all['DATE'] = pd.to_datetime(game_all['DATE'])
pitcher = pd.read_csv('pitcher_full.csv')
hitter = pd.read_csv('hitter_full.csv')
df_temp = game_all.loc[(game_all['DATE'] < datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())) & (game_all['SCORE_A'].isnull())]

# 這邊缺要處理已完成的場次

hitter['DATE'] = pd.to_datetime(hitter['DATE'])
pitcher['DATE'] = pd.to_datetime(pitcher['DATE'])

# -----------  預測場次的資料+ML  -------------

def each_game(no_each,dt_each):

    # 單一場次抓打線 (可能要在開打前 - 半小~一小才會有)
    def findline(no,dt):
        yr = dt[0:4]
        with tempfile.NamedTemporaryFile('w+t') as f:
            writer = csv.writer(f)
            url = f'http://www.cpbl.com.tw/games/box.html?&game_type=01&game_id={no}&game_date={dt}&pbyear={yr}'
            sp = BeautifulSoup(requests.get(url).text, 'html.parser')
            # 先有場次編號
            bat_line = [no]
            for bat in sp.select('.gap_b20 td a'):
                if bat.text not in set(sub.text for sub in sp.select('.gap_b20 td.sub a')):
                    bat_line.append(bat.text)
            bat_line = bat_line[0:19]
            # 以上只取打線，以下處理先發投手
            bat_line.extend([sp.select('.gap_b20')[2].select('.half_block')[team].find('a').text for team in range(2)])
            writer.writerow(bat_line)
            f.read()
            line = pd.read_csv(f.name,header=None)
            line.columns=['GAME','A1','A2','A3','A4','A5','A6','A7','A8','A9','H1','H2','H3','H4','H5','H6','H7','H8','H9','AP','HP']
            line['YEAR'] = int(yr)
        return line
    lineup = findline(no=no_each, dt=dt_each)

    # 單一場次 - 打者對戰對方先發投手成績 (可能要在開打前 - 半小~一小才會有)
    def vs(no,dt):

        # 對戰成績-crawl底層
        def findvs(sp):
            for n,yr in enumerate(sp.select('.h3_side')):
                #第一層,只取2020的對戰成績
                if (str(year) in yr.text) | (str(year-1) in yr.text):
                    for tr in sp.select('table')[n].select('tr'):
                        for i,th in enumerate(tr.select('td')):
                            df_raw[f"A{i}"].append(th.text)
                            if i == 9 :
                                df_raw["A10"].append(no) #新增場次->分析用
        year = int(dt[0:4])
        df_raw = {}
        for i in range(11):
            df_raw[f"A{i}"] = []
        # 如果原本欄位是日期格式，則需要{dt:%Y-%m-%d};這邊日期原本是文字格式
        url = f'http://www.cpbl.com.tw/pitcher/pbscore.html?&game_type=01&game_id={no}&game_date={dt}&pbyear={year}&type=&sgameno='
        sp = BeautifulSoup(requests.get(url).text, 'html.parser')
        findvs(sp=sp)
        df = pd.DataFrame(df_raw)
        df.columns = ['PLAYER', 'BATS', 'AB', 'H', 'HR', 'BB', 'IBB', 'HBP', 'SO', 'AVG', 'GAME']
        df['YEAR'] = year
        for col in ['AB', 'H', 'HR', 'BB', 'IBB', 'HBP', 'SO']:
            df[col] = df[col].astype(int)
        return df

    vsbat = vs(no=no_each, dt=dt_each)

    rfc_01 = joblib.load('rfc_01.pkl')
    rfc_score = joblib.load('rfc_score.pkl')

    game = game_all[game_all['GAME'] == no_each]

    # 函數 - 近幾天勝率計算 (主隊 & 客隊)  (資料,隊伍名,日期,近幾天)
    def win_rate(df,tm,dt,dif=14):
        # 篩選出要計算的表
        re1 = df.loc[(df['AWAY TEAM'].str.contains(tm))  & 
                        (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt- datetime.timedelta(days=1)))] # between 含前含後
        re2 = df.loc[ (df['HOME TEAM'].str.contains(tm)) & 
                    (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt- datetime.timedelta(days=1)))] # between 含前含後
        re = pd.concat([re1,re2])

        w_r = re[re['W'].str.contains(tm)]['W'].count() / re[re['W'] != "-"]['W'].count()
        get_s = pd.concat([re1['SCORE_A'],re2['SCORE_H']]).median()
        lose_s = pd.concat([re1['SCORE_H'],re2['SCORE_A']]).median()
        return w_r, get_s, lose_s

    # 函數 - 近期對戰成績 (主隊勝率)  (資料,隊伍名,日期,近幾天)
    def vs_w_rate(df,tm,tm2,dt,dif=28):
        # 篩選出要計算的表
        re1 = df.loc[ ((df['AWAY TEAM'].str.contains(tm)) & (df['HOME TEAM'].str.contains(tm2)))  &
                        (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt- datetime.timedelta(days=1)))] # between 含前含後
        re2 = df.loc[ ((df['AWAY TEAM'].str.contains(tm2)) & (df['HOME TEAM'].str.contains(tm)))  &
                      (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt- datetime.timedelta(days=1)))] # between 含前含後
        re = pd.concat([re1,re2])
        h_w_r = re[re['W'].str.contains(tm)]['W'].count() / re[re['W'] != "-"]['W'].count()
        h_get_s = pd.concat([re1['SCORE_A'],re2['SCORE_H']]).median()
        h_lose_s = pd.concat([re1['SCORE_H'],re2['SCORE_A']]).median() 
        return h_w_r, h_get_s, h_lose_s

    a = game.apply(lambda x: win_rate(game_all,x['AWAY TEAM'],x['DATE']), axis=1)
    game['A_WIN'] = a.map(lambda x:x[0])
    game['A_SCORE'] = a.map(lambda x:x[1])
    game['A_LOSE'] = a.map(lambda x:x[2])

    a = game.apply(lambda x: win_rate(game_all,x['HOME TEAM'],x['DATE']), axis=1)
    game['H_WIN'] = a.map(lambda x:x[0])
    game['H_SCORE'] = a.map(lambda x:x[1])
    game['H_LOSE'] = a.map(lambda x:x[2])

    a= game.apply(lambda x: vs_w_rate(game_all,x['HOME TEAM'],x['AWAY TEAM'],x['DATE']), axis=1)
    game['H_WIN_VS'] = a.map(lambda x:x[0])
    game['H_SCORE_VS'] = a.map(lambda x:x[1])
    game['H_LOSE_VS'] = a.map(lambda x:x[2])

    game = pd.merge(game,lineup,left_on=['GAME','YEAR'],right_on=['GAME','YEAR'],how='left')

    # 函數 - 打者近幾天 [安打+上壘,打席,打點] (資料,球員名,日期,近幾天)
    def bat_r(df, nm, dt, dif=7):
        try:
            re = df.loc[(df['NM'].str.contains(nm)) & 
                        (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt - datetime.timedelta(days=1)))]
            h_bb = (re['H'].sum()+re['BB'].sum())
            pa =  re['PA'].sum()
            rbi = re['RBI'].sum()
            return h_bb, pa, rbi
        except:
            print(nm)
            return (np.nan,np.nan,np.nan)

    for col in ['A1','A2','A3','A4','A5','A6','A7','A8','A9','H1','H2','H3','H4','H5','H6','H7','H8','H9']:
        a = game.apply(lambda x: bat_r(hitter,x[col],x['DATE']), axis=1)
        game[f'{col}_H_BB'] = a.map(lambda x:x[0])
        game[f'{col}_PA'] = a.map(lambda x:x[1])
        game[f'{col}_RBI'] = a.map(lambda x:x[2])

    # 函數 - 先發投手近幾天 [自責分率,平均每場投幾局,whip+HR,每九局三振]  (資料,球員名,日期,近幾天)
    def pit_r(df, nm, dt, dif=16):
        try:
            re = df.loc[(df['NM'].str.contains(nm)) & 
                        (df['DATE'].between(dt - datetime.timedelta(days=dif) ,dt - datetime.timedelta(days=1)))]
            era = re['ER'].sum()*9/re['IP'].sum()
            per_ip = re['IP'].sum()/len(re)
            whip_hr = (re['H'].sum() + re['BB'].sum()+re['HR'].sum()*4)/re['IP'].sum()
            so = re['SO'].sum()*9/re['IP'].sum()
            return era, per_ip , whip_hr, so
        except:
            print(nm)
            return (np.nan,np.nan,np.nan,np.nan)

    for col in ['AP','HP']:
        a = game.apply(lambda x: pit_r(pitcher,x[col],x['DATE']), axis=1)
        game[f'{col}_ERA'] = a.map(lambda x:x[0])
        game[f'{col}_PER_IP'] = a.map(lambda x:x[1])
        game[f'{col}_WHIP_HR'] = a.map(lambda x:x[2])
        game[f'{col}_SO'] = a.map(lambda x:x[3])

    # 函數 - 對戰打擊成績 [H+BB,打席數,被三振數]]  (資料,球員名,場次,年度)
    def hit_vs(df, nm, gm, yr):
        try:
            re = df.loc[(df['PLAYER'].str.contains(nm)) & (df['GAME'] == gm) & (df['YEAR'] == yr)]
            h_bb = (re['H'].sum()+re['BB'].sum())
            pa =  re['AB'].sum()+re['BB'].sum()
            bso = re['SO'].sum()
            return h_bb, pa, bso
        except:
            print(nm)
            return (np.nan,np.nan,np.nan)

    for col in ['A1','A2','A3','A4','A5','A6','A7','A8','A9','H1','H2','H3','H4','H5','H6','H7','H8','H9']:
        a = game.apply(lambda x: hit_vs(vsbat,x[col],x['GAME'],x['YEAR']), axis=1)
        game[f'{col}_H_BB_VS'] = a.map(lambda x:x[0])
        game[f'{col}_PA_VS'] = a.map(lambda x:x[1])
        game[f'{col}_BSO_VS'] = a.map(lambda x:x[2])

    #主客場球隊 近期狀況  
    game['H-A_WIN'] = game['H_WIN'] - game['A_WIN']
    game['H-A_SCORE'] = game['H_SCORE'] - game['A_SCORE']
    game['H-A_LOSE'] = game['H_LOSE'] - game['A_LOSE']

    #近期主客場球隊 打擊狀況 [1-6棒 and 4-6棒打點能力]
    game['A_OBP_16'] = (game['A1_H_BB']+game['A2_H_BB']+game['A3_H_BB']+game['A4_H_BB']+game['A5_H_BB']+game['A6_H_BB'])/(game['A1_PA']+game['A2_PA']+game['A3_PA']+game['A4_PA']+game['A5_PA']+game['A6_PA'])
    game['A_OBP_79'] = (game['A7_H_BB']+game['A8_H_BB']+game['A9_H_BB'])/(game['A7_PA']+game['A8_PA']+game['A9_PA'])
    game['A_RBI_46'] = (game['A4_RBI']+game['A5_RBI']+game['A6_RBI'])/(game['A4_PA']+game['A5_PA']+game['A6_PA'])

    game['H_OBP_16'] = (game['H1_H_BB']+game['H2_H_BB']+game['H3_H_BB']+game['H4_H_BB']+game['H5_H_BB']+game['H6_H_BB'])/(game['H1_PA']+game['H2_PA']+game['H3_PA']+game['H4_PA']+game['H5_PA']+game['H6_PA'])
    game['H_OBP_79'] = (game['H7_H_BB']+game['H8_H_BB']+game['H9_H_BB'])/(game['H7_PA']+game['H8_PA']+game['H9_PA'])
    game['H_RBI_46'] = (game['H4_RBI']+game['H5_RBI']+game['H6_RBI'])/(game['H4_PA']+game['H5_PA']+game['H6_PA'])

    #對戰對方先發投手打擊成績 [1-6棒 OBP及被三振率]
    game['A_OBP_16_VS'] = (game['A1_H_BB_VS']+game['A2_H_BB_VS']+game['A3_H_BB_VS']+game['A4_H_BB_VS']+game['A5_H_BB_VS']+game['A6_H_BB_VS'])/(game['A1_PA_VS']+game['A2_PA_VS']+game['A3_PA_VS']+game['A4_PA_VS']+game['A5_PA_VS']+game['A6_PA_VS'])
    game['A_OBP_79_VS'] = (game['A7_H_BB_VS']+game['A8_H_BB_VS']+game['A9_H_BB_VS'])/(game['A7_PA_VS']+game['A8_PA_VS']+game['A9_PA_VS'])

    game['H_OBP_16_VS'] = (game['H1_H_BB_VS']+game['H2_H_BB_VS']+game['H3_H_BB_VS']+game['H4_H_BB_VS']+game['H5_H_BB_VS']+game['H6_H_BB_VS'])/(game['H1_PA_VS']+game['H2_PA_VS']+game['H3_PA_VS']+game['H4_PA_VS']+game['H5_PA_VS']+game['H6_PA_VS'])
    game['H_OBP_79_VS'] = (game['H7_H_BB_VS']+game['H8_H_BB_VS']+game['H9_H_BB_VS'])/(game['H7_PA_VS']+game['H8_PA_VS']+game['H9_PA_VS'])

    game['A_BSO_16_VS'] = (game['A1_BSO_VS']+game['A2_BSO_VS']+game['A3_BSO_VS']+game['A4_BSO_VS']+game['A5_BSO_VS']+game['A6_BSO_VS'])/(game['A1_PA_VS']+game['A2_PA_VS']+game['A3_PA_VS']+game['A4_PA_VS']+game['A5_PA_VS']+game['A6_PA_VS'])
    game['A_BSO_79_VS'] = (game['A7_BSO_VS']+game['A8_BSO_VS']+game['A9_BSO_VS'])/(game['A7_PA_VS']+game['A8_PA_VS']+game['A9_PA_VS'])

    game['H_BSO_16_VS'] = (game['H1_BSO_VS']+game['H2_BSO_VS']+game['H3_BSO_VS']+game['H4_BSO_VS']+game['H5_BSO_VS']+game['H6_BSO_VS'])/(game['H1_PA_VS']+game['H2_PA_VS']+game['H3_PA_VS']+game['H4_PA_VS']+game['H5_PA_VS']+game['H6_PA_VS'])
    game['H_BSO_79_VS'] = (game['H7_BSO_VS']+game['H8_BSO_VS']+game['H9_BSO_VS'])/(game['H7_PA_VS']+game['H8_PA_VS']+game['H9_PA_VS'])

    train_X = game[['H-A_WIN', 'H-A_SCORE', 'H-A_LOSE', 'H_WIN_VS', 'H_SCORE_VS', 'H_LOSE_VS', 'AP_ERA', 'AP_PER_IP', 'AP_WHIP_HR', 'AP_SO', 'HP_ERA', 'HP_PER_IP', 'HP_WHIP_HR', 'HP_SO','A_OBP_16', 'A_OBP_79', 'A_RBI_46', 'H_OBP_16', 'H_OBP_79', 'H_RBI_46', 'A_OBP_16_VS', 'A_OBP_79_VS', 'H_OBP_16_VS', 'H_OBP_79_VS', 'A_BSO_16_VS', 'A_BSO_79_VS', 'H_BSO_16_VS', 'H_BSO_79_VS']]

    miss_dict = {'AP_ERA': [4.017857142857143, 6.0],
    'AP_PER_IP': [5.6, 6.1],
    'AP_SO': [6.9230769230769225, 9.0],
    'AP_WHIP_HR': [1.8, 2.272727272727273],
    'A_BSO_16_VS': [0.15384615384615385, 0.19391562595187328],
    'A_BSO_79_VS': [0.18181818181818182, 0.25],
    'A_OBP_16': [0.3645833333333333, 0.40304525758305243],
    'A_OBP_16_VS': [0.36231884057971014, 0.40663997019929227],
    'A_OBP_79': [0.30303030303030304, 0.36363636363636365],
    'A_OBP_79_VS': [0.3125, 0.37037037037037035],
    'A_RBI_46': [0.16071428571428573, 0.21739130434782608],
    'H-A_LOSE': [0.0, 1.5],
    'H-A_SCORE': [0.0, 1.5],
    'H-A_WIN': [0.0, 0.2],
    'HP_ERA': [4.122137404580153, 6.545454545454546],
    'HP_PER_IP': [5.55, 6.1],
    'HP_SO': [6.9230769230769225, 8.809143222506394],
    'HP_WHIP_HR': [1.8, 2.321428571428572],
    'H_BSO_16_VS': [0.15384615384615385, 0.19575873827791987],
    'H_BSO_79_VS': [0.16666666666666666, 0.23715538847117795],
    'H_LOSE_VS': [5.0, 7.0],
    'H_OBP_16': [0.37, 0.4105263157894737],
    'H_OBP_16_VS': [0.36, 0.40514881970578176],
    'H_OBP_79': [0.3103448275862069, 0.37037037037037035],
    'H_OBP_79_VS': [0.3103448275862069, 0.375],
    'H_RBI_46': [0.16326530612244897, 0.22641509433962265],
    'H_SCORE_VS': [5.0, 7.0],
    'H_WIN_VS': [0.5, 0.6666666666666666]}

    for col in train_X.columns:
        train_X[col] = train_X[col].fillna(miss_dict[col][0])
        # 在計算時 X/0 不會出錯(因為為numpy float64格式，會回傳inf); 因此數值本就應就高，故以下以Q3取代
        train_X[col] = train_X[col].replace(np.inf,miss_dict[col][1]).replace(-np.inf,miss_dict[col][1])

    y_pred = rfc_01.predict_proba(train_X)[:,1]
    y_p_pred = rfc_score.predict(train_X)

    return y_pred[0] ,y_p_pred[0]


# ---------- route設定 ----------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gamelist')
def game_tb():
    return render_template("gamelist.html", data=
                           game_all[['GAME','DATE','AWAY TEAM','SCORE_A','HOME TEAM','SCORE_H','W']].to_html(
                               index = False, na_rep = "-", float_format = '%.0f', 
                               classes=["table","table-sm", "table-hover","table-striped","table-borderless","table-responsive-lg","text-primary","text-right"]))

@app.route("/predict", methods=['GET', 'POST'])
def select_records():
    if request.method == 'POST':
        # each_game(no_each = ,dt_each)
        for y in request.form.values():
            for d in game_all[game_all['GAME'] == int(y.strip("'"))].index:
                dd = f"{game_all.loc[d,'DATE']:%Y-%m-%d}"
                away = game_all.loc[d,'AWAY TEAM']
                home = game_all.loc[d,'HOME TEAM']
                pred = each_game(no_each = int(y.strip("'")),dt_each= dd)
        return render_template("pred.html", p1 = pred[0], p2=pred[1], h=home, a=away, dd=dd)
    else:
        df_5g = game_all.loc[game_all['DATE'] < datetime.datetime.now(),['GAME','DATE','AWAY TEAM','HOME TEAM']].tail(5)
        gg = [i for i in game_all.loc[game_all['DATE'] < datetime.datetime.now(),'GAME']]
        gg = set(gg[:len(gg)-6:-1])
        return render_template("game_select.html", uniques=gg, df_5g=df_5g.to_html(
            index=False,
            classes=["table","table-borderless", "table-hover","table-responsive-lg","text-right"]))

@app.route("/graph_pit")
def graph_pit():
    df_empty = pd.DataFrame()
    df_empty['NM'] = pitcher['NM']
    df_empty['DATE'] = pitcher['DATE']
    df_empty['ERA'] = pitcher['ERA']
    df_empty['DT'] = pitcher['DATE'].map(lambda x: str(x)[0:10])
    df_empty['H_IP'] = np.where(pitcher['IP'] != 0,pitcher['H'] / pitcher['IP'],np.nan)
    df_empty['NP_IP'] = np.where(pitcher['IP'] != 0,pitcher['NP'] / pitcher['IP'],np.nan)
    df_empty['S_NP'] = np.where(pitcher['NP'] != 0,pitcher['S'] / pitcher['NP'],np.nan)
    df_empty['BB9'] = np.where(pitcher['IP'] != 0,(pitcher['BB']*9) / pitcher['IP'],np.nan)
    df_empty['K9'] = np.where(pitcher['IP'] != 0,(pitcher['SO']*9) / pitcher['IP'],np.nan)
    df_empty = df_empty.sort_values('DATE')
    df_pit = df_empty[['DT','NM','ERA','H_IP','NP_IP','S_NP','BB9','K9']].to_dict('list')
    return render_template("graph_pit.html", df_pit = df_pit)
   
if __name__ == '__main__':
    app.run()
