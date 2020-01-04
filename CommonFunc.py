# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import lightgbm as lgb

# %%
def ExchangeData(data, key, checkbase, outputfile = False):
    colidx = data.columns.get_loc(key)
    datanum = data.shape[0]
    check = checkbase
    for idx in range(datanum):
        #データがない場合(NAはNAのまま)
        if pd.isnull(data.iat[idx,colidx]) == False :
            if np.any(check == data.iat[idx,colidx]) == False :
                check = np.append(check, data.iat[idx,colidx])
            #置き換え
            checkidx = np.where(check == data.iat[idx,colidx])
            #値の割り振りは1からスタートとする。(0はNANとかぶったりするので)
            data.iat[idx,colidx] = checkidx[0][0] + 1

    if outputfile == True :
        with open("exchangedata/exchangedata.txt", mode='a') as f:
            f.write(key+'\t')
            num = 0
            for checklist in check:
                num = num + 1
                f.write(str(num) + ':' + checklist + ',')
            f.write('\n')

    return check

#分類値に変換する
def ExchangeDataAll(data_train,data_test):
    f = open("exchangedata/exchangedata.txt", mode='w')
    f.close()
    for clmname in data_train.columns :
        if data_train[clmname].dtype == "object" :
            checkbase = np.arange(0)
            checkbase = ExchangeData(data_train,clmname,checkbase)
            checkbase = ExchangeData(data_test,clmname,checkbase, True)

def GraphOutput(data, x_name, y_name, sort=False):
    #並び替え必要であれば、昇順にソートする
    if(sort == True):
        df_s = train.sort_values(x_name)
        datax = df_s[x_name]
        datay = df_s[y_name]
    else:
        datax = data[x_name]
        datay = data[y_name]

    #グラフをファイルに出力
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(datax.values,datay.values)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.savefig('Graph/x_' + x_name + '_y_' + y_name + '.png')

#全要素の散布図出力
def GraphOutputAll(data, y_name):
    for clmname in data.columns :
        #yの要素はグラフ不要のため、対象外とする
        if clmname != y_name:
            GraphOutput(data,clmname,y_name)

#ヒストグラム出力
def HstgramOutput(data, key, binsValue=-1):
    #指定なしの場合、データ量から適当な分割数を割り出す
    if binsValue == -1:
        binsValue = int(1 + math.log(data[key].shape[0]))
    plt.hist(data[key], bins=binsValue)
    plt.savefig('Graph/00_HistGram_' + key + '.png')

def setfillNa(train_data, test_data, key, value):
    #中央値
    #a = train_data[key].median()
    #平均値
    #b = train_data[key].mean()
    train_data[key].fillna(value, inplace=True)
    test_data[key].fillna(value, inplace=True)

#対数変換
def LogConvert(data, key):
    newKey = key + "_Log"
    data[newKey] = np.log(data[key])

#対数変換
def ExpConvert(data, key):
    newKey = key + "_Exp"
    data[newKey] = np.exp(data[key])

#クロスバリレーション実行
#引数Xデータ、Yデータ、バリレーションデータ、データ数、
def CVExe(data_x, data_y, vali_data, expchange = False):
    tmp_data_x = np.insert(data_x, data_x.shape[1], vali_data, axis=1)
    tmp_data_y = np.insert(data_y, data_y.shape[1], vali_data, axis=1)
    RMSLE_AVE = 0.0
    RMSE_AVE = 0.0
    for idx in range(vali_data.max()):
        train_x = tmp_data_x[tmp_data_x[:,data_x.shape[1]]!=idx]
        train_y = tmp_data_y[tmp_data_y[:,data_y.shape[1]]!=idx]
        train_x = train_x[:,:data_x.shape[1]]
        train_y = train_y[:,data_y.shape[1]-1]

        test_x = tmp_data_x[tmp_data_x[:,data_x.shape[1]]==idx]
        test_y = tmp_data_y[tmp_data_y[:,data_y.shape[1]]==idx]
        test_x = test_x[:,:data_x.shape[1]]
        test_y = test_y[:,data_y.shape[1]-1]

# ランダムフォレスト
#        model=RandomForestRegressor()
# LightGBM
        model = lgb.LGBMRegressor()
        model.fit(train_x,train_y)
        Y_pred = model.predict(test_x)
        if expchange == True:
            test_y = np.exp(test_y.astype(float))
            Y_pred = np.exp(Y_pred)
        RMSLE=np.sqrt(mean_squared_log_error (test_y, Y_pred))
        RMSLE_AVE += RMSLE
        print("RMSLE:" + str(RMSLE))
#        RMSE=np.sqrt(mean_squared_error(test_y, Y_pred))
#        RMSE_AVE += RMSE
#        print("RMSE:" + str(RMSE))

    print("RMSLE AVE:" + str(RMSLE_AVE/vali_data.max()))
#    print("RMSE AVE:" + str(RMSE_AVE/vali_data.max()))

def ImportantFeature(data_x, data_y, data_name):
# ランダムフォレスト
#   model=RandomForestRegressor()
# LightGBM
    model = lgb.LGBMRegressor()
    model.fit(data_x, data_y[:,0])
    print('Training done using Random Forest')

    ranking = np.argsort(-model.feature_importances_)
    f, ax = plt.subplots(figsize=(11, 9))
    sns.barplot(x=model.feature_importances_[ranking], y=data_name[ranking], orient='h')
    ax.set_xlabel("feature importance")
    plt.tight_layout()
    plt.savefig('Graph/01_Feature_Ranking.png')

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
#checkbase = np.arange(0)
#checkbase = ExchangeData(train,"MSZoning",checkbase)
#checkbase = ExchangeData(test,"MSZoning",checkbase)
#ExchangeDataAll(train, test)
#GraphOutput(train, "LotFrontage", "SalePrice")
#GraphOutputAll(train,"SalePrice")
#HstgramOutput(train, "SalePrice")
#setfillNa(train, test, "SalePrice",10)
#LogConvert(train,"SalePrice")
#HstgramOutput(train, "SalePrice_Log")
#ExpConvert(train,"SalePrice_Log")
#HstgramOutput(train, "SalePrice_Log_Exp")
"""
ExchangeDataAll(train, test)
#NAを中央値変換
setfillNa(train, test, "GarageYrBlt", train["GarageYrBlt"].median())
setfillNa(train, test, "LotFrontage", train["LotFrontage"].median())
setfillNa(train, test, "MasVnrArea", train["MasVnrArea"].median())
setfillNa(train, test, "BsmtFinSF1", train["BsmtFinSF1"].median())
setfillNa(train, test, "GarageArea", train["GarageArea"].median())
setfillNa(train, test, "TotalBsmtSF", train["TotalBsmtSF"].median())
#NAを一番多い分類に変換
setfillNa(train, test, "MSZoning", 1)
setfillNa(train, test, "GarageCars", 2)
#NAをその他に分類
setfillNa(train, test, "Alley", 0)
setfillNa(train, test, "BsmtQual", 0)
setfillNa(train, test, "Fence", 0)
setfillNa(train, test, "FireplaceQu", 0)
setfillNa(train, test, "KitchenQual", 0)

validata = 0
datanum = train.shape[0]
np_vali = np.zeros(datanum)
for idx in range(datanum):
    np_vali[idx] = validata
    validata = validata + 1
    validata = validata % 5
train['vali_value'] = np_vali.astype(int)
print(train['vali_value'])

#正規分布にするために対数変換
LogConvert(train,"SalePrice")

train_data = train.values
FeatureData = [1,2,3,6,11,13,14,17,18,19,20,26,27,28,29,30,34,38,41,43,44,46,49,51,53,54,56,57,59,61,62,65,66,67,68,70,73]
x_tra_name = train.columns.values[FeatureData]
x_tra = train_data[:,FeatureData]

exchange_log = True
if exchange_log == False:
    #対数変換なしの場合
    colidx = train.columns.get_loc("SalePrice")
    y_tra = train_data[:,[colidx,colidx]]
else :
    #対数変換ありの場合
    colidx = train.columns.get_loc("SalePrice_Log")
    y_tra = train_data[:,[colidx,colidx]]



#CVExe(x_tra, y_tra, train['vali_value'].values, train.shape[0])
ImportantFeature(x_tra, y_tra, x_tra_name)
"""