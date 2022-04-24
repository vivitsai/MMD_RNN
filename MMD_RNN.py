#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
from json import JSONDecoder
import warnings
warnings.filterwarnings("ignore")



B_goods = pd.read_csv('B_goods.csv',encoding='utf-8')
B_order_info = pd.read_csv('B_order_info.csv',encoding='utf-8')
B_order_goods = pd.read_csv('B_order_goods.csv',encoding='utf-8')
B_comment = pd.read_csv('B_comment.csv',encoding='utf-8')


# 处理时间，转为年-月-日 时：分：秒


B_order_info = B_order_info.sort_values('add_time(下单时间)')
B_comment = B_comment.sort_values('add_time(评论时间时间戳)')
B_order_info['add_time(下单时间)'] = pd.to_datetime(B_order_info['add_time(下单时间)'],unit='s')
B_order_info['pay_time(支付时间)'] = pd.to_datetime(B_order_info['pay_time(支付时间)'],unit='s')
B_order_info['shipping_time(发货时间)'] = pd.to_datetime(B_order_info['shipping_time(发货时间)'],unit='s')
B_order_info['pay_time-add_time'] = (B_order_info['pay_time(支付时间)']-B_order_info['add_time(下单时间)']).dt.total_seconds()
B_order_info['shipping_tim-pay_time'] = (B_order_info['shipping_time(发货时间)']-B_order_info['pay_time(支付时间)']).dt.total_seconds()
B_order_info['pay_time(支付时间)'] = B_order_info['pay_time(支付时间)'].dt.date
B_order_info['add_time(下单时间)'] = B_order_info['add_time(下单时间)'].dt.date
B_order_info['shipping_time(发货时间)'] = B_order_info['shipping_time(发货时间)'].dt.date
B_comment['add_time(评论时间时间戳)'] = pd.to_datetime(B_comment['add_time(评论时间时间戳)'],unit='s').dt.date


# 从B_order_info中获取每一个订单的时间拼接到B_order_goods中


temp = B_order_info[['order_id(订单id)','add_time(下单时间)']]
temp.columns = ['order_id(订单号)','订单时间']
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left')


# # 构造要预测目标数据


#计算每一个商品的销量
df = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])['number(订购数量)'].agg([('number(订购数量) 日销量',sum)]).reset_index()
df['订单时间'] = pd.to_datetime(df['订单时间'])


# 缺失日期填充



#缺失日期填充
data = pd.DataFrame()
for group in df.groupby(['goods_id(商品ID)']):
    group = group[1].resample('1 D',on='订单时间').first()
    group = group.drop(['订单时间'],axis=1).reset_index()
    group['goods_id(商品ID)'] = group['goods_id(商品ID)'].fillna(group['goods_id(商品ID)'].mode()[0]).astype(int)
    group['number(订购数量) 日销量'] = group['number(订购数量) 日销量'].fillna(0)
    data = pd.concat([data,group])


# 这个这个商品这个时间节点未来7天的销量和



data['number(订购数量) 未来7日销量'] = np.sum([data.groupby(['goods_id(商品ID)'])['number(订购数量) 日销量'].shift(-i) for i in range(1,8)],axis=0)



data['订单时间'] = data['订单时间'].dt.date



data.head()


# # 特征工程-B_order_info

# province(省)	city(市)	district(区/县)这几个特征采取embedding的方式


import gensim



area_col,vec_num = 'province(省)',2
#获取order_info中每一个订单的省份
temp = B_order_info[['order_id(订单id)',area_col]]
temp.columns = ['order_id(订单号)',area_col]
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
#获取每一个商品每一天的省份
goods_date_area = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])[area_col].apply(lambda x:
                                                                         list(x.dropna().astype(int).astype(str))).reset_index()
#doc2vec向量化
corpus = []
for i, line in enumerate(goods_date_area[area_col]):
    corpus.append(gensim.models.doc2vec.TaggedDocument(line,[i]))
#50维，最小两个词
model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_num, min_count=1, epochs=100)
model.build_vocab(corpus)
doc2vec = goods_date_area[area_col].map(lambda x:model.infer_vector(x))
doc2vec = pd.DataFrame(doc2vec.values.tolist())
doc2vec.columns = [area_col+'_'+str(i) for i in range(vec_num)]
#拼接到data中
goods_date_area = pd.concat([goods_date_area,doc2vec],axis=1)
goods_date_area['订单时间'] = goods_date_area['订单时间'].dt.date
print(goods_date_area.head())
del goods_date_area[area_col]
data = data.merge(goods_date_area,on=['goods_id(商品ID)','订单时间'],how='left')



area_col,vec_num = 'city(市)',3
#获取order_info中每一个订单的省份
temp = B_order_info[['order_id(订单id)',area_col]]
temp.columns = ['order_id(订单号)',area_col]
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
#获取每一个商品每一天的省份
goods_date_area = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])[area_col].apply(lambda x:
                                                                         list(x.dropna().astype(int).astype(str))).reset_index()
#doc2vec向量化
corpus = []
for i, line in enumerate(goods_date_area[area_col]):
    corpus.append(gensim.models.doc2vec.TaggedDocument(line,[i]))
#50维，最小两个词
model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_num, min_count=1, epochs=100)
model.build_vocab(corpus)
doc2vec = goods_date_area[area_col].map(lambda x:model.infer_vector(x))
doc2vec = pd.DataFrame(doc2vec.values.tolist())
doc2vec.columns = [area_col+'_'+str(i) for i in range(vec_num)]
#拼接到data中
goods_date_area = pd.concat([goods_date_area,doc2vec],axis=1)
goods_date_area['订单时间'] = goods_date_area['订单时间'].dt.date
print(goods_date_area.head())
del goods_date_area[area_col]
data = data.merge(goods_date_area,on=['goods_id(商品ID)','订单时间'],how='left')



area_col,vec_num = 'district(区/县)',4
#获取order_info中每一个订单的省份
temp = B_order_info[['order_id(订单id)',area_col]]
temp.columns = ['order_id(订单号)',area_col]
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
#获取每一个商品每一天的省份
goods_date_area = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])[area_col].apply(lambda x:
                                                                         list(x.dropna().astype(int).astype(str))).reset_index()
#doc2vec向量化
corpus = []
for i, line in enumerate(goods_date_area[area_col]):
    corpus.append(gensim.models.doc2vec.TaggedDocument(line,[i]))
#50维，最小两个词
model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_num, min_count=1, epochs=100)
model.build_vocab(corpus)
doc2vec = goods_date_area[area_col].map(lambda x:model.infer_vector(x))
doc2vec = pd.DataFrame(doc2vec.values.tolist())
doc2vec.columns = [area_col+'_'+str(i) for i in range(vec_num)]
#拼接到data中
goods_date_area = pd.concat([goods_date_area,doc2vec],axis=1)
goods_date_area['订单时间'] = goods_date_area['订单时间'].dt.date
print(goods_date_area.head())
del goods_date_area[area_col]
data = data.merge(goods_date_area,on=['goods_id(商品ID)','订单时间'],how='left')


# 没有销量的日期填充为0


for col in ['province(省)_0', 'province(省)_1', 'district(区/县)_0', 'district(区/县)_1',
       'district(区/县)_2', 'district(区/县)_3', 'city(市)_0', 'city(市)_1','city(市)_2']:
    data[col] = data[col].fillna(0)


# ### 快递特征

# 先对快递特征进行替换，属于同一个快递的合并



shipping_replace = {'中通速递[全场默认此快递]':'中通速递',
                    '顺丰速运[部分顺丰包邮产品适用]':'顺丰速运',
                   '韵达快运':'韵达速递',
                    'EMS快递':'邮政包裹',
                    '邮政快递包裹':'邮政包裹',
                   '汇通快运':'汇通快递'}
shipping_replace_keys = shipping_replace.keys()
def get_shipping_replace(x):
    if x in shipping_replace_keys:
        return shipping_replace[x]
    else:
        return x



B_order_info['shipping_id(快递名称)'] = B_order_info['shipping_id(快递名称)'].map(lambda x:get_shipping_replace(x))


# 这里计算的是每一天每一种快递的数量


temp = B_order_info[['order_id(订单id)','shipping_id(快递名称)']]
temp.columns = ['order_id(订单号)','shipping_id(快递名称)']
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中



temp = B_order_goods[['goods_id(商品ID)','订单时间','shipping_id(快递名称)']]
temp = pd.get_dummies(temp,columns=['shipping_id(快递名称)'])
temp = temp.groupby(['goods_id(商品ID)','订单时间']).sum().reset_index()
temp.head()



data = data.merge(temp,on=['goods_id(商品ID)','订单时间'],how='left')


# 缺失的填充为0



for col in ['shipping_id(快递名称)_中通速递', 'shipping_id(快递名称)_国通快递', 'shipping_id(快递名称)_圆通速递',
       'shipping_id(快递名称)_天天', 'shipping_id(快递名称)_汇通快递','shipping_id(快递名称)_申通快递', 'shipping_id(快递名称)_邮政包裹',
       'shipping_id(快递名称)_韵达速递', 'shipping_id(快递名称)_顺丰速运']:
    data[col] = data[col].fillna(0)


# ### pay_name(支付方式)特征

# 支付方式合并为三种


def get_pay_name(x):
    if '网银' in x:
        return '网银支付'
    elif '微信' in x:
        return '微信支付'
    else:
        return '支付宝'



B_order_info['pay_name(支付方式)'] = B_order_info['pay_name(支付方式)'].astype(str).map(lambda x:get_pay_name(x))


# 计算每一天每一种支付方式的数量


temp = B_order_info[['order_id(订单id)','pay_name(支付方式)']]
temp.columns = ['order_id(订单号)','pay_name(支付方式)']
B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
temp = B_order_goods[['goods_id(商品ID)','订单时间','pay_name(支付方式)']]
temp = pd.get_dummies(temp,columns=['pay_name(支付方式)'])
temp = temp.groupby(['goods_id(商品ID)','订单时间']).sum().reset_index()
temp.head()



data = data.merge(temp,on=['goods_id(商品ID)','订单时间'],how='left')



for col in ['pay_name(支付方式)_微信支付', 'pay_name(支付方式)_支付宝','pay_name(支付方式)_网银支付']:
    data[col] = data[col].fillna(0)


# ### 是否类特征

# 每一天每一种是否类的数量


for col in ['shipping_fee(邮费)','integral_money(积分抵扣金额)','bonus(优惠券金额)','from_ad(广告位ID)',
            'discount(打折)','bonus_id(优惠券ID)']:
    B_order_info[col] = B_order_info[col].map(lambda x: 1 if x>0 else 0)
    temp = B_order_info[['order_id(订单id)',col]]
    temp.columns = ['order_id(订单号)',col]
    B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
    temp = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])[col].sum().reset_index()
    print(temp.head())
    data = data.merge(temp,on=['goods_id(商品ID)','订单时间'],how='left')
    data[col] = data[col].fillna(0)


# ### 计算分布的特征

# 计算每一天分布的均值，和，方差作为特征


for col in ['pay_time-add_time', 'shipping_tim-pay_time','goods_amount(订单总金额)','money_paid(实付金额)']:
    temp = B_order_info[['order_id(订单id)',col]]
    temp.columns = ['order_id(订单号)',col]
    temp = temp[temp[col]>=0]
    B_order_goods = B_order_goods.merge(temp,on='order_id(订单号)',how='left') #拼接到order_goods中
    temp = B_order_goods.groupby(['goods_id(商品ID)','订单时间'])[col].agg([(col+'_mean','mean'),
                                                                     (col+'_sum','sum'),
                                                                     (col+'_var','var')]).reset_index()
    print(temp.head())
    data = data.merge(temp,on=['goods_id(商品ID)','订单时间'],how='left')
    data[col+'_mean'] = data[col+'_mean'].fillna(0)
    data[col+'_sum'] = data[col+'_sum'].fillna(0)
    data[col+'_var'] = data[col+'_var'].fillna(0)


# 'market_price(市场价格)','goods_price(售价)'



data = data.merge(B_order_goods[['goods_id(商品ID)','订单时间','market_price(市场价格)','goods_price(售价)']],
                  on=['goods_id(商品ID)','订单时间'],how='left')



data['market_price(市场价格)'] = data.groupby(['goods_id(商品ID)'])['market_price(市场价格)'].fillna(method='ffill')
data['goods_price(售价)'] = data.groupby(['goods_id(商品ID)'])['goods_price(售价)'].fillna(method='ffill')


# In[ ]:


data['goods_price/market'] = data['goods_price(售价)']/data['market_price(市场价格)']


# ## 情感计算 [文本模态]



from snownlp import SnowNLP


# #计算每一条评论的情感值


B_comment['情感值'] = B_comment['content(评论内容)'].astype(str).map(lambda x:SnowNLP(x).sentiments)


# 每一个商品每一天评论的均值，数量，和



temp = B_comment.groupby(['add_time(评论时间时间戳)','goods_id(商品ID)'])['情感值'].agg([('情感值_mean','mean'),
                                                                     ('评论数量','count'),
                                                                            ('情感值_sum','sum')]).reset_index()


# 商品评论历史均值


temp['历史平均情感值'] = temp.groupby(['goods_id(商品ID)'])['情感值_sum'].cumsum()/temp.groupby(['goods_id(商品ID)'])['评论数量'].cumsum()


# 和data拼接起来



temp.columns = ['订单时间', 'goods_id(商品ID)', '情感值_mean', '评论数量', '情感值_sum','历史平均情感值']
data = data.merge(temp,on=['goods_id(商品ID)','订单时间'],how='left')


# 缺失填充



data['情感值_mean'] = data['情感值_mean'].fillna(0.5)
data['评论数量'] = data['评论数量'].fillna(0)
data['情感值_sum'] = data['情感值_sum'].fillna(0.5)
data['历史平均情感值'] = data.groupby(['goods_id(商品ID)'])['历史平均情感值'].fillna(method='ffill')




data['历史评论数量'] = data.groupby(['goods_id(商品ID)'])[ '评论数量'].cumsum()



#颜值计算 (图像模态)

def facecalculation(filepath):
    request_url = "https://api-cn.faceplusplus.com/facepp/v1/skinanalyze"
    key = "************************"  #API Key
    secret = "************************"  #API 密钥

    data = {"api_key": key, "api_secret": secret, "outer_id": 'zbpm'}
    files = {"image_file": open(filepath, "rb")}
    response = requests.post(request_url, data=data, files=files)
    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    result1 = req_dict["result"]["eye_pouch"]["value"]
    result2 = req_dict["result"]["dark_circle"]["value"]
    result3 = req_dict["result"]["forehead_wrinkle"]["value"]
    result4 = req_dict["result"]["crows_feet"]["value"]
    result5 = req_dict["result"]["eye_finelines"]["value"]
    result6 = req_dict["result"]["glabella_wrinkle"]["value"]
    result7 = req_dict["result"]["nasolabial_fold"]["value"]
    result8 = req_dict["result"]["skin_type"]["details"]["1"]["value"]
    result9 = req_dict["result"]["pores_forehead"]["value"]
    result10 = req_dict["result"]["pores_left_cheek"]["value"]
    result11 = req_dict["result"]["pores_right_cheek"]["value"]
    result12 = req_dict["result"]["pores_jaw"]["value"]
    result13 = req_dict["result"]["blackhead"]["value"]
    result14 = req_dict["result"]["acne"]["value"]
    result15 = req_dict["result"]["dark_circle"]["value"]
    result16 = req_dict["result"]["right_eyelids"]["value"]
    result17 = req_dict["result"]["skin_spot"]["value"]

    total_score = result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8 + result9 + result10 + result11 + result12 + result13 + result14 + result15 + result16 + result17
    face_value = total_score / 17

    return face_value


B_user = pd.read_csv('B_users.csv',encoding='utf-8')
B_user['颜值'] = B_user['avatar(用户头像)'].astype(str).map(lambda x:facecalculation(x))

temp.columns = ['订单时间', 'user_id(用户id)', '颜值']
data = data.merge(temp,on=['user_id(用户id)','订单时间'],how='left')
# 缺失填充
data['颜值'] = data['颜值'].fillna(0)


data.info()


# # 构建级联混合循环神经网络预测模型

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error



for col in ['number(订购数量) 日销量','shipping_id(快递名称)_中通速递', 'shipping_id(快递名称)_国通快递',
       'shipping_id(快递名称)_圆通速递', 'shipping_id(快递名称)_天天',
       'shipping_id(快递名称)_汇通快递', 'shipping_id(快递名称)_申通快递',
       'shipping_id(快递名称)_邮政包裹', 'shipping_id(快递名称)_韵达速递',
       'shipping_id(快递名称)_顺丰速运', 'pay_name(支付方式)_微信支付', 'pay_name(支付方式)_支付宝',
       'pay_name(支付方式)_网银支付', 'shipping_fee(邮费)', 'integral_money(积分抵扣金额)',
       'bonus(优惠券金额)', 'from_ad(广告位ID)', 'discount(打折)', 'bonus_id(优惠券ID)',
       'pay_time-add_time_mean', 'pay_time-add_time_sum',
       'pay_time-add_time_var', 'shipping_tim-pay_time_mean',
       'shipping_tim-pay_time_sum', 'shipping_tim-pay_time_var',
       'goods_amount(订单总金额)_mean', 'goods_amount(订单总金额)_sum',
       'goods_amount(订单总金额)_var', 'money_paid(实付金额)_mean',
       'money_paid(实付金额)_sum', 'money_paid(实付金额)_var', 'market_price(市场价格)',
       'goods_price(售价)', 'goods_price/market', '情感值_mean', '评论数量', '情感值_sum',
       '历史平均情感值', '历史评论数量']:
    data[col] =  (data[col]-data[col].min())/(data[col].max()-data[col].min())




data = data.fillna(0)


# 划分训练集测试集



step = 10
X = []
y = []
y_time = []
for group in data.groupby(['goods_id(商品ID)']):
    m,n = group[1].shape
    if m>step+7:
        for i in range(step,m-8):
            X.append(group[1].iloc[i-step:i+1,[2]+list(range(4,n))].values)
            y.append(group[1].iloc[i,3])
            y_time.append(group[1].iloc[i,0])



y_time = pd.Series(y_time)


y_time = y_time.sort_values()


X = np.array(X)
y = np.array(y)


X = X[pd.Series(y).notnull()]


y_time = y_time[pd.Series(y).notnull()]


y_time.index = range(len(y_time))


y = y[pd.Series(y).notnull()]


X_train = X[list(y_time[:-20000].index)]
y_train = y[list(y_time[:-20000].index)]


X_val = X[list(y_time[-20000:-10000].index)]
y_val = y[list(y_time[-20000:-10000].index)]


X_test = X[list(y_time[-10000:].index)]
y_test = y[list(y_time[-10000:].index)]


train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32),y_train.astype(np.float32)))
val_ds = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32),y_val.astype(np.float32)))
test_ds = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32),y_test.astype(np.float32)))
train_ds = train_ds.shuffle(256).batch(256)
val_ds = val_ds.shuffle(256).batch(256)


# 模型


model = Sequential()
model.add(Bidirectional(LSTM(16,return_sequences=True)))
model.add(Bidirectional(GRU(16,return_sequences=True)))
model.add(Bidirectional(LSTM(16,return_sequences=True)))
model.add(Bidirectional(GRU(16,return_sequences=True)))
model.add(Bidirectional(LSTM(16,return_sequences=True)))
model.add(Bidirectional(GRU(16,return_sequences=False)))

model.add(Dense(2))
model.compile(optimizer='adam',
              loss='mae',
              metrics='mae'
             )
early_stop = EarlyStopping(monitor='val_loss', patience=20) #设置早停得到最优结果
#开始训练模型
history = model.fit(train_ds,
          epochs=500,
          validation_data=val_ds,
          callbacks=[early_stop],
          verbose=1
          #validation_freq=1
         )



def MSE(y_true, y_pred):
     return mean_squared_error(y_true, y_pred)
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
def score(y_true, y_pred):
    return [MSE(y_true, y_pred),MAE(y_true, y_pred),RMSE(y_true, y_pred)]



y_pred_lstm = model.predict(X_test)[:,0]
pred_score = score(y_test, y_pred_lstm)



print('MSE:{:.4f},MAE:{:.4f},RMSE:{:.4f}'.format(pred_score[0],pred_score[1],pred_score[2]))
