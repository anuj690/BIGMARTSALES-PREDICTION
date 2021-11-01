import numpy as np
import pandas as pd
import sklearn
import xgboost
import pickle
import joblib
from scipy.stats import mode
train = pd.read_csv(r"C:\Users\hp\.jupyter\sales\Train.csv")
df = train.copy()
median=df["Item_Weight"].median()
df["Item_Weight"].fillna(median, inplace=True)

frequent=df['Outlet_Size'][764]
df['Outlet_Size'].fillna(frequent,inplace=True)

df['Item_Visibility']=df['Item_Visibility'].replace(0,np.median(df['Item_Visibility']))
df['Item_Visibility_sqrt']=np.sqrt(df['Item_Visibility'])
    
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['Item_Type_Combined'] =df['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
    
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

df['Item_Fat_Content']=df['Item_Fat_Content'].replace('LF','Low Fat')
df['Item_Fat_Content']=df['Item_Fat_Content'].replace('reg','Regular')
df['Item_Fat_Content']=df['Item_Fat_Content'].replace('low fat','Low Fat')
    

df.loc[df['Item_Type_Combined'] == "Non-Consumable", "Item_Fat_Content"] = "Non-Edible"
df['price_per_weight']=df['Item_MRP']//df["Item_Weight"] 
df['price_per_weight_unt']=np.sqrt(df['price_per_weight'])


Ordinal_dict={'Small':0,'Medium':1,'High':2}
df['Outlet_Encoded']=df.Outlet_Size.map(Ordinal_dict)
   

Order_dict={'Tier 3':0,'Tier 2':1,'Tier 1':2}
df['Outlet_Location_Type']=df.Outlet_Location_Type.map(Order_dict)
    

df.loc[df['Item_MRP'] <= 70, 'Item_MRP'] = 1
df.loc[(df['Item_MRP'] > 70) & (df['Item_MRP'] <= 140), 'Item_MRP'] = 2
df.loc[(df['Item_MRP'] > 140) & (df['Item_MRP'] <= 210), 'Item_MRP']   = 3
df.loc[ df['Item_MRP'] > 210, 'Item_MRP'] = 4
df['Item_MRP'] = df['Item_MRP'].astype(int)
    
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df[['Outlet_Type']]=oe.fit_transform(df[['Outlet_Type']]).astype(int)
df = pd.get_dummies(df, columns=['Item_Fat_Content','Item_Type_Combined'],drop_first = True)
df['Item_Outlet_Sales']=np.sqrt(df['Item_Outlet_Sales'])
features_drop=["Item_Identifier","Item_Type","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size",
               "Item_Visibility","price_per_weight"]
df=df.drop(features_drop,axis=1)

y = df.Item_Outlet_Sales.values
X = df.drop('Item_Outlet_Sales',axis = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_prepared = sc.fit_transform(X)
from xgboost.sklearn import XGBRegressor
from xgboost import XGBRegressor
model = XGBRegressor(learning_rate =0.1,n_estimators=100, max_depth=4,min_child_weight=6,subsample=0.8,
                     colsample_bytree=0.9,random_state=42 )
model.fit(X_prepared,y)


joblib.dump(model,r'C:\Users\hp\BIGMARTSALE\MODEL\XGBOOST.pkl')


    
    

