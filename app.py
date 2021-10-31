from  flask import Flask,render_template,request,jsonify
import os
import joblib
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open('models.pkl', 'rb'))
app = Flask(__name__,template_folder='templates')

@app.route("/")
def index():
    return render_template('Sales.html')


@app.route('/predict', methods=['POST','GET'])
def result():
    
    
    Item_Weight = float(request.form.get('Item_Weight',False))
    price_per_weight_unt = float(request.form.get('price_per_weight_unt',False))

    Item_Fat_Content = request.form.get('Item_Fat_Content',False)

    if (Item_Fat_Content == 'Low Fat'):
        Item_Fat_Content = 0, 0
    elif (Item_Fat_Content == 'Regular'):
        Item_Fat_Content = 0, 1
    else:
        Item_Fat_Content = 1, 0

    Item_Fat_Content_1,Item_Fat_Content_2 = Item_Fat_Content

    Item_Visibility_sqrt = float(request.form.get('Item_Visibility_sqrt',False))

    Outlet_Years = int(request.form.get('Outlet_Years',False))

    Item_MRP =  float(request.form.get('Item_MRP',False))

    if (Item_MRP <=70 ):
        Item_MRP = 1
    elif (Item_MRP >70 & Item_MRP <=140):
        Item_MRP = 2
    elif(Item_MRP > 140 & Item_MRP <= 210):
        Item_MRP=3
    else:
        Item_MRP=4


    Outlet_Encoded = request.form.get('Outlet_Encoded',False)
    if (Outlet_Encoded == 'Medium'):
        Outlet_Encoded = 1
    elif (Outlet_Encoded == 'Small'):
        Outlet_Encoded = 0
    else:
        Outlet_Encoded = 2



    Outlet_Location_Type = request.form.get('Outlet_Location_Type',False)
    if (Outlet_Location_Type == 'Tier 2'):
        Outlet_Location_Type = 1
    elif (Outlet_Location_Type == 'Tier 3'):
        Outlet_Location_Type = 0
    else:
        Outlet_Location_Type = 2


    Outlet_Type = request.form.get('Outlet_Type',False)
    if (Outlet_Type == 'Supermarket Type1'):
        Outlet_Type = 1
    elif (Outlet_Type == 'Grocery Store'):
        Outlet_Type = 0
    elif (Outlet_Type == 'Supermarket Type3'):
        Outlet_Type = 3
    else:
        Outlet_Type = 2



    Item_Type_Combined = request.form.get('Item_Type_Combined',False)

    if (Item_Type_Combined == "Drinks"):
        Item_Type_Combined = 0, 0
    elif (Item_Type_Combined == "Food"):
        Item_Type_Combined = 1, 0
    else:
        Item_Type_Combined = 0, 1

    Item_Type_Combined_1, Item_Type_Combined_2 = Item_Type_Combined

    data=np.array([Item_Weight, Item_Visibility_sqrt, Item_MRP, Outlet_Years, Item_Fat_Content_1,Item_Fat_Content_2,
            Outlet_Location_Type,Outlet_Encoded ,Outlet_Type, Item_Type_Combined_1, Item_Type_Combined_2,price_per_weight_unt])




    features_value = [np.array(data)]

    features_name = ['Item_Weight', 'Item_Visibility_sqrt', 'Item_MRP', 'Outlet_Years', 'Item_Fat_Content_1','Item_Fat_Content_2',
            'Outlet_Location_Type','Outlet_Encoded' ,'Outlet_Type', 'Item_Type_Combined_1', 'Item_Type_Combined_2','price_per_weight_unt']

    
    df = pd.DataFrame(features_value, columns=features_name)
    #model_path= r'C:\Users\hp\BIGMARTSALE\MODEL\XGBOOST.pkl'
    #model= joblib.load(r'C:\Users\hp\BIGMARTSALE\MODEL\XGBOOST.pkl')


    Y_pred = model.predict(df)
    return render_template('result.html',prediction = Y_pred)
if __name__ == '__main__':
    app.run(debug=True)

