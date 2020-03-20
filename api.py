from flask import Flask,jsonify,request,json
import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from flask_cors import CORS
import json
import pyodbc

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=sqldev.centralindia.cloudapp.azure.com;'
                      'Database=DSX_MOCP_Dev;'
                      'UID=DSXSaaS;'
                      'PWD=M7uHeNT3dgkGwaj5FVFU;'
                      )
moData = pd.read_sql_query('''SELECT * FROM vwRouteData order by item,noofdays,routing  ''',conn)
moData['Start Date'] =  pd.to_datetime(moData['Start Date'])
moData['Due Date'] =  pd.to_datetime(moData['Due Date'])
moData['diff_days'] =  moData['Due Date'] - moData['Start Date']
moData['diff_days'] =  moData['diff_days']/np.timedelta64(1,'D')
df_model = moData.copy()
scaler = StandardScaler()
features = [['Quantity', 'Capacity', 'Item', 'diff_days']]
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])
y = df_model.iloc[:,-3].values
X = df_model.iloc[:,[1,2,7,8]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=2) 

app = Flask(__name__)
CORS(app)


@app.route('/prdiction',methods=['POST','GET'])
def KNeighborPrediction():
    if request.method == 'POST':
        data = request.get_json()
        result = prdiction(data['data'])
        return jsonify(result)
    else:
        return jsonify({"result":"test"})

def prdiction(value):
    value = np.array([value])
    kn = KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=5) 
    kn.fit(X_train, y_train) 
    prediction = kn.predict(value)
    return json.dumps({"data":prediction[0]})

if __name__ == '__main__':
    app.run(debug=True) 