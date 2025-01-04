from flask import Flask
from flask import render_template, request
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib

data=pd.read_csv('Churn_Modelling.csv')

data=data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)
data = pd.get_dummies(data,drop_first=True)

X=data.drop('Exited',axis=1)
y=data['Exited']

X_res,y_res=SMOTE().fit_resample(X,y)
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)
sc = StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)

svm=svm.SVC()
svm.fit(X_train,y_train)

rf=RandomForestClassifier()
X_res,y_res=SMOTE().fit_resample(X,y)
X_res=sc.fit_transform(X_res)
rf.fit(X_res,y_res)

joblib.dump(rf,'churn_predict_model')
model=joblib.load('churn_predict_model')

app = Flask(__name__)

@app.route("/" , methods=["GET" , "POST"])
def hello_world():
    if request.method == "POST":
        creditScore = request.form["creditscore"]
        age = request.form["age"]
        tenure = request.form["tenure"]
        balance = request.form["balance"]
        no_of_products = request.form["no_of_products"]
        hasCrCard = request.form["hasCrCard"]
        isActiveMember = request.form["isactivemember"]
        estimatedsalary = request.form["estimatedsalary"]
        geography = request.form["geography"]
        gender = request.form["gender"]
        prediction = model.predict([[creditScore, age , tenure , balance , no_of_products , hasCrCard , isActiveMember , estimatedsalary , geography , gender,0]])[0]
        return render_template("prediction.html" , number = prediction)
    return render_template('prc.html')