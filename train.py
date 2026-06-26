import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import joblib

url='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

df=pd.read_csv(url)


df['Age']=df['Age'].fillna(df['Age'].median())

le=LabelEncoder()

df['Sex']=le.fit_transform(df['Sex'])

X=df[['Age','Pclass','Fare','Sex']]
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train_scaler=scaler.fit_transform(X_train)

X_test_scaler=scaler.transform(X_test)

model=LogisticRegression(max_iter=200)

model.fit(X_train_scaler,y_train)
prediction=model.predict(X_test_scaler)
print("acctual outcome:",y_test.values)
print("prediction :",prediction)

print("\n accuracy :",accuracy_score(y_test,prediction))
print("\n confusion matrix:",confusion_matrix(y_test,prediction))

probability=model.predict_proba(X_test_scaler)
print('\n probability:',probability)

joblib.dump(model,'models/titanic_model.joblib')
joblib.dump(scaler,'models/titanic_scaler.joblib')
joblib.dump(le,'models/titanic_label.joblib')