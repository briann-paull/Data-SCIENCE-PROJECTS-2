#importing api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#reading data
data=pd.read_csv("c.csv")


#looking for nul value
print(data.isnull().any())


heattmap=data.corr()



data_columnss=data.columns.to_list()
data_columnss

#cal target unique value
unique=data["default.payment.next.month"].value_counts(sort=True)

#ploting unique value

unique.plot(kind="bar",rot=0)
plt.title="0/1"
plt.show()



#undersampling data


x=data.iloc[:,:24]
y=data.iloc[:,-1]

from imblearn.under_sampling import NearMiss

samp=NearMiss()
x_n,y_n=samp.fit_sample(x, y)
print(x_n.shape)
print(y_n.shape)

unique2=y_n.value_counts(sort=True)

from collections import Counter

print("Original Data",format(Counter(y)))
print("resample Data",format(Counter(y_n)))



unique2.plot(kind="bar",rot=0)
plt.title="0/1 in undersampling"
plt.show()

dataset=pd.concat([x_n,y_n],axis=1)



#preprocessing data

dataset.rename(columns={"PAY_0":"PAY_1"},inplace=True)

print(dataset["EDUCATION"].value_counts(sort=True))
print(dataset["MARRIAGE"].value_counts(sort=True))

dataset['EDUCATION']=dataset['EDUCATION'].map({1:1,2:2,3:3,4:4,5:4,6:4,0:4})
dataset['MARRIAGE']=dataset['MARRIAGE'].map({1:1,2:1,3:2,0:2})


print(dataset["EDUCATION"].value_counts(sort=True))
print(dataset["MARRIAGE"].value_counts(sort=True))

df=dataset

from sklearn.preprocessing import MinMaxScaler
min=MinMaxScaler()
df=min.fit_transform(df)


x=data.iloc[:,:24].values
y=data.iloc[:,-1].values

'''plt.plot(x,y,"ro")
plt.show()'''

#encoding minmxscaler




from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold()
 
for train_index, test_index in folds.split(x,y):
    X_train, X_test, y_train, y_test = x[train_index], x[test_index],y[train_index], y[test_index]

    
    
from sklearn.model_selection import cross_val_score


print("LogistRegressionScore")
score0=cross_val_score(LogisticRegression(solver='saga', max_iter=1000000),X_train,y_train,cv=3)
print(score0)
print(np.average(score0))

print("SVC Score")
score1=cross_val_score(SVC(),X_train,y_train,cv=3)
print(score1)
print(np.average(score1))


print("NavieBayes Score")
score2=cross_val_score(GaussianNB(),X_train,y_train,cv=3)
print(score2)
print(np.average(score2))


print("Random_Forest Score ")
score5=cross_val_score(RandomForestClassifier(n_estimators=15),X_train,y_train,cv=5)
print(score5)
print(np.average(score5))

print("XGB Sore")
score6=cross_val_score(XGBClassifier(learning_rate=0.1,n_estimators=8,random_state=4),X_train,y_train,cv=8)
print(score6)
print(np.average(score6))


model=XGBClassifier()
fit=model.fit(X_train,y_train)

y_pred=fit.predict(x_test)






















