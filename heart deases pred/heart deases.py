#importing api's


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("heart_d.csv")


#looking up for null value
print(data.isnull().any())

#looking up for target ratio
Target_ration=data["target"].value_counts(sort=True)
print(Target_ration)


#analyzing data 

heatmap=data.corr()
sns.heatmap(heatmap,annot=True)
plt.show()



#encoding categorical data

dataset = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],drop_first=True)




#encoding minmaxscaler cause we dont know whetr the cure is normally distrubted

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])





#spliting up variable
y=dataset.iloc[:,5].values
x=dataset.drop(["target"],axis=1).values



#analyzing target variable

plt.plot(x,y,"ro")
plt.show()


'''


#find variable corelation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

F=SelectKBest(score_func=chi2,k=2)

best=F.fit(x,y)

Feature_column=pd.DataFrame(x.columns)
Score=pd.DataFrame(best.scores_)

Best_feature=pd.concat([Feature_column,Score],axis=1)
Best_feature.columns=['COLUMNS','SCORES']
print(Best_feature.nlargest(13,"SCORES"))'''






#import models api


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier




#using kfold to selct best model

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold()
 
for train_index, test_index in folds.split(x,y):
    X_train, X_test, y_train, y_test = x[train_index], x[test_index],y[train_index], y[test_index]




#cross val with average score
from sklearn.model_selection import cross_val_score

print("Logistic Reg score")
score0=cross_val_score(LogisticRegression(), X_train,y_train)
print(np.average(score0))

print("SVC score")
score1=cross_val_score(SVC(),X_train,y_train)
print(np.average(score1))

print("Naive bayes score")
score2=cross_val_score(GaussianNB(), X_train,y_train)
print(np.average(score2))

print("random forest score")
score3=cross_val_score(RandomForestClassifier(n_estimators=40),X_train,y_train,cv=12)
print(np.average(score3))

print("xgboost score")
score4=cross_val_score(XGBClassifier(),X_train,y_train)
print(np.average(score4))

#predicting model with logistic regression(higest score)

model=LogisticRegression()
fit=model.fit(X_train,y_train)
y_pred=fit.predict(X_test)



from sklearn.model_selection import RandomizedSearchCV
model2=GaussianNB()
grid = RandomizedSearchCV(model2, params)
