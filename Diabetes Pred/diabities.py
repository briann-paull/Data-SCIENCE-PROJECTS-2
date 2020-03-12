#importing modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading data

data=pd.read_csv("d.csv")

print(data.head(5))



#BOLL TO INT
diabetes_map={True:1,False:0}
data["diabetes"]=data["diabetes"].map(diabetes_map)

#lookup for null value
print(data.isnull().any())


#look up for 0 values
total_zeros=(data.isin([0]).sum())
                      
'''
#lookup for zero value   METHOD_
print("number of rows missing in diab_pred",pd.value_counts(data['diab_pred']==0))
print("number of misssing rows in glucose_conc",pd.value_counts(data['glucose_conc']==0))
print("number of missing row in diastolic_bp ",pd.value_counts(data['diastolic_bp']==0))
print("number of missing row in thickness",pd.value_counts(data["thickness"]==0))
print("number of rows missing in insulin",pd.value_counts(data['insulin']==0))
print("number of misssing rows in bmi",pd.value_counts(data['bmi']==0))
print("number of missing row in age",pd.value_counts(data["age"]==0))
print("number of missing row skin ",pd.value_counts(data["skin"]==0))'''




#replace zero with nan
xndata=data.iloc[:,:9].replace(0,np.NaN)
yndata=data["diabetes"]

df=pd.concat([xndata,yndata],axis=1)


#deleting rows with threshold of more then 5 value and resting index

df=df.dropna(thresh=6).reset_index(drop=True)


#filling 0 value with mean

df=df.fillna(df.mean())
#df["diab_pred"].fillna(df["diab_pred"].mean(),inplace=True)...to print for specific column



#to see unique value of output variable


unique=yndata.value_counts(sort=True)
print(unique)



#plotting imbalance data

unique.plot(kind="bar",rot=0)
plt.title("0/1")
plt.show()




#finding coreelation of variables

cor=df.corr()
sns.heatmap(df.corr(),annot=True)
plt.show()



#spliting variable into dependend and independent variables

x=df.iloc[:,:9]
y=df.iloc[:,-1]


#feature selection for model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

top=SelectKBest(score_func=chi2,k=0)
fit=top.fit(x,y)

fc=pd.DataFrame(x.columns)
fs=pd.DataFrame(fit.scores_)
feature_scoring=pd.concat([fc,fs],axis=1)
feature_scoring.columns=["feature","scores"]

top7=feature_scoring.nlargest(10,"scores")
print(top7)




#spliting up new kbest fit variables

col=["insulin","glucose_conc","age","bmi","thickness","num_preg","diastolic_bp"]
X=x[col].values

Y=y





#handling impure dataset with over sampling


from imblearn.combine import SMOTETomek

smk=SMOTETomek()

X_res,y_res=smk.fit_sample(X,Y)

unique2=y_res.value_counts(sort=True)


unique2.plot(kind="bar",rot=0)
plt.title("1/0")
plt.show()


#using Kfold to select best model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier




#using kfold to selct best model

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold()
 
for train_index, test_index in folds.split(X_res,y_res):
    X_train, X_test, y_train, y_test = X_res[train_index], X_res[test_index], \
                                       y_res[train_index], y_res[test_index]
   


#cross val
from sklearn.model_selection import cross_val_score

print("cross_liR_score")
score0=cross_val_score(LinearRegression(), X_train,y_train,cv=3)
print(np.average(score0))

print("cross_lR_score")
score1=cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'),X_train,y_train,cv=3)
print(np.average(score1))

print("cross_svc_score")
score2=cross_val_score(SVC(gamma='auto'), X_train,y_train,cv=3)
print(np.average(score2))

print("cross_RF_score")
score3=cross_val_score(RandomForestClassifier(n_estimators=40),X_train,y_train,cv=4)
print(np.average(score3))

#predicting model
model=RandomForestClassifier()
fit=model.fit(X_train,y_train)

y_pred=fit.predict(X_test)
print(fit.predict_proba(X_test))

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

#traning through ANN



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 15, init = 'he_uniform',activation='relu',input_dim = 7))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 20, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,batch_size = 1, nb_epoch = 5)


print(classifier.evaluate(X_test, y_test))#loss and accuracy

ypANN = classifier.predict(X_test)




from sklearn.metrics import accuracy_score
score=accuracy_score(ypANN,y_test)




















      