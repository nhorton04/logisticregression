import pandas as pd
import matplotlib as plt
import seaborn as sbn

df = pd.read_csv("Titanic.csv")

df["family_size"] = df["SibSp"] + df["Parch"] + 1
df["Age"].fillna(value=df["Age"].median(), inplace=True)
df.isnull().sum()
df["Embarked"].fillna(value="S", inplace=True)
df.isnull().sum()

embarked ={"S":0, "C":1, "Q":2}
df.Embarked = [embarked[item] for item in df.Embarked]
gender ={"female":1, "male":0}
df.Sex = [gender[item] for item in df.Sex]

dfx = df[["Age", "Sex", "family_size", "Embarked", "Fare"]].copy(deep=True)
dfy = df[["Survived"]].copy(deep=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
x = scaler.fit_transform(dfx)

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1)

y_train = np.array(y_train).flatten()
model.fit(x_train, y_train)

ypred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)

from sklearn.metrics import accuracy_score
print("Base rate accuracy is: %0.2f" %(accuracy_score(y_test, ypred)))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

logit_roc_auc = roc_auc_score(y_test, ypred)
print("Logistic AUC = %0.2f" %logit_roc_auc)
print(classification_report(y_test, ypred))

from sklearn.metrics import roc_curve
b = model.predict_proba(x_test)[:,1]
print(b[0:5])
fpr, tpr, threshold = roc_curve(y_test, b)

# plotting ROC curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %logit_roc_auc)
plt.plot([0,1], [0,1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
