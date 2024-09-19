# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

df =pd.read_csv('data.csv')

X=df['REPORTS']
y=df['RESULT']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#testing for each of the algorithm

text_clf1 = Pipeline([('tfidf',TfidfVectorizer()), ('clf', LinearSVC())])

text_clf1.fit(X_train,y_train)
prediction1 = text_clf1.predict(X_test)


cm = confusion_matrix(y_test,prediction1)
print(cm)

def get_confusion_matrix_values(y_test,prediction1):
    cm = confusion_matrix(y_test,prediction1)
    return(cm[2][2], cm[1][2], cm[2][1], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test,prediction1)
print("True Negative = ",TN)
print("False Positive = ",FP)
print("False Negative = ",FN)
print("True Positive = ",TP)

sensitivity  = (TP / (TP+FN)) * 100
specificity  = (TN / (TN+FP)) * 100
pos_pred_val  = (TP/ (TP+FP)) * 100
neg_pred_val = (TN/ (TN+FN)) * 100
accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 
fscore = 2 * ((pos_pred_val * sensitivity) / (pos_pred_val + sensitivity))


print("Sensitivity / Recall = ", sensitivity)
print("Specificity = ", specificity)
print("Positive Predictive Value / Precision = ", pos_pred_val)
print("Negative Predictive Value = ", neg_pred_val)
print("Accuracy = ", accuracy)
print("F1-Score = ", fscore)

#trying out cross validation score

accuracies = cross_val_score(estimator=text_clf1, X=X_train, y= y_train, cv =10)
accuracies.mean()

report = classification_report(y_test,prediction1, digits=4)
print (report)


Accuracy = accuracy_score(y_test, prediction1, normalize=True, sample_weight=None)
#print("Accuracy for Linear Support Vector Classifier: " + str(Accuracy))

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_labels_trf = labels.transform(y_train)

print(labels.classes_)

cm = confusion_matrix(y_test, prediction1)
cm_df = pd.DataFrame(cm, index = [labels.classes_], columns = [labels.classes_])


plt.figure(figsize = (5.5, 4))
sns.heatmap(cm_df, annot = True)
plt.title('Confusion Matrix for Cardiomegaly Detection classification model\n Accuracy:{0:4f}'.format(accuracy))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#print(text_clf1.predict(['There cardiac enlargement with CTR of 0.44. The aorta is unfolded. The hilar are congested. Both hemidiaphragm and pleural recesses are preserved. Is the patient hypertensive? Degenerative changes are noted in the thoracic vertebrae. ']))

#dump the model
from sklearn.externals import joblib
joblib.dump(text_clf1,'noncalibrated.pkl')