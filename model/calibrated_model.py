
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import confusion_matrix, classification_report


reports_df = pd.read_csv("data.csv")
#reports_df.head()

#'''overview of the data'''

reports_df['RESULT'].unique()

#'''listing unique classes'''

reports_filtered_df = reports_df[pd.notnull(reports_df['REPORTS'])]
reports_filtered_df.info()

#'''''''''''''overview of the input dataset after removing null rows
fig = plt.figure(figsize = (10,6))
df = reports_filtered_df[['RESULT', 'REPORTS']]
df.groupby('RESULT').count().plot.bar(ylim=0)
plt.show()

#'''distirbution of classes in the dataset'''

labels = df['RESULT']
text = df['REPORTS']

X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.3)

count_vect = TfidfVectorizer()
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)


print(count_vect.get_feature_names())
#print(X_train_counts.toarray())
#print(X_train_counts.shape)

tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_transformed = tf_transformer.transform(X_test_counts)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X_train_counts)

# print idf values
#df_idf = pd.DataFrame(tfidf_transformer.idf_, index = count_vect.get_feature_names(),columns=["tf_idf_weights"])
 
# sort ascending
#df_idf.sort_values(by=['tf_idf_weights'])

 
# count matrix
count_vector = count_vect.transform(text)
 
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)

feature_names = count_vect.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)


# get class label
labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_labels_trf = labels.transform(y_train)

print(labels.classes_)

#'''preprocessing input data'''

linearsvc = LinearSVC()
clf = linearsvc.fit(X_train_transformed, y_train_labels_trf)

calibrated_svc = CalibratedClassifierCV(base_estimator = linearsvc, cv = "prefit", method = "sigmoid")

calibrated_svc.fit(X_train_transformed, y_train_labels_trf)
predicted = calibrated_svc.predict(X_test_transformed)

to_predict = ["VARIABLES"]
p_count = count_vect.transform(to_predict)
p_tfidf = tf_transformer.transform(p_count)

#print('Average accuracy on test set = {}'. format(np.mean(predicted == labels.transform(y_test))))

print('Predicted probabilities of the input string are')
print(calibrated_svc.predict_proba(p_tfidf))


#'''''''''''''''Training using classifier'''''''''''

pd.DataFrame(calibrated_svc.predict_proba(p_tfidf)*100, columns = labels.classes_)

#'''''''''''''Prediction''''''''''''''''


report = classification_report(labels.transform(y_test), predicted, digits=4)
print (report)


def get_confusion_matrix_values(y_test, predicted):
    cm = confusion_matrix(labels.transform(y_test), predicted)
    return(cm[1][1], cm[1][2], cm[2][1], cm[2][2])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, predicted)
print("True Positive = ",TP)
print("False Positive = ",FP)
print("False Negative = ",FN)
print("True Negative = ",TN)

sensitivity  = (TP / (TP+FN)) * 100
specificity  = (TN / (TN+FP)) * 100
pos_pred_val = (TP/ (TP+FP)) * 100
neg_pred_val = (TN/ (TN+FN)) * 100
accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 
fscore = 2 * ((pos_pred_val * sensitivity) / (pos_pred_val + sensitivity))


print("Sensitivity  = ", sensitivity)
print("Specificity = ", specificity)
print("Positive Predictive Value = ", pos_pred_val)
print("Negative Predictive Value = ", neg_pred_val)
print("Accuracy = ", accuracy)
print("F1-Score = ", fscore)


cm = confusion_matrix(labels.transform(y_test), predicted)
cm_df = pd.DataFrame(cm, index = [labels.classes_], columns = [labels.classes_])

plt.figure(figsize = (5.5, 4))
sns.heatmap(cm_df, annot = True)
plt.title('Contigency Table \n Accuracy: {0:4f}'.format(accuracy))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


from sklearn.externals import joblib
joblib.dump(clf,'pipedcalibrated.pkl')