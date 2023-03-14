import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('D:/Uni/Masters/Python/IBMMachineLearning/Course 3/FinalProj/emails.csv')
print(df.head())
print(df.shape) #5172 rows and 3002 columns 
#Check for null 
print(df.isnull().sum().sum()) #no null values 

#Check if predictions is imbalanced 
print(df['Prediction'].value_counts(normalize=True))
#0.709977 for class 0 
#0.290023 for class 1

rs = 42 
#feature cols 
feature_cols = df.columns.drop('Prediction') 
X = df[feature_cols] 
y = df['Prediction'] 

#train, test, split 
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = rs, test_size = 0.3, stratify = y) 
print(y_train.value_counts(normalize = True)) #Same distribution
print(y_test.value_counts(normalize = True)) #Same distribuion 

#Model 1: Logistic Regression (Original, Class weights, Over and under sampling)
LR = LogisticRegression(random_state = rs, max_iter = 1000) 
LR = LR.fit(X_train, y_train) 
y_pred = LR.predict(X_test) 

print('For Normal Linear Regression: ') 
print('--Accuracy is: ', accuracy_score(y_test, y_pred)) #0.971 
print('--Recall is: ', recall_score(y_test, y_pred)) #0.951
print('--Precision is: ', precision_score(y_test, y_pred)) #0.949
print('--F1 score is: ', f1_score(y_test, y_pred)) #0.95
print('--AUC score is: ', roc_auc_score(y_test, y_pred)) #0.965

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(cm, annot = True, fmt = 'd', annot_kws={'size':40, 'weight':'bold'})

labels = ['Spam', 'Not Spam']
ax.set_xticklabels(labels, fontsize=25)
ax.set_yticklabels(labels, fontsize = 25) 
ax.set_ylabel('Ground Truth', fontsize = 30)
ax.set_xlabel('Prediction', fontsize = 30)
plt.show()


#For KNN neighbors, we need to first scale our data 
min_max = MinMaxScaler()
X_train_minmax = min_max.fit_transform(X_train)
X_test_minmax = min_max.transform(X_test)
X_test_minmax

max_k = 40 
f1_scores = list() 
error_rates = list() 
for k in range(1,max_k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(X_train_minmax, y_train)
    y_pred = knn.predict(X_test_minmax)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(((k,round(f1,4))))
    accuracy = accuracy_score(y_test,y_pred)
    error_rates.append((k, 1-accuracy))
f1_results = pd.DataFrame(f1_scores, columns = ['K', 'F1 score'])
error_results = pd.DataFrame(error_rates, columns = ['K', 'Error Rates'])

print(f1_results)
print(error_results)

plt.figure(dpi = 300) 
ax = f1_results.set_index('K').plot(figsize = (10,10), linewidth=6)
ax.set(xlabel = 'K', ylabel='F1 score')
ax.set_xticks(range(1,max_k,2))
plt.title('KNN F1 score') 
plt.show() 

plt.figure(dpi=300)
ax = error_results.set_index('K').plot(figsize=(10,10), linewidth=6) 
ax.set(xlabel = 'K', ylabel='Error Results')
ax.set_xticks(range(1,max_k,2))
plt.title('KNN Error Results') 
plt.show()

knn_best = KNeighborsClassifier(n_neighbors=1)
knn_best.fit(X_train_minmax, y_train) 
y_pred = knn_best.predict(X_test_minmax)
print('For KNN Classification: ') 
print('--Accuracy is: ', accuracy_score(y_test, y_pred)) 
print('--Recall is: ', recall_score(y_test, y_pred)) 
print('--Precision is: ', precision_score(y_test, y_pred)) 
print('--F1 score is: ', f1_score(y_test, y_pred)) 
print('--AUC score is: ', roc_auc_score(y_test, y_pred)) 

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cm, annot = True, fmt = 'd', annot_kws={'size':40, 'weight':'bold'})

labels = ['Not Spam', 'Spam']
ax.set_xticklabels(labels, fontsize=25)
ax.set_yticklabels(labels, fontsize = 25) 
ax.set_ylabel('Ground Truth', fontsize = 30)
ax.set_xlabel('Prediction', fontsize = 30)
plt.show()

#DecisionTree 
dt = DecisionTreeClassifier(random_state=rs, max_depth = 10, max_features=60)
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('For Decision Tree Classification: ') 
print('--Accuracy is: ', accuracy_score(y_test, y_pred)) 
print('--Recall is: ', recall_score(y_test, y_pred)) 
print('--Precision is: ', precision_score(y_test, y_pred)) 	
print('--F1 score is: ', f1_score(y_test, y_pred)) 
print('--AUC score is: ', roc_auc_score(y_test, y_pred)) 

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cm, annot = True, fmt = 'd', annot_kws={'size':40, 'weight':'bold'})

labels = ['Not Spam', 'Spam']
ax.set_xticklabels(labels, fontsize=25)
ax.set_yticklabels(labels, fontsize = 25) 
ax.set_ylabel('Ground Truth', fontsize = 30)
ax.set_xlabel('Prediction', fontsize = 30)
plt.show()


