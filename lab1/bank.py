import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv('C:/Users/HOANG NAM/Desktop/bank/bank-full.csv',sep=';')
df.head(10)

df.describe()

#check null in col
nulls = []
for i in df.columns:
    nulls.append(df[i].isnull().sum())
print(nulls)
#xóa cột duration
df = df.drop(['duration'],axis=1)
#Ciu cột là biến sô => 
temp_day = []
for i in df['day']:
    if i<8:
        temp_day.append("start")
    elif i>23:
        temp_day.append("end")
    else:
        temp_day.append("middle")

df['day'] = temp_day

# hist
plt.style.use('seaborn-whitegrid')
df.hist(bins=20, figsize=(14,10), color='#E14906')
plt.show()


#sắp giảm campaign
df.nlargest(20,"campaign")


#populate a list with string-type columns
str_columns = []
for i in df.columns:
    if(type(df[i][0])==str):
        str_columns.append(i)

#đồ thị
fig, axes = plt.subplots(2,2)
fig.tight_layout(h_pad=2)

bin_list = [1,2,3,4,5,6,7,8,9,10]

for i in range(1,11):
    plt.subplot(5, 2, i)
    n = bin_list[i-1]
    counts = df[str_columns[n]].value_counts()
    plt.barh(y=counts.index, width=counts)
    plt.title('Distribution of '+str_columns[n])
    plt.ylabel(str_columns[n])
    plt.xlabel('Count')

plt.subplots_adjust(left=.25, bottom=5, right=2.75, top=10)

#chuyển no -> 0 , yes -> 1
df['default'] = [0 if i == "no" else 1 for i in df['default']]
df['housing'] = [0 if i == "no" else 1 for i in df['housing']]
df['loan']    = [0 if i == "no" else 1 for i in df['loan']]
df['y'] = [0 if i == "no" else 1 for i in df['y']]


#xóa 4 biến ở trên trong str_columns
for i in ['default','housing','loan','y']:
    str_columns.remove(i)    
print(str_columns)


#df.head()

# đồ thị
fig, axes = plt.subplots(4,2)
fig.tight_layout(h_pad=.25)

j = 0
k = 0 
for i in range(1,8):
    if k==4:
        k=0
        j=1
    plt.subplot2grid((4, 2), (k,j))
    n = i-1
    counts = df[str_columns[n]].value_counts()
    plt.barh(y=counts.index, width=counts)
    plt.title('Distribution of '+str_columns[n])
    plt.ylabel(str_columns[n])
    plt.xlabel('Count')
    k=k+1
                     
plt.subplots_adjust(left=.25, bottom=.5, right=2.75, top=5)

#tạo biến giả từ các values trogn columns
df = pd.get_dummies(df)
print(len(df.columns))
print(df.columns)

#bản đồ nhiệt độ biểu hiện của values
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,30))
## Plotting heatmap. Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center = 0, );
plt.title("Heatmap of all the Features of Train data set", fontsize = 25);

#Xóa các columns dư thừa
df.pop("poutcome_unknown")
df.pop("marital_single")
df.pop("day_end")
df.pop("contact_unknown")

#model
def Models(models, X_train, X_test, y_train, y_test, title):
    model = models
    model.fit(X_train,y_train)
    
    X, y = Definedata()
    train_matrix = pd.crosstab(y_train, model.predict(X_train), rownames=['Actual'], colnames=['Predicted'])    
    test_matrix = pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
    matrix = pd.crosstab(y, model.predict(X), rownames=['Actual'], colnames=['Predicted'])
    
    f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(20, 3))
    #f = plt.figure(figsize=(20, 3))
    
    g1 = sns.heatmap(train_matrix, annot=True, fmt=".1f", cbar=False,ax=ax1)
    g1.set_title(title)
    g1.set_ylabel('Total Deposit = {}'.format(y_train.sum()), fontsize=14, rotation=90)
    g1.set_xlabel('Accuracy score for Training Dataset = {}'.format(accuracy_score(model.predict(X_train), y_train)))
    g2 = sns.heatmap(test_matrix, annot=True, fmt=".1f",cbar=False,ax=ax2)
    g2.set_title(title)
    g2.set_ylabel('Total Deposit = {}'.format(y_test.sum()), fontsize=14, rotation=90)
    g2.set_xlabel('Accuracy score for Testing Dataset = {}'.format(accuracy_score(model.predict(X_test), y_test)))
    g3 = sns.heatmap(matrix, annot=True, fmt=".1f",cbar=False,ax=ax3)
    g3.set_title(title)
    g3.set_ylabel('Total Deposit = {}'.format(y.sum()), fontsize=14, rotation=90)
    g3.set_xlabel('Accuracy score for Total Dataset = {}'.format(accuracy_score(model.predict(X), y)))
    
    plt.show()
    return y, model.predict(X)
#Đồ thị
import plotly.graph_objects as go
def Featureimportances(models, X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    importances = model.feature_importances_
    features = df.columns
    if len(importances)<len(features): 
        features = df.columns[:len(importances)]
    else:
        importances = model.feature_importances_[:len(features)]
    imp = pd.DataFrame({'Features': features, 'Importance': importances})
    imp = imp.sort_values(by = 'Importance', ascending=False)[:15]
    imp['Sum Importance'] = imp['Importance'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=imp.Features,y=imp.Importance, marker=dict(color=list(range(20)), colorscale="Sunsetdark")))

    fig.update_layout(title="Feature Importance",
                                 xaxis_title="Features", yaxis_title="Importance",title_x=0.5, paper_bgcolor="mintcream",
                                 title_font_size=20)
    fig.show()

###
def Definedata():
    # define dataset
    X=df.drop(columns=['y']).values
    y=df['y'].values
    return X, y

####
df.dropna(inplace=True)
X=df.drop(columns=['y']).values
y=df['y'].values

####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)


###
title = 'RandomForestClassifier'
%time Models(RandomForestClassifier(),X_train, X_test, y_train, y_test, title)


#####
from sklearn.metrics import confusion_matrix,auc,roc_curve

title = 'RandomForestClassifier'
y, ypred =  Models(RandomForestClassifier(),X_train, X_test, y_train, y_test, title)

fpr, tpr, thresholds = roc_curve(y, ypred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

###
Featureimportances(RandomForestClassifier(), X_train, y_train)















