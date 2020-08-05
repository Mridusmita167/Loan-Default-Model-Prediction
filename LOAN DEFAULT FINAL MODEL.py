#!/usr/bin/env python
# coding: utf-8

# # Data Processing

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns



# In[32]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


# In[4]:


df = pd.read_csv("D:\\New folder\\PROJECT\\bank_final.csv")


# In[5]:


df.head()


# In[6]:


df.shape  #(rows,columns)


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.nunique()


# In[10]:


df = df.drop_duplicates()


# In[11]:


df.shape


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[15]:


null_rate = df.isnull().sum(axis = 0).sort_values(ascending = False)/float((len(df)))
null_rate[null_rate > 0.6]


# In[16]:


list=[
        'DisbursementGross', 'BalanceGross',
        'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']

for i in list:
  df[i] = df[i].str.replace(',', '').str.replace('$', '').astype(float)


# In[18]:


df.MIS_Status.value_counts()


# In[19]:


#dropping the NAs in MIS_Status and changing it into a integer data type
#replacing pif:1 & chgoff:0
df.dropna(subset=['MIS_Status'], how='all', inplace=True)
df['MIS_Status'] = df['MIS_Status'].str.lower().replace({'p i f': 1, 'chgoff': 0})
df.MIS_Status.value_counts()


# In[20]:


df.LowDoc.value_counts()


# In[21]:


#to check the number of rows without Y or N and assign them nan 
cond = df[(df['LowDoc'] != "Y") & (df['LowDoc'] != "N")]
for i in cond.index:
    df.loc[i,'LowDoc'] = np.nan


# In[22]:


#dropping all the nan values and changing it into a integer data type
#replacing y:1 & n:0
df.dropna(subset=['LowDoc'], how='all', inplace=True)
df['LowDoc'] = df['LowDoc'].str.lower().replace({'y': 1, 'n': 0})
df.LowDoc.value_counts()


# In[23]:


#Histogram
df.MIS_Status.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Status of MIS")
plt.ylabel('Status')
plt.xlabel('Count')
plt.show()


# In[24]:


#Histogram
df.LowDoc.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Low Documentation")
plt.ylabel('Status')
plt.xlabel('Count')
plt.show()


# In[25]:


plt.figure(figsize=(12,9))               #correlation matrix
ax = sns.heatmap(df.corr(),annot = True)
bottom, top = ax.get_ylim()              #fixing seaborn plotting issues
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[27]:


df.head()


# Columns like Name and BankName are unique values and Zip, State are not useful as well for default.
# Columns "Disbursment Date", "DisbursementGross", "BalanceGross", "SBA_Appv" and "ChgOffPrinGr" are important after default.
# Column "ChgOffDate" contains a lot of NAN so are not useful.
# Columns ApprovalDate, ApprovalFY with dates are not needed as well.
# Column "RevLineCr" contains a lot  of NAN and other unnecessary values so is not useful for default

# In[29]:
from sklearn.utils import resample

df_majority = df[df.MIS_Status== 1]
df_minority = df[df.MIS_Status== 0]

majority_down = resample(df_majority, replace=False, n_samples=70000, random_state=123)
down_sample_df = pd.concat([majority_down,df_minority])

print(down_sample_df.MIS_Status.value_counts())


feature_cols = ['Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'UrbanRural', 'GrAppv', 'LowDoc']


# In[30]:


X = down_sample_df[feature_cols] # Predictor variable
y = down_sample_df.MIS_Status # Target variable


# In[33]:


#Train-Test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y.values, test_size=0.3, random_state=0)


# In[34]:


#Standardize X_train with fit_transform, X_test with tranform only.
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train_raw)
X_test = stdsc.transform(X_test_raw)


# In[35]:


#Prepare for cross validation
seed = 10
kfold = model_selection.KFold(n_splits=10, random_state=seed)


# # Begin model train/test/evaluate

# We will define model sets to compare multiple models

# In[36]:


different_model_comparison = {
    "Random Forest":RandomForestClassifier(random_state=0,n_estimators=10),
    "Logistic Regression":LogisticRegression(random_state=0),
    "Decision Tree":DecisionTreeClassifier(random_state=0),
    "KNN":KNeighborsClassifier(n_neighbors=5),
    "GNB": GaussianNB()
   }

# Tried to use 1~30
different_tree_number_comaprison = {
    "Random Forest(1)":RandomForestClassifier(random_state=0,n_estimators=1),
    "Random Forest(5)":RandomForestClassifier(random_state=0,n_estimators=5),
    "Random Forest(10)":RandomForestClassifier(random_state=0,n_estimators=10),
    "Random Forest(15)":RandomForestClassifier(random_state=0,n_estimators=15),
    "Random Forest(20)":RandomForestClassifier(random_state=0,n_estimators=20),
    "Random Forest(25)":RandomForestClassifier(random_state=0,n_estimators=25),
    "Random Forest(30)":RandomForestClassifier(random_state=0,n_estimators=30)
}
#Tried to use n = 3,5,10
different_neighbour_comparison = {
    "KNN(3)":KNeighborsClassifier(n_neighbors=3),
    "KNN(5)":KNeighborsClassifier(n_neighbors=5),
    "KNN(10)":KNeighborsClassifier(n_neighbors=10)
}


# Function to train model, return a dictionary of trained model

# In[37]:


# Wrapping this function so we can easily change from original data to balanced data
# function to train model, return a dictionary of trained model
def train_model(model_dict,X_train,y_train):
    for model in model_dict:
        print("Training:",model)
        model_dict[model].fit(X_train,y_train)
    return model_dict


# Function to evaluate model performance

# In[42]:


# Wrapping this function so we can easily change the model and evaluate them
# function to evaluate model performance 
from sklearn import metrics
def model_eval(clf_name,clf,X_test,y_test):
    print("Evaluating:",clf_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:,1]
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
    report = pd.Series({
        "model":clf_name,
        "precision":metrics.precision_score(y_test, y_pred),
        "recall":metrics.recall_score(y_test, y_pred),
        "f1":metrics.f1_score(y_test, y_pred),
        'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
        "Accuracy":metrics.accuracy_score(y_test, y_pred)
    })
    cross_validation_result = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    print(cross_validation_result)
    # draw ROC 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)
    plt.figure(1, figsize=(6,6))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot(fpr, tpr,label=clf_name)
    plt.plot([0,1],[0,1], color = 'black')
    plt.legend()
    return report,confusion_matrix


# The function that calls train_model and model_eval, this makes model training and evaluation much easier.

# In[43]:


def train_eval_model(model_dict,X_train,y_train,X_test,y_test):
    cols = ['model', 'roc_auc_score', 'precision', 'recall','f1','Accuracy']
    model_report = pd.DataFrame(columns = cols)
    cm_dict = {}
    model_dict = train_model(model_dict,X_train,y_train)
    for model in model_dict:
        report,confusion_matrix = model_eval(model,model_dict[model],X_test,y_test)
        model_report = model_report.append(report,ignore_index=True)
        cm_dict[model] = confusion_matrix
    return model_report,cm_dict


# Visualization function

# In[44]:


def plot_which_bar(df,col_name):
    df.set_index("model").loc[:,col_name].plot(kind='bar', stacked=True, sort_columns=True, figsize = (16,10))
    plt.title(col_name)
    plt.show()


# # Train and Evaluation of the models

# Run models using unbalanced data

# In[75]:


model_report,cm_dict = train_eval_model(different_model_comparison,X_train,y_train,X_test,y_test)


# In[76]:


cm_dict


# In[77]:


model_report_n_trees,cm_dict_n_trees = train_eval_model(different_tree_number_comaprison,X_train,y_train,X_test,y_test)


# In[78]:


cm_dict_n_trees


# In[79]:


model_n_neighbour,cm_dict_n_neighbour = train_eval_model(different_neighbour_comparison,X_train,y_train,X_test,y_test)


# In[80]:


cm_dict_n_neighbour


# Run models using balanced Data

# In[81]:


index_split = int(len(X)*0.7) #30% testing

X_train_bal, y_train_bal = SMOTE(random_state=0).fit_sample(X_train,y_train)
X_test_bal, y_test_bal = SMOTE(random_state=0).fit_sample(X_test, y_test)


# In[82]:


len(X_train_bal)


# In[83]:


len(X_train)


# In[84]:


sum(y_train_bal)/len(y_train_bal)


# In[85]:


sum(y_train)/len(y_train)


# In[86]:


model_report_bal,cm_dict_bal = train_eval_model(different_model_comparison,X_train_bal,y_train_bal,X_test_bal,y_test_bal)


# In[87]:


cm_dict_bal


# In[88]:


model_report_n_trees_bal,cm_dict_n_trees_bal = train_eval_model(different_tree_number_comaprison,X_train_bal,y_train_bal,X_test_bal,y_test_bal)


# In[89]:


cm_dict_n_trees_bal


# In[90]:


model_report_n_nb_bal,cm_dict_n_nb_bal = train_eval_model(different_neighbour_comparison,X_train_bal,y_train_bal,X_test_bal,y_test_bal)


# In[62]:


cm_dict_n_nb_bal


# # Compare the performances of various models of balanced and unbalanced data

# Unbalanced data

# In[63]:


model_report


# Balanced data

# In[71]:


model_report_bal


# Different parameters of Random Forest of Unbalanced data

# In[65]:


model_report_n_trees


# In[66]:


plot_which_bar(model_report_n_trees,"recall")


# Different parameters of Random Forest of Balanced data

# In[67]:


model_report_n_trees_bal


# In[73]:


plot_which_bar(model_report_n_trees_bal,"recall")


# Different parameters of Random Forest of Unbalanced data

# In[74]:


model_report_n_nb


# In[ ]:


plot_which_bar(model_report_n_nb,"recall")


# Different parameters of Random Forest of Balanced data

# In[ ]:


model_report_n_nb_bal


# In[ ]:


plot_which_bar(model_report_n_nb_bal,"recall")



import pickle

rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=30,criterion="entropy")

rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

