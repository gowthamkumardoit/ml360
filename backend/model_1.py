#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare,chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, classification_report, roc_curve
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[37]:


stdscaler = StandardScaler()
labelencoder = LabelEncoder()


# In[2]:


data = pd.read_csv('loan.csv')
data = data.replace('?', np.nan)
data = data.replace('*', np.nan)
data = data.replace('NA.', np.nan)
data = data.replace('N.A.', np.nan)


# In[3]:


data.describe()


# In[4]:


print('Original shape of the data :',data.shape)
data.head()


# In[5]:


data.info()


# ## Removal of columns having all unique values

# In[6]:


##Removing columns having all unique values

def all_unique_values(data):

    col_names = data.columns

    col_to_drop = []

    for i in col_names:
        all_unique = len(data[i].value_counts().index)
        if all_unique == len(data):
            col_to_drop.append(i)

    ##Dropping those columns from Dataframe
    data = data.drop(col_to_drop, axis = 1)
    
#     print('Columns dropped :', col_to_drop)
#     print('Shape of the data after removing columns with all unique value :', data.shape)
    
    return data
    


# ## Removal of columns having more than 85% of missing values

# In[7]:


##Getting count and percentage of missing value in each column

def missing_value_column(data):
    count_of_null = data.isnull().sum()
    percent_of_missing = data.isnull().sum() * 100 / len(data)
    missing_value_data = pd.DataFrame({'percent_missing': percent_of_missing,'Count_of_Missing_Values ': count_of_null })

    ##Dropping columns having more than 85% null values

    columns_to_be_removed = missing_value_data[missing_value_data['percent_missing'] >= 80].index
    data = data.drop(columns_to_be_removed, axis = 1)

#     print('\n\nColumns dropped :', columns_to_be_removed)
#     print('Shape of the data after removing columns with missing values more than 85% : ', data.shape)
    
    return data


# ## Removal of rows having more  than 60% missing value

# In[8]:


def missing_value_row(data):
    row_wise_null = (data.isnull().sum(axis=1) / data.shape[1]) * 100
    data['row_wise_null'] = row_wise_null

    ##Dropping rows having more than 65% missing values

    i = data[data['row_wise_null'] > 60].index
    num_of_rows_removed = len(i)
    data = data.drop(i)

    data = data.drop('row_wise_null', axis = 1)

#     print('\n\nNumber of rows dropped :', num_of_rows_removed)
#     print('Shape of the data after removing rows with missing values more than 65% : ', data.shape)
    
    return data
    


# ## Removal of columns having only one unique class or values

# In[9]:


##Removing columns which has only 1 unique values which will be of no use

def one_unique(data):
    col_names= data.columns

    col_drop = []

    for i in col_names:
        check_unique = len(data[i].value_counts().index)
        if check_unique ==1:
            col_drop.append(i)

    #print('\n\nColumns dropped :',col_drop)

    ##Dropping those columns from dataframe
    data = data.drop(col_drop, axis = 1)

    #print('Shape of the data after removing columns with single unique value :', data.shape)
    
    return data
    


# ## Basics EDA of Above all four fucntions

# In[10]:


def basic_eda(data):
    data = all_unique_values(data)
    data = missing_value_column(data)
    data = missing_value_row(data)
    data = one_unique(data)
    return data


# ## Separation of categorical and numerical columns

# In[11]:


##Separating numerical and categorical columns 

def num_cat_separation(data):
    col_names = data.columns
    
    col_names_updated_cat = []
    col_names_updated_num = []

    for i in col_names:
        counts_of_individual_cols = data[i].value_counts()
        check = len(data[i])/len(counts_of_individual_cols.index)
        if check >30:

            col_names_updated_cat.append(i)
            cat_col_names = data[col_names_updated_cat].columns

            for i in cat_col_names:
                data[i] = data[i].astype('object')

        else:
            col_names_updated_num.append(i)
            num_col_names = data[col_names_updated_num].columns

            for i in num_col_names:
                data[i] = data[i].astype('float64')
            
    return list(col_names_updated_num), list(col_names_updated_cat)

#     print('Numerical columns :\n', col_names_updated_num)
#     print('Numerical columns :\n', col_names_updated_cat)
    


# ## Imputing missing values

# In[12]:


# def imputation(data, target):
    
#     data = basic_eda(data)
    
#     count_of_null = data.isnull().sum()
#     percent_of_missing = data.isnull().sum() * 100 / len(data)
#     missing_value_data = pd.DataFrame({'percent_missing': percent_of_missing,'Count_of_Missing_Values ': count_of_null })
    
#     col_names = data.columns

#     col_names_updated_cat = []
#     col_names_updated_num = []

#     for i in col_names:
#         counts_of_individual_cols = data[i].value_counts()
#         check = len(data[i])/len(counts_of_individual_cols.index)
#         if check >30:

#             col_names_updated_cat.append(i)
#             cat_col_names = data[col_names_updated_cat].columns

#             for i in cat_col_names:
#                 data[i] = data[i].astype('object')

#         else:
#             col_names_updated_num.append(i)
#             num_col_names = data[col_names_updated_num].columns

#             for i in num_col_names:
#                 data[i] = data[i].astype('float64')
    
#     global data_null_treated 
#     data_null_treated = data.copy()
#     label_encoder =  LabelEncoder()
    
    

#     cols_to_be_imputed =  missing_value_data[missing_value_data['percent_missing'] > 0].sort_values('percent_missing', ascending=False).index
#     cols_to_be_imputed = list(cols_to_be_imputed)
#     if 'target' in cols_to_be_imputed:
#         cols_to_be_imputed.remove('target')


#     for i in cols_to_be_imputed:

#         data_dup = data_null_treated.copy()

#         ##Replacing column having below 2 percent missing values with median and mode

#         below_2_percent_columns = missing_value_data[missing_value_data['percent_missing'] < 2].index
#         below_2_percent_columns = list(below_2_percent_columns)
#         if i in below_2_percent_columns:
#             below_2_percent_columns.remove(i)


#         for j in below_2_percent_columns:

#             if j in col_names_updated_num:
#                 data_dup[j] = data_dup[[j]].apply(lambda x:x.fillna(x.median()), axis = 0)
#             else:
#                 data_dup[j] = data_dup[[j]].apply(lambda x:x.fillna(data_dup[j].value_counts().index.max()))


#         ##Seperating rows without null for train 
#         data_dup_train = data_dup[data_dup[i].isna()==False]

#         data_dup_train_copy = data_dup_train.copy()

#         ##Dropping null values in other columns
#         data_dup_train = data_dup_train.dropna() 

#         ##Seperating rows with null for test
#         data_dup_test = data_dup[data_dup[i].isna()] 

#         ##Removing column having above 15 percent missing values

#         above_15_percent_columns = missing_value_data[missing_value_data['percent_missing'] > 15].index
#         above_15_percent_columns = list(above_15_percent_columns)
#         data_dup_train = data_dup_train.drop(above_15_percent_columns, axis = 1)
#         data_dup_test = data_dup_test.drop(above_15_percent_columns, axis = 1)

#         ##Train test split

#         x_test = data_dup_test.drop(i, axis = 1)    
#         x_test = pd.get_dummies(x_test, drop_first=True)
#         x_test_columns = x_test.columns
#         for k in x_test_columns:
#             if x_test[k].dtype == 'float64':
#                 x_test[k] = x_test[[k]].apply(lambda x:x.fillna(x.median()), axis = 0)
#             else:
#                 x_test[k] = x_test[[k]].apply(lambda x:x.fillna(x_test[k].value_counts().index.max()))

#         x_train = data_dup_train.drop(i, axis = 1)
#         x_train = pd.get_dummies(x_train, drop_first=True)
#         x_train = x_train[x_test.columns]

#         y_train = data_dup_train[[i]]
#         if y_train[i].dtype == 'O':
#             y_train[i] = label_encoder.fit_transform(y_train[i])
#             y_train[[i]] = y_train[[i]].astype('int')


#         ##Building model
#         if i in col_names_updated_num:
#             model_rf = RandomForestRegressor(n_estimators=100, max_depth=6)
#         else:
#             model_rf = RandomForestClassifier(n_estimators=100, max_depth=6)

#         model_rf.fit(x_train , y_train)
#         rf_score = model_rf.score(x_train , y_train)
#         print('RandomForest Score :' , rf_score)

#         if i in col_names_updated_num:
#             model_lr = LinearRegression()
#         else:
#             model_lr = LogisticRegression()

#         model_lr.fit(x_train , y_train)
#         lr_score = model_lr.score(x_train , y_train)
#         print('\nLogisticRegression Score :' , lr_score)

#         ##Checking which model is better
#         if rf_score > lr_score:
#             print('\nFor', i, ' RandomForest performs better. So we will go with this.\n')
#             model = model_rf
#         else:
#             print('\n\nFor', i ,' Logistic Regression performs better. So we will go with this.')
#             model = model_lr   

#         prediction = model.predict(x_test)  
#         print(prediction.dtype,'\n\n')
#         if prediction.dtype == 'int32':
#             prediction = label_encoder.inverse_transform(prediction)

#         prediction_df = pd.DataFrame(prediction)
#         #print('\n\n Predicted count of ', i , '  :' , prediction_df[0].value_counts())

#         data_dup_test = data_dup_test.drop(i , axis = 1)

#         data_dup_test[i] = prediction

#         data_dup_complete = pd.concat([data_dup_train_copy , data_dup_test])

#         data_dup_complete = data_dup_complete.sort_index()

#         predicted = data_dup_complete[[i]]

#         data_null_treated = data_null_treated.drop(i , axis = 1)

#         data_null_treated[i] = predicted  
          


# In[13]:


def imputation(data):
    
    data = basic_eda(data)
    
    count_of_null = data.isnull().sum()
    percent_of_missing = data.isnull().sum() * 100 / len(data)
    missing_value_data = pd.DataFrame({'percent_missing': percent_of_missing,'Count_of_Missing_Values ': count_of_null })
    
    global numerical_column_names
    global categorical_column_names
    
    numerical_column_names, categorical_column_names = num_cat_separation(data)
    
    global data_null_treated 
    data_null_treated = data.copy()
    label_encoder =  LabelEncoder()
     

    cols_to_be_imputed =  missing_value_data[missing_value_data['percent_missing'] > 0].sort_values('percent_missing', ascending=False).index
    cols_to_be_imputed = list(cols_to_be_imputed)
#     print(cols_to_be_imputed)
#     if target in cols_to_be_imputed:
#         cols_to_be_imputed.remove(target)
        
# #     cols_to_be_imputed = [ x for x in cols_to_be_imputed if x is not target ]
#     print(cols_to_be_imputed, '\n\n')

    Imputed_column_array = []
    for i in cols_to_be_imputed:

        data_dup = data_null_treated.copy()

        ##Replacing column having below 2 percent missing values with median and mode

        below_2_percent_columns = missing_value_data[missing_value_data['percent_missing'] < 2].index
        below_2_percent_columns = list(below_2_percent_columns)
        if i in below_2_percent_columns:
            below_2_percent_columns.remove(i)


        for j in below_2_percent_columns:

            if j in numerical_column_names:
                data_dup[j] = data_dup[[j]].apply(lambda x:x.fillna(x.median()), axis = 0)
            else:
                data_dup[j] = data_dup[[j]].apply(lambda x:x.fillna(data_dup[j].value_counts().index.max()))


        ##Seperating rows without null for train 
        data_dup_train = data_dup[data_dup[i].isna()==False]

        data_dup_train_copy = data_dup_train.copy()

        ##Dropping null values in other columns
        data_dup_train = data_dup_train.dropna() 

        ##Seperating rows with null for test
        data_dup_test = data_dup[data_dup[i].isna()] 

        ##Removing column having above 15 percent missing values except the column to be imputed in particular iteration

        above_15_percent_columns = missing_value_data[missing_value_data['percent_missing'] > 15].index
        above_15_percent_columns = list(above_15_percent_columns)
        if i in above_15_percent_columns:
            above_15_percent_columns.remove(i)
        data_dup_train = data_dup_train.drop(above_15_percent_columns, axis = 1)
        data_dup_test = data_dup_test.drop(above_15_percent_columns, axis = 1)

        ##Train test split

        x_test = data_dup_test.drop(i, axis = 1)    
        x_test = pd.get_dummies(x_test, drop_first=True)
        x_test_columns = x_test.columns
        for k in x_test_columns:
            if x_test[k].dtype == 'float64':
                x_test[k] = x_test[[k]].apply(lambda x:x.fillna(x.median()), axis = 0)
            else:
                x_test[k] = x_test[[k]].apply(lambda x:x.fillna(x_test[k].value_counts().index.max()))

        x_train = data_dup_train.drop(i, axis = 1)
        x_train = pd.get_dummies(x_train, drop_first=True)
        x_train = x_train[x_test.columns]

        y_train = data_dup_train[[i]]
        if y_train[i].dtype == 'O':
            y_train[i] = label_encoder.fit_transform(y_train[i])
            y_train[[i]] = y_train[[i]].astype('int')


        ##Building model
        if i in numerical_column_names:
            model_rf = RandomForestRegressor(n_estimators=100, max_depth=6)
        else:
            model_rf = RandomForestClassifier(n_estimators=100, max_depth=6)

        model_rf.fit(x_train , y_train)
        rf_score = model_rf.score(x_train , y_train)
        print('RandomForest Score :' , rf_score)

        if i in numerical_column_names:
            model_lr = LinearRegression()
        else:
            model_lr = LogisticRegression()

        model_lr.fit(x_train , y_train)
        lr_score = model_lr.score(x_train , y_train)
        print('\nLogisticRegression Score :' , lr_score)

        ##Checking which model is better
        if rf_score > lr_score:
            print('\nFor', i, ' RandomForest performs better. So we will go with this.\n')
            model = model_rf
            Imputed_column_array.append({i:'Random Forest'})
        else:
            print('\n\nFor', i ,' Logistic Regression performs better. So we will go with this.')
            model = model_lr   
            Imputed_column_array.append({i:'Logistic Regression'})

        prediction = model.predict(x_test)  
        print(prediction.dtype,'\n\n')
        if prediction.dtype == 'int32':
            prediction = label_encoder.inverse_transform(prediction)

        prediction_df = pd.DataFrame(prediction)
        #print('\n\n Predicted count of ', i , '  :' , prediction_df[0].value_counts())

        data_dup_test = data_dup_test.drop(i , axis = 1)

        data_dup_test[i] = prediction

        data_dup_complete = pd.concat([data_dup_train_copy , data_dup_test])

        data_dup_complete = data_dup_complete.sort_index()

        predicted = data_dup_complete[[i]]

        data_null_treated = data_null_treated.drop(i , axis = 1)

        data_null_treated[i] = predicted  
        
    return Imputed_column_array
          


# In[14]:


imputation(data)


# In[15]:


data_null_treated.info()


# ## Handling class imbalance in target column

# In[16]:


# #Getting each class in target columns as a list

# unique_class = data.Loan_Status.value_counts().index
# unique_class = list(unique_class)

# #Finding the equal percentage for number of classes there in target column
# if len(unique_class) == 2:
#     equal_percentage = ((len(data.Loan_Status) / len(data.Loan_Status.value_counts().index)) / (len(data.Loan_Status))) * 100
#     equal_percentage = equal_percentage / 2
# else:
#     equal_percentage = ((len(data.Loan_Status) / len(data.Loan_Status.value_counts().index)) / (len(data.Loan_Status))) * 100

# #Finding classes with imbalance issue
# columns_with_no_imbalance_issue = []
# columns_with_imbalance_issue = []
# for i in unique_class:
#     if ((data.Loan_Status.value_counts()[i]) / len(data.Loan_Status)) * 100 >= equal_percentage:
#         print('Class with no imbalance issue :', i )
#         columns_with_no_imbalance_issue.append(i)
#     else:
#         print('Class with imbalance issue :', i)
#         columns_with_imbalance_issue.append(i)
        
        
# ##Finding the columns which need to be upsampled        
# each_class_percentage = []
# each_class = []
# for i in unique_class:
#     b = (data.Loan_Status.value_counts()[i] / len(data)) * 100
#     each_class.append(i)
#     each_class_percentage.append(b)
    
# final = pd.DataFrame({'Class':each_class, 'Percentage_present':each_class_percentage})

# max_class = final.sort_values('Percentage_present',ascending=False)['Class'][0]
# max_class_percentage = final.sort_values('Percentage_present',ascending=False)['Percentage_present'][0]

# print('\nTop most Majority class is :', max_class, '\t\tAnd the percentage of that class is :', max_class_percentage)

# classes = list(final['Class'])
# classes_to_be_upsampled = []
# for i in classes:
#     diff_percent = max_class_percentage - final[final['Class'] == i]['Percentage_present']
#     diff_percent = np.array(diff_percent)
#     if diff_percent > equal_percentage:
#         classes_to_be_upsampled.append(i)
        
# print('\nClasses to be upsampled :', classes_to_be_upsampled)


# ## Feature selection

# In[16]:


data_null_treated.info()


# In[17]:


def feature_selection(data_null_treated, target):
    
    final_features_choosed = []
    ## Running Chisqaure for Categorical columns
    chi_columns = data_null_treated[categorical_column_names].drop(target, axis = 1).columns
    p_value_for_chisq = []
    name = []
    for i in chi_columns:
        cont = pd.crosstab(data_null_treated[i],
                           data_null_treated[target])
        name.append(i)
        p_value_for_chisq.append(chi2_contingency(cont)[1])
        chisqaure_df = pd.DataFrame({'Variables':name,'P_value':p_value_for_chisq})

    ## Getting columns which are dependent to our target column at 90% confidence interval    
    chi_square_imp_feature = chisqaure_df[chisqaure_df['P_value'] < 0.10]['Variables']

    ## Getting dataframe with Categorical(which are dependent) and numerical columns
    data_complete = pd.concat([data_null_treated[chi_square_imp_feature], data_null_treated[numerical_column_names]], axis = 1)
    data_complete[target] = data_null_treated[[target]]
    print(data_complete.info())


    ## Running Random Forest for those important columns from chisquare
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=5000, min_samples_leaf=2)
    x = data_complete.drop(target, axis = 1)
    #y = data_complete[[target]]
    y = np.asarray(data_complete[target], dtype = '|S6')
    x = pd.get_dummies(x,  drop_first=True)
    rf.fit(x,  y)
    rf.score(x ,y)

    ## Plotting those feature with their importance
    variables = x.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10,8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [variables[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


    ## Creating a dataframe with variables and their importance as well as the running total of the importance
    variables = x.columns
    feature_imp = pd.DataFrame({'Variables':variables, 'Importance':rf.feature_importances_})
    feature_imp = feature_imp.sort_values('Importance', ascending = False).reset_index().drop('index', axis=1)
    variable_imp_values = pd.Series(feature_imp['Importance'])
    running_total = variable_imp_values.cumsum()
    feature_imp['Running_Total'] = running_total
    feature_imp

    ##Proceeding with variables which contribute upto 90% 
    final_variables = list(feature_imp[feature_imp['Running_Total'] < 0.94]['Variables'])

    x_dummies = pd.get_dummies(data_complete.drop(target, axis = 1), drop_first=True)
    global final_data_for_modelling
    final_data_for_modelling = x_dummies[final_variables]
    final_data_for_modelling[target] = data_complete[[target]]
    final_features_choosed = final_data_for_modelling.columns
    print(feature_imp,'\n\n')
    return final_features_choosed


# In[18]:


feature_selection(data_null_treated, 'Loan_Status')


# In[20]:


# # Train Test split

# x = final_data.drop('Loan_Status', axis =1)
# y = final_data[['Loan_Status']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# # cross-validation using random forest on train data

# rfc_cv = RandomForestClassifier(n_estimators=100, max_depth = 5, min_samples_split=4)
# scores = cross_val_score(rfc_cv, x_train, y_train, cv=10, scoring = "accuracy")
# # print("Scores:", scores)
# print("Cross-validation Accuracies:", np.round(scores*100, 2))
# print("Mean:", scores.mean())
# print("Standard Deviation:", scores.std())


# ## Final Modelling 

# In[19]:


final_data_for_modelling.info()


# #### For categorical

# In[73]:


def LogisticRegression_modelling(data, target):
    X = data.drop(target, axis = 1)
    Y = data[[target]]
    model = LogisticRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 555)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_prob = model.predict_proba(X_test)
    Accuracy = metrics.accuracy_score(Y_test, prediction)
    print('Accuracy :', Accuracy )
    F1_score = metrics.f1_score(Y_test, prediction, pos_label=2, average=None)
    print('F1_square :', F1_score )
    AUC_score = metrics.roc_auc_score(Y_test, prediction_prob[:,1])
    print('Test_auc :', AUC_score)


# In[74]:


def RandomForest_modelling(data, target):

    X = data.drop(target,axis=1)
    Y = data[[target]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 7)

    param_grid = {'n_estimators':(100,150),'min_samples_split':np.arange(2,6),'max_depth':(5,6)}
    gs = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=10)
    gs.fit(X_train,Y_train)
#     print("RandomForest : %0.4f [%s]"%(gs.best_score_,gs.best_params_))
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = RandomForestClassifier(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
#     model_score = model.score(X_train, Y_train)
    prediction_prob = model.predict_proba(X_test)
    Accuracy = metrics.accuracy_score(Y_test, prediction)
    print('Accuracy :', Accuracy )
    F1_score = metrics.f1_score(Y_test, prediction, pos_label=2, average=None)
    print('F1_square :', F1_score )
    AUC_score = metrics.roc_auc_score(Y_test, prediction_prob[:,1])
    print('Test_auc :', AUC_score)


# In[75]:


def GB_modelling(data, target):

    X = data.drop(target,axis=1)
    Y = data[[target]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 7)

    param_grid = {'n_estimators':(100,150),'min_samples_split':np.arange(2,6),'max_depth':(5,6)}
    gs = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=10)
    gs.fit(X_train,Y_train)
#     print("RandomForest : %0.4f [%s]"%(gs.best_score_,gs.best_params_))
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = GradientBoostingClassifier(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
#     model_score = model.score(X_train, Y_train)
    prediction_prob = model.predict_proba(X_test)
    Accuracy = metrics.accuracy_score(Y_test, prediction)
    print('Accuracy :', Accuracy )
    F1_score = metrics.f1_score(Y_test, prediction, pos_label=2, average=None)
    print('F1_square :', F1_score )
    AUC_score = metrics.roc_auc_score(Y_test, prediction_prob[:,1])
    print('Test_auc :', AUC_score)


# In[76]:


LogisticRegression_modelling(final_data_for_modelling, 'Loan_Status')


# In[77]:


RandomForest_modelling(final_data_for_modelling, 'Loan_Status')


# In[78]:


GB_modelling(final_data_for_modelling,'Loan_Status')


# #### For numerical

# In[20]:


def AdjustedRsqaure(N, p, R_square):
    # N = length of dataset, p = Number of predictors 
    result = 1 - (1 - R_square) * (N - 1) / (N - p - 1)
    return result


# In[26]:


def LinearRegression_modelling(data, target):
    X = data.drop(target, axis = 1)
    X = pd.get_dummies(X)
    
    ##Variables for Adjusted R_sqaure
    N = len(X)
    p = X.shape[1]
    
    Y = data[[target]]
    model = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 555)
    model.fit(X_train, Y_train)
#     print('Train score', model.score(X_train, Y_train))
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('LinearRegression :', RMSE)
    return RMSE, R_square , Adjusted_r_sqaure


# In[27]:


def RandomForest_modelling(data, target):

    X = data.drop(target,axis=1)
    X = pd.get_dummies(X)
    Y = data[[target]]
    
    ##Variables for Adjusted R_sqaure
    N = len(X)
    p = X.shape[1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 7)

    param_grid = {'n_estimators':(100,150),'min_samples_split':np.arange(2,6),'max_depth':(5,6)}
    gs = GridSearchCV(RandomForestRegressor(),param_grid=param_grid,cv=10)
    gs.fit(X_train,Y_train)
#     print("RandomForest : %0.4f [%s]"%(gs.best_score_,gs.best_params_))
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = RandomForestRegressor(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    model_score = model.score(X_train, Y_train)
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('Random forest : ', RMSE)
    return RMSE, R_square, Adjusted_r_sqaure


# In[28]:


def KNN_modelling(data, target):
    X = data.drop(target, axis = 1)
    X = pd.get_dummies(X)
    Y = data[[target]]
    
    ##Variables for Adjusted R_sqaure
    N = len(X)
    p = X.shape[1]
    
    model = KNeighborsRegressor()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 555)
    model.fit(X_train, Y_train)
#     print('Train score', model.score(X_train, Y_train))
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('KNN :', RMSE)
    return RMSE, R_square, Adjusted_r_sqaure


# In[35]:


def KNN_modelling_scaled(data, target):
    X = data.drop(target, axis = 1)
    X = pd.get_dummies(X)
    X = stdscaler.fit_transform(X)
    Y = data[[target]]
    
    ##Variables for Adjusted R_sqaure
    N = len(X)
    p = X.shape[1]
    
    model = KNeighborsRegressor()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 555)
    model.fit(X_train, Y_train)
#     print('Train score', model.score(X_train, Y_train))
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('KNN :', RMSE)
    return RMSE, R_square, Adjusted_r_sqaure


# In[30]:


LinearRegression_modelling(final_data_for_modelling, 'LoanAmount')


# In[31]:


RandomForest_modelling(final_data_for_modelling, 'LoanAmount')


# In[32]:


KNN_modelling(final_data_for_modelling, 'LoanAmount')


# In[36]:


KNN_modelling_scaled(final_data_for_modelling, 'LoanAmount')


# In[ ]:




