import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare, chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder


def all_unique_values(data):

    col_names = data.columns

    col_to_drop = []

    for i in col_names:
        all_unique = len(data[i].value_counts().index)
        if all_unique == len(data):
            col_to_drop.append(i)

    # Dropping those columns from Dataframe
    data = data.drop(col_to_drop, axis=1)

#     print('Columns dropped :', col_to_drop)
#     print('Shape of the data after removing columns with all unique value :', data.shape)

    return data

# Getting count and percentage of missing value in each column


def missing_value_column(data):
    count_of_null = data.isnull().sum()
    percent_of_missing = data.isnull().sum() * 100 / len(data)
    missing_value_data = pd.DataFrame(
        {'percent_missing': percent_of_missing, 'Count_of_Missing_Values ': count_of_null})

    # Dropping columns having more than 85% null values

    columns_to_be_removed = missing_value_data[missing_value_data['percent_missing'] >= 80].index
    data = data.drop(columns_to_be_removed, axis=1)

#     print('\n\nColumns dropped :', columns_to_be_removed)
#     print('Shape of the data after removing columns with missing values more than 85% : ', data.shape)

    return data

# ## Removal of rows having more  than 60% missing value


def missing_value_row(data):
    row_wise_null = (data.isnull().sum(axis=1) / data.shape[1]) * 100
    data['row_wise_null'] = row_wise_null

    # Dropping rows having more than 65% missing values

    i = data[data['row_wise_null'] > 60].index
    num_of_rows_removed = len(i)
    data = data.drop(i)

    data = data.drop('row_wise_null', axis=1)

#     print('\n\nNumber of rows dropped :', num_of_rows_removed)
#     print('Shape of the data after removing rows with missing values more than 65% : ', data.shape)

    return data


# Removing columns which has only 1 unique values which will be of no use

def one_unique(data):
    col_names = data.columns

    col_drop = []

    for i in col_names:
        check_unique = len(data[i].value_counts().index)
        if check_unique == 1:
            col_drop.append(i)

    #print('\n\nColumns dropped :',col_drop)

    # Dropping those columns from dataframe
    data = data.drop(col_drop, axis=1)

    #print('Shape of the data after removing columns with single unique value :', data.shape)

    return data

# ## Basics EDA of Above all four fucntions


def basic_eda(data):
    data = all_unique_values(data)
    data = missing_value_column(data)
    data = missing_value_row(data)
    data = one_unique(data)
    return data


# Separating numerical and categorical columns

def num_cat_separation(data):
    col_names = data.columns

    col_names_updated_cat = []
    col_names_updated_num = []

    for i in col_names:
        counts_of_individual_cols = data[i].value_counts()
        check = len(data[i])/len(counts_of_individual_cols.index)
        if check > 30:

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


def imputation(data):

    data = basic_eda(data)

    count_of_null = data.isnull().sum()
    percent_of_missing = data.isnull().sum() * 100 / len(data)
    missing_value_data = pd.DataFrame(
        {'percent_missing': percent_of_missing, 'Count_of_Missing_Values ': count_of_null})

    global numerical_column_names
    global categorical_column_names

    numerical_column_names, categorical_column_names = num_cat_separation(data)

    global data_null_treated
    data_null_treated = data.copy()
    label_encoder = LabelEncoder()

    cols_to_be_imputed = missing_value_data[missing_value_data['percent_missing'] > 0].sort_values(
        'percent_missing', ascending=False).index
    cols_to_be_imputed = list(cols_to_be_imputed)
    # if target in cols_to_be_imputed:
    #     cols_to_be_imputed.remove(target)

    Imputed_column_array = []
    for i in cols_to_be_imputed:

        data_dup = data_null_treated.copy()

        # Replacing column having below 2 percent missing values with median and mode

        below_2_percent_columns = missing_value_data[missing_value_data['percent_missing'] < 2].index
        below_2_percent_columns = list(below_2_percent_columns)
        if i in below_2_percent_columns:
            below_2_percent_columns.remove(i)

        for j in below_2_percent_columns:

            if j in numerical_column_names:
                data_dup[j] = data_dup[[j]].apply(
                    lambda x: x.fillna(x.median()), axis=0)
            else:
                data_dup[j] = data_dup[[j]].apply(
                    lambda x: x.fillna(data_dup[j].value_counts().index.max()))

        # Seperating rows without null for train
        data_dup_train = data_dup[data_dup[i].isna() == False]

        data_dup_train_copy = data_dup_train.copy()

        # Dropping null values in other columns
        data_dup_train = data_dup_train.dropna()

        # Seperating rows with null for test
        data_dup_test = data_dup[data_dup[i].isna()]

        # Removing column having above 15 percent missing values

        above_15_percent_columns = missing_value_data[missing_value_data['percent_missing'] > 15].index
        above_15_percent_columns = list(above_15_percent_columns)
        if i in above_15_percent_columns:
            above_15_percent_columns.remove(i)
            
        data_dup_train = data_dup_train.drop(above_15_percent_columns, axis=1)
        data_dup_test = data_dup_test.drop(above_15_percent_columns, axis=1)

        # Train test split

        x_test = data_dup_test.drop(i, axis=1)
        x_test = pd.get_dummies(x_test, drop_first=True)
        x_test_columns = x_test.columns
        for k in x_test_columns:
            if x_test[k].dtype == 'float64':
                x_test[k] = x_test[[k]].apply(
                    lambda x: x.fillna(x.median()), axis=0)
            else:
                x_test[k] = x_test[[k]].apply(lambda x: x.fillna(
                    x_test[k].value_counts().index.max()))

        x_train = data_dup_train.drop(i, axis=1)
        x_train = pd.get_dummies(x_train, drop_first=True)
        x_train = x_train[x_test.columns]

        y_train = data_dup_train[[i]]
        if y_train[i].dtype == 'O':
            y_train[i] = label_encoder.fit_transform(y_train[i])
            y_train[[i]] = y_train[[i]].astype('int')

        # Building model
        if i in numerical_column_names:
            model_rf = RandomForestRegressor(n_estimators=100, max_depth=6)
        else:
            model_rf = RandomForestClassifier(n_estimators=100, max_depth=6)

        model_rf.fit(x_train, y_train)
        rf_score = model_rf.score(x_train, y_train)
        print('RandomForest Score :', rf_score)

        if i in numerical_column_names:
            model_lr = LinearRegression()
        else:
            model_lr = LogisticRegression()

        model_lr.fit(x_train, y_train)
        lr_score = model_lr.score(x_train, y_train)
        print('\nLogisticRegression Score :', lr_score)

        # Checking which model is better
        if rf_score > lr_score:
            print('\nFor', i, ' RandomForest performs better. So we will go with this.\n')
            model = model_rf
            Imputed_column_array.append({i: 'Random Forest'})
        else:
            print(
                '\n\nFor', i, ' Logistic Regression performs better. So we will go with this.')
            model = model_lr
            Imputed_column_array.append({i: 'Logistic Regression'})

        prediction = model.predict(x_test)
        print(prediction.dtype, '\n\n')
        if prediction.dtype == 'int32':
            prediction = label_encoder.inverse_transform(prediction)

        prediction_df = pd.DataFrame(prediction)
        #print('\n\n Predicted count of ', i , '  :' , prediction_df[0].value_counts())

        data_dup_test = data_dup_test.drop(i, axis=1)

        data_dup_test[i] = prediction

        data_dup_complete = pd.concat([data_dup_train_copy, data_dup_test])

        data_dup_complete = data_dup_complete.sort_index()

        predicted = data_dup_complete[[i]]

        data_null_treated = data_null_treated.drop(i, axis=1)

        data_null_treated[i] = predicted

    return (Imputed_column_array, data_null_treated)
