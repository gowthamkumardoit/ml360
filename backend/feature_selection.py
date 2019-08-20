import Eda_imputation_new as eda
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chisquare, chi2_contingency
from flask import jsonify, make_response
import numpy as np
def feature_selection(data_null_treated, target):
    
    final_features_choosed = []
    ## Running Chisqaure for Categorical columns
    chi_columns = data_null_treated[eda.categorical_column_names].drop(target, axis = 1).columns
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
    data_complete = pd.concat([data_null_treated[chi_square_imp_feature], data_null_treated[eda.numerical_column_names]], axis = 1)
    data_complete[target] = data_null_treated[[target]]


    ## Running Random Forest for those important columns from chisquare
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=5000, min_samples_leaf=2)
    x = data_complete.drop(target, axis = 1)
    y = np.asarray(data_complete[target], dtype="|S6")
    x = pd.get_dummies(x,  drop_first=True)
    rf.fit(x,  y)
    rf.score(x ,y)

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

    return (list(final_features_choosed), feature_imp.to_dict(orient='records'))
