
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, classification_report, roc_curve
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

stdscaler = StandardScaler()
labelencoder = LabelEncoder()

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

def RandomForest_modelling(data, target):

    X = data.drop(target,axis=1)
    Y = data[[target]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 7)

    param_grid = {'n_estimators':(100,150),'min_samples_split':np.arange(2,6),'max_depth':(5,6)}
    gs = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=10)
    gs.fit(X_train,Y_train)
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = RandomForestClassifier(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_prob = model.predict_proba(X_test)
    Accuracy = metrics.accuracy_score(Y_test, prediction)
    print('Accuracy :', Accuracy )
    F1_score = metrics.f1_score(Y_test, prediction, pos_label=2, average=None)
    print('F1_square :', F1_score )
    AUC_score = metrics.roc_auc_score(Y_test, prediction_prob[:,1])
    print('Test_auc :', AUC_score)

def GB_modelling(data, target):

    X = data.drop(target,axis=1)
    Y = data[[target]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 7)

    param_grid = {'n_estimators':(100,150),'min_samples_split':np.arange(2,6),'max_depth':(5,6)}
    gs = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,cv=10)
    gs.fit(X_train,Y_train)
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = GradientBoostingClassifier(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    prediction_prob = model.predict_proba(X_test)
    Accuracy = metrics.accuracy_score(Y_test, prediction)
    print('Accuracy :', Accuracy )
    F1_score = metrics.f1_score(Y_test, prediction, pos_label=2, average=None)
    print('F1_square :', F1_score )
    AUC_score = metrics.roc_auc_score(Y_test, prediction_prob[:,1])
    print('Test_auc :', AUC_score)

def AdjustedRsqaure(N, p, R_square):
    # N = length of dataset, p = Number of predictors 
    result = 1 - (1 - R_square) * (N - 1) / (N - p - 1)
    return result

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
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('LinearRegression :', RMSE)
    return RMSE, R_square , Adjusted_r_sqaure

def RandomForest_modelling_regressor(data, target):
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
    n_estimators_gv, min_sample_leaf_gv, max_depth_gv = gs.best_params_['n_estimators'], gs.best_params_['max_depth'], gs.best_params_['min_samples_split']

    model = RandomForestRegressor(max_depth =max_depth_gv , n_estimators=n_estimators_gv, min_samples_leaf=min_sample_leaf_gv)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('Random forest : ', RMSE)
    return RMSE, R_square, Adjusted_r_sqaure

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
    prediction = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(Y_test, prediction))
    R_square = metrics.r2_score(Y_test, prediction)
    Adjusted_r_sqaure = AdjustedRsqaure(N, p, R_square)
    print('KNN :', RMSE)
    return RMSE, R_square, Adjusted_r_sqaure


