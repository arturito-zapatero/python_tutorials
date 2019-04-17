# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 12:50:28 2018

@author: aszewczyk
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sys import platform
if platform == "linux" or platform == "linux2":
    from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import logging
import pandas as pd
import os


def load_data_from_csv(path, file_name, logger):
    logger.info('Loading Data Starts')
    try:
        bt_data = pd.read_csv(path + file_name)
#    except (SystemExit, KeyboardInterrupt):
#        raise
    except Exception, e:
        logger.error('Failed to open file: ', exc_info=True)
        raise
    logger.info('Loading Data Success')
    return bt_data


def set_parameters(model_selection, logger):
    """
    Defines parameters for the model, depeneding on the model selected.
    Can be done using .txt file in future
    @model_selection - selected model
    @param_test_eval - if nfolds=1 no CV is made, param_test_eval are used as
    parameters for model. Only evalaution using test set is conducted. It is 
    also possible to use multiple parameters for this evaluation.
    @param_grid - if nfolds>1 CV is made using param_grid as search grid and
    the best set of parameters are used for model evaluation
    """
    if model_selection == 'rf': 
        param_test_eval = pd.DataFrame({'max_features' : pd.Series([75]), 
                                     'n_estimators'    : pd.Series([ 100]),
                                     'min_samples_leaf' : pd.Series([7])})
    
        max_features = [5, 10, 20]
        n_estimators = [ 100, 150, 300]
        min_samples_leaf = [   1,   1,  1]
        max_features = [20]
        n_estimators = [300]
        min_samples_leaf = [1]
        
        param_grid = {'max_features': max_features,
                      'n_estimators' : n_estimators,
                      'min_samples_leaf' : min_samples_leaf}
        
    if model_selection == 'svr': 
        param_test_eval = pd.DataFrame({'max_features' : pd.Series([75]), 
                                        'n_estimators'    : pd.Series([ 100]),
                                        'min_samples_leaf' : pd.Series([7])})
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}

    if model_selection == 'xgb': 
        param_test_eval = pd.DataFrame({'min_child_weight' : pd.Series([4]), 
                                        'gamma'     : pd.Series([0.5]),
                                        'subsample' : pd.Series([0.6]),
                                        'colsample_bytree' : pd.Series([1.0]),
                                        'max_depth' : pd.Series([4])})
      
        param_grid = {'min_child_weight':[4,5],
              'gamma':[i/10.0 for i in range(3,6)],
              'subsample':[i/10.0 for i in range(6,11)],
              'colsample_bytree':[i/10.0 for i in range(6,11)],
              'max_depth': [2,3,4]}    
#            param_grid = {'min_child_weight':[4,5],
#                          'gamma':[i/10.0 for i in range(3,4)],
#                          'subsample':[i/10.0 for i in range(6,7)],
#                          'colsample_bytree':[i/10.0 for i in range(6,7)],
#                          'max_depth': [2]}   

    return(param_test_eval, param_grid)

def bt_add_trend_column_monthly(data, year_variable, month_variable, logger):
  
    index_month_year = data\
                         .loc[:,[year_variable, month_variable]]\
                         .groupby([year_variable, month_variable])\
                         .count()\
                         .reset_index()
    
    #create calendar with index for each year/month (trend index)
    calendar_tmp = pd.DataFrame(np.nan, index=np.arange(len(index_month_year)), columns=['month', 'year'])
    for i in range(len(index_month_year)):
        calendar_tmp.loc[i, 'month'] = index_month_year.loc[i,month_variable]
        calendar_tmp.loc[i, 'year'] = index_month_year.loc[i,year_variable]
    
    calendar_tmp = calendar_tmp.reset_index().rename(columns={'index':'trend_index'})
    
    #merge trend with the original df
    data = calendar_tmp.merge(data,
                              left_on=['month', 'year'],
                              right_on=[month_variable, year_variable],
                              how='left')\
                             .drop(['month', 'year'], axis=1)
    
    return(data)


def data_ohe(data, ohe_variables, logger):
    """
    Depreceted, one can use standard function form pandas:
    
    data = pd.get_dummies(data, columns=ohe_variables, drop_first=True, prefix = ohe_variables)
    
    One hot encoding using pandas for multiple columns within the data frame. 
    Converts categorical variable into dummy/indicator variables, categorical 
    columns are afterward removed.
    @data - pandas d.f. that includes the columns for convertion
    @ohe_variables - list with variables' names to be converted
    @logger - logger to write the function log
    """
    logger.info('One hot encoding starts using variable(s): ')
    try:
        ohe_columns_out_names = list()
        for i in ohe_variables:
            logger.info(i)
            if i in data.columns:
                dummies = pd.get_dummies(data[i], drop_first = True, prefix = i)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop([i], axis=1)
                ohe_columns_out_names = ohe_columns_out_names + list(dummies.columns)
                
    except Exception, e:
        logger.error('Problem with one hot encoding: ', exc_info=True)
        raise

    logger.info('One hot encoding success')
    return data, ohe_columns_out_names







def split_data_random(data, j, columns_to_drop, target_variable, use_val_set, logger):
        #print ('Analizing data for All Routes for test set betweeen ' + test_start + ' and ' + test_end + ': start')
       
    X_train = np.asarray(data.query('trend_index >= 0  & trend_index <= @j')\
                                .drop(columns_to_drop, axis = 1)\
                                .astype(float))
                                 
    y_train = np.asarray(data.query('trend_index >= 0  & trend_index <= @j').loc[:,target_variable])\
                .astype(float)
    
    #one month used for test only, the month following directly the training set
    X_test = np.asarray(data.query('trend_index >= @j+1  & trend_index <= @j+1')\
                                .drop(columns_to_drop, axis = 1)\
                                .astype(float))
                                 
    y_test =  pd.DataFrame(data.query('trend_index >= @j+1  & trend_index <= @j+1').loc[:,target_variable])\
                .astype(float)
     
        #correct!
    if use_val_set:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)
    
    return(X_train, X_test, y_train, y_test)

def split_data_time_series(data, split_variable, split_value, predictor_variables, target_variable, use_val_set, logger):
    """
    Splits model data into train and test set using an ordered index rather than 
    random split. Useful for time series. Also drops the columns that are not 
    included in target/predictor variables.
    @data - pandas d.f. with data to be split.
    @split_variable - string that specifies the variable is to be used for split
    @split_index - numeric value of the split variable which be used for split. E.g.
    for split_variable = 'at_year' and split_index = 2049 the data up to 2049 
    (including) are used as training set, data since 2050 as test set
    @predictor_variables - predictors to be included in X-train and X_test
    @target_variable - target variable to be included in y-train and y_test
    @use_val_set - should the train set should be further split into validation set,
    not implemented as far
    @logger - logging handle
    @X_train - train set with predictor variables, numpy array - no column name 
    and index information, but required as an input for fit modue of the scikit learn
    @X_test - test set with predictor variables, pandas data frame - is not used
    for scikit learn, hence we leave it as pandas df
    @y_train - train set with target variable, numpy array
    @y_test - test set with target variable, pandas data frame
    """
    X_train =  np.asarray(data\
                          .loc[(data[split_variable] <= split_value), predictor_variables]\
                          .astype(float))
    y_train =  np.asarray(data\
                          .loc[(data[split_variable] <= split_value), target_variable]\
                          .astype(float))
    X_test =  pd.DataFrame(data\
                          .loc[(data[split_variable] > split_value), predictor_variables]\
                          .astype(float))
    y_test =  pd.DataFrame(data\
                          .loc[(data[split_variable] > split_value), target_variable]\
                          .astype(float))
    
#correct!
#    if use_val_set:
#        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)
    
    return(X_train, X_test, y_train, y_test)


def add_lag_features(data, lags_num, lags_dist, lags_shift, date_column, lag_columns, unique_columns, logger):
#    #a list with a column with date which is used to create lag features
#    date_column = ['trend_index']
#    #a list with columns with metrics which should be used as lag feature columns
#    lag_columns = ['met_revenue_total', 'met_pax_total']
#    #a list with columns that makes the data frame unique
#    unique_columns = ['atr_code_airline', 'atr_num_flight', 'atr_route_macro']
#    #number of lag features
#    lags_num = 4
#    #distance in terms of column date_column for each lag feature
#    lags_dist = 1
#    #the shift of the first lag feature, lags_shift=0 means no shift 
#    lags_shift = 1 
    
    
    data_tmp = data.copy()
    lag_columns_list = []
    for i in range(lags_num):
        #create temporary distance for i-th lag
        lags_dist_tmp = lags_dist*(i) + lags_shift
        #add date column with i-th shift
        data_tmp[date_column[0] + '_lag_' + str(lags_dist_tmp)] = data_tmp.loc[:,date_column[0]] - lags_dist_tmp
        
        data_tmp = \
        data_tmp.merge(data.loc[:,lag_columns + unique_columns + date_column],
                                      how='left',
                                      left_on=unique_columns +  [date_column[0] + '_lag_' + str(lags_dist_tmp)],
                                      right_on=unique_columns + date_column,
                                      suffixes=['','_lag_'+str(lags_dist_tmp)])
        
        #save the column names for lag columns
        lag_columns_list = [x + '_lag_'+str(lags_dist_tmp) for x in lag_columns] + lag_columns_list
        data_tmp = data_tmp.drop(date_column[0] + '_lag_' + str(lags_dist_tmp), axis=1)
    
    
    return data_tmp, lag_columns_list


def model_calc(X_train, X_test, y_train, y_test, column_names, param_test_eval, eval_dates, param_grid, nfolds, use_val_set, logger):     
        
    if nfolds==1:
        print('no CV')
        model_def = RandomForestRegressor(n_estimators=int(param_test_eval.n_estimators),  
                                          max_features=(param_test_eval.max_features), 
                                          min_samples_leaf=int(param_test_eval.min_samples_leaf),
                                          max_depth=int(param_test_eval.max_depth),
                                          oob_score=False,
                                          random_state = 1,
                                          n_jobs=-1)
        
        model_def = RandomForestRegressor(n_estimators=100,  
                                          max_features='sqrt', 
                                          min_samples_leaf=1,
                                          max_depth=5,
                                          oob_score=False,
                                          random_state = 1,
                                          n_jobs=-1)
        model_def.fit(X_train, y_train)
        
        y_hats = model_def.predict(X_test)
        
    if nfolds>1:
        print('CV')
        model_def = RandomForestRegressor(n_estimators=100,n_jobs=6)
        grid_search = GridSearchCV(model_def, param_grid, cv=nfolds)
        grid_search.fit(X_train, y_train)
        y_hats = grid_search.predict(X_test)
        print grid_search.best_params_
        print grid_search.cv_results_
        logger.info(grid_search.best_params_)
        best_params = grid_search.best_params_
        
    model_selection=='rf'
    if (model_selection=='rf'):
        print "Features sorted by their score:"
        print sorted(zip(map(lambda x: round(x, 4), model_def.feature_importances_), column_names), 
        reverse=True)
        print 'Out of the bag score: %.3f' % model_def.oob_score_
        #X_s = model_def.transform(X_train)
 
    y_test['pred_values'] = y_hats
    
    #data.loc[y_test.index, 'abt_pred'] = y_test['pred_values']
        
    #if nfolds==1:        
    #    mean_error_model = model_eval(data=data, pred_eval=True, logger=logger)
    
    if nfolds==1:
        return(y_test)#, mean_error_model)
    else:
        return(y_test, grid_search, best_params)

def model_eval_mae(results_model, target_variable, pred_variable, logger):
    """
    Calculates mean absolute error for all instances, also for each instance
    adds a column 'ae' (absolute error) in the original df with abs error for
    this instance
    @results_model - pandas df that contains the target variable column and 
    a column with results of the model predictions
    @target_variable - column with target variable
    @pred_variable - column with model predictions for target variable
    @mae - mean absolute error
    """
    results_model['ae'] = results_model.loc[:,pred_variable] - results_model.loc[:, target_variable]
    mae  = sum(abs(results_model['ae'])*1.00)/results_model.shape[0]
    
    return(results_model, mae)

def model_eval_mape(results_model, target_variable, pred_variable, logger):
    """
    Calculates mean absolute percentage error for all instances, also for each
    instance adds a column 'ape' (absolute percentageerror) in the original df 
    with abs error for this instance
    @results_model - pandas df that contains the target variable column and 
    a column with results of the model predictions
    @target_variable - column with target variable
    @pred_variable - column with model predictions for target variable
    @mape - mean absolute percentage error
    """
    results_model['ape'] = \
    abs((results_model.loc[:, target_variable] - results_model.loc[:,pred_variable])/results_model.loc[:, target_variable])*100
    
    results_model = results_model.replace([np.inf, -np.inf], np.nan)
    results_model = results_model.dropna(how='any')
    mape  = sum((results_model['ape'])*1.00)/results_model.shape[0]
    
    return(results_model, mape)



#        if model_selection=='svr':
#            if nfolds==1:
#                model_def = svm.SVR(C=0.001, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#                                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#            
#            if nfolds>1:
#                grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, nfolds=nfolds)
#                grid_search.fit(X_train, y_train)
#                grid_search.best_params_
#                print grid_search.best_params_
#        
#                        
#        if model_selection=='xgb':
#            if use_val_set:
#                eval_set = [(X_val, y_val)]
#            
#            if nfolds==1:
#                logger.info('Training using model: ' + model_selection)
#                model_def = XGBRegressor(min_child_weight=param_test_eval.min_child_weight, 
#                                        gamma=param_test_eval.gamma,
#                                        subsample=param_test_eval.subsample,
#                                        colsample_bytree=param_test_eval.colsample_bytree,
#                                        max_depth=int(param_test_eval.max_depth),
#                                        learning_rate=0.1,
#                                        nthread=4,
#                                        seed = 23)
#                print model_def
#                if use_val_set:
#                    print 'Using evaluation set'
#                    model_def.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=False)
#                else:
#                    model_def.fit(X_train, y_train)
#                
#                y_hats = model_def.predict(X_test)
#                var_importance = get_xgb_imp(model_def, names)
#                print 'Variable Importance'
#                print var_importance
#                
#            if nfolds>1:
#                print 'CV for model: ' + model_selection
#                logger.info('CV for model: ' + model_selection)
#                model_def = XGBRegressor(nthread=4) 
#                grid_search = GridSearchCV(model_def, param_grid, cv=nfolds)
#                if use_val_set:
#                    grid_search.fit(X_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=True)
#                else:
#                    grid_search.fit(X_train, y_train)
#                grid_search.best_params_
#                print(r2_score(y_test, grid_search.best_estimator_.predict(X_test))) 
#                print grid_search.best_estimator_
#                logger.info(grid_search.best_estimator_)
#                y_hats = grid_search.best_estimator_.predict(X_test)
#                
#                #xgb.plot_importance(bst)