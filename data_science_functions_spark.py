# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 16:28:03 2018

@author: aszewczyk
"""
def df_add_error_cols(transformed_1m, transformed_2m, transformed_3m, col_target, col_predict):    
    """
    Created in 2018
    @author: aszewczyk
    Function that SPARK
    Input:
        @
    Returns:
        @
    """
    transformed_1m = transformed_1m\
                    .select('*', abs((transformed_1m[col_target] - transformed_1m[col_predict])
                                     /transformed_1m[col_target]*100)\
                    .alias(col_target+'abs_error_perc'))
    transformed_1m = transformed_1m\
                    .select('*', abs((transformed_1m[col_target] - transformed_1m[col_predict])
                                     )\
                    .alias(col_target+'abs_error'))

    transformed_2m = transformed_2m\
                    .select('*', abs((transformed_2m[col_target] - transformed_2m[col_predict])
                                     /transformed_2m[col_target]*100)\
                    .alias(col_target+'abs_error_perc'))
    transformed_2m = transformed_2m\
                    .select('*', abs((transformed_2m[col_target] - transformed_2m[col_predict])
                                     )\
                    .alias(col_target+'abs_error'))

    transformed_3m = transformed_3m\
                    .select('*', abs((transformed_3m[col_target] - transformed_3m[col_predict])
                                     /transformed_3m[col_target]*100)\
                    .alias(col_target+'abs_error_perc'))
    transformed_3m = transformed_3m\
                    .select('*', abs((transformed_3m[col_target] - transformed_3m[col_predict])
                                     )\
                    .alias(col_target+'abs_error'))
    
    return(transformed_1m, transformed_2m, transformed_3m)

def df_model_evaluation_spark(transformed_1m,
                        transformed_2m,
                        transformed_3m,
                        col_target,
                        i_value, j_value, k_value,
                        i_param, j_param, k_param,
                        logger):
    """
    Created in 2018
    @author: aszewczyk
    Function that
    Input:
        @
    Returns:
        @
    """   
    logger.info('Used parameters are: \n subsamplingRate: ' +\
    '\n ' + i_param + ': ' + str(i_value) +\
    '\n ' + j_param + ': ' + str(j_value) +\
    '\n ' + k_param + ': ' + str(k_value))

    logger.info('MAPE for the current month is ' +
          str(round(transformed_1m.agg({col_target+'abs_error_perc': 'avg'}).collect()[0][0], 3)) + ' %')
    logger.info('MAPE for the next month is ' +
          str(round(transformed_2m.agg({col_target+'abs_error_perc': 'avg'}).collect()[0][0], 3)) + ' %')
    logger.info('MAPE for the next next month is ' +
          str(round(transformed_3m.agg({col_target+'abs_error_perc': 'avg'}).collect()[0][0], 3)) + ' %')
    logger.info('MAE for the current month is ' +
          str(round(transformed_1m.agg({col_target+'abs_error': 'avg'}).collect()[0][0], 3)) + ' ')
    logger.info('MAE for the next month is ' +
          str(round(transformed_2m.agg({col_target+'abs_error': 'avg'}).collect()[0][0], 3)) + ' ')
    logger.info('MAE for the next next month is ' +
          str(round(transformed_3m.agg({col_target+'abs_error': 'avg'}).collect()[0][0], 3)) + ' ')

    agg_day_1m = transformed_1m.groupBy('dt_flight_date_local', 'cd_airport_pair')\
                               .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_day_1m = agg_day_1m.select('*', abs(agg_day_1m['sum(pax_seat)'] - agg_day_1m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_day_1m = agg_day_1m.select('*', abs((agg_day_1m['sum(pax_seat)'] - agg_day_1m['sum(pax_seat_pred)'])
                                           /agg_day_1m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on day-airport pair level for the current month is ' +
                 str(round(agg_day_1m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on day-airport pair level for the current month is ' +
                 str(round(agg_day_1m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_month_1m = transformed_1m.groupBy('dt_flight_year_month', 'cd_airport_pair')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_month_1m = agg_month_1m.select('*', abs(agg_month_1m['sum(pax_seat)'] - agg_month_1m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_month_1m = agg_month_1m.select('*', abs((agg_month_1m['sum(pax_seat)'] - agg_month_1m['sum(pax_seat_pred)'])
                                               /agg_month_1m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on month-airport pair level for the current month is ' +
                 str(round(agg_month_1m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on month-airport pair level for the current month is ' +
                 str(round(agg_month_1m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')



    agg_day_2m = transformed_2m.groupBy('dt_flight_date_local', 'cd_airport_pair')\
                               .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_day_2m = agg_day_2m.select('*', abs(agg_day_2m['sum(pax_seat)'] - agg_day_2m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_day_2m = agg_day_2m.select('*', abs((agg_day_2m['sum(pax_seat)'] - agg_day_2m['sum(pax_seat_pred)'])
                                           /agg_day_2m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on day-airport pair level for the next month is ' +
                 str(round(agg_day_2m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on day-airport pair level for the next month is ' +
                 str(round(agg_day_2m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_month_2m = transformed_2m.groupBy('dt_flight_year_month', 'cd_airport_pair')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_month_2m = agg_month_2m.select('*', abs(agg_month_2m['sum(pax_seat)'] - agg_month_2m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_month_2m = agg_month_2m.select('*', abs((agg_month_2m['sum(pax_seat)'] - agg_month_2m['sum(pax_seat_pred)'])
                                               /agg_month_2m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on month-airport pair level for the next month is ' +
                 str(round(agg_month_2m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on month-airport pair level for the next month is ' +
                 str(round(agg_month_2m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')



    agg_day_3m = transformed_3m.groupBy('dt_flight_date_local', 'cd_airport_pair')\
                               .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_day_3m = agg_day_3m.select('*', abs(agg_day_3m['sum(pax_seat)'] - agg_day_3m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_day_3m = agg_day_3m.select('*', abs((agg_day_3m['sum(pax_seat)'] - agg_day_3m['sum(pax_seat_pred)'])
                                           /agg_day_3m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on day-airport pair level for the next next month is ' +
                 str(round(agg_day_3m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on day-airport pair level for the current month is ' +
                 str(round(agg_day_3m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_month_3m = transformed_3m.groupBy('dt_flight_year_month', 'cd_airport_pair')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_month_3m = agg_month_3m.select('*', abs(agg_month_3m['sum(pax_seat)'] - agg_month_3m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_month_3m = agg_month_3m.select('*', abs((agg_month_3m['sum(pax_seat)'] - agg_month_3m['sum(pax_seat_pred)'])
                                               /agg_month_3m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on month-airport pair level for the next next  month is ' +
                 str(round(agg_month_3m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on month-airport pair level for the next next month is ' +
                 str(round(agg_month_3m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')

    agg_days_1m = transformed_1m.groupBy('dt_flight_date_local')\
                   .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_days_1m = agg_days_1m.select('*', abs(agg_days_1m['sum(pax_seat)'] - agg_days_1m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_days_1m = agg_days_1m.select('*', abs((agg_days_1m['sum(pax_seat)'] - agg_days_1m['sum(pax_seat_pred)'])
                                           /agg_days_1m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on daily level for the current month is ' +
                 str(round(agg_days_1m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on daily level for the current month is ' +
                 str(round(agg_days_1m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_months_1m = transformed_1m.groupBy('dt_flight_year_month')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_months_1m = agg_months_1m.select('*', abs(agg_months_1m['sum(pax_seat)'] - agg_months_1m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_months_1m = agg_months_1m.select('*', abs((agg_months_1m['sum(pax_seat)'] - agg_months_1m['sum(pax_seat_pred)'])
                                               /agg_months_1m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on monthly level for the current month is ' +
                 str(round(agg_months_1m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on monthly level for the current month is ' +
                 str(round(agg_months_1m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')




    agg_days_2m = transformed_2m.groupBy('dt_flight_date_local')\
                               .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_days_2m = agg_days_2m.select('*', abs(agg_days_2m['sum(pax_seat)'] - agg_days_2m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_days_2m = agg_days_2m.select('*', abs((agg_days_2m['sum(pax_seat)'] - agg_days_2m['sum(pax_seat_pred)'])
                                           /agg_days_2m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on daily level for the next month is ' +
                 str(round(agg_days_2m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on daily level for the next month is ' +
                 str(round(agg_days_2m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_months_2m = transformed_2m.groupBy('dt_flight_year_month')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_months_2m = agg_months_2m.select('*', abs(agg_months_2m['sum(pax_seat)'] - agg_months_2m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_months_2m = agg_months_2m.select('*', abs((agg_months_2m['sum(pax_seat)'] - agg_months_2m['sum(pax_seat_pred)'])
                                               /agg_months_2m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on monthly level for the next month is ' +
                 str(round(agg_months_2m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on monthly level for the next month is ' +
                 str(round(agg_months_2m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')



    agg_days_3m = transformed_3m.groupBy('dt_flight_date_local')\
                               .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_days_3m = agg_days_3m.select('*', abs(agg_days_3m['sum(pax_seat)'] - agg_days_3m['sum(pax_seat_pred)'])\
                                   .alias('daily_abs_error'))
    agg_days_3m = agg_days_3m.select('*', abs((agg_days_3m['sum(pax_seat)'] - agg_days_3m['sum(pax_seat_pred)'])
                                           /agg_days_3m['sum(pax_seat)']*100)\
                           .alias('daily_abs_perc_error'))

    logger.info('MAE agg on daily level for the next next month is ' +
                 str(round(agg_days_3m.agg({'daily_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on daily level for the next next month is ' +
                 str(round(agg_days_3m.agg({'daily_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')


    agg_months_3m = transformed_3m.groupBy('dt_flight_year_month')\
                         .agg({'pax_seat': 'sum', 'pax_seat_pred': 'sum'})
    agg_months_3m = agg_months_3m.select('*', abs(agg_months_3m['sum(pax_seat)'] - agg_months_3m['sum(pax_seat_pred)'])\
                                      .alias('monthly_abs_error'))
    agg_months_3m = agg_months_3m.select('*', abs((agg_months_3m['sum(pax_seat)'] - agg_months_3m['sum(pax_seat_pred)'])
                                               /agg_months_3m['sum(pax_seat)']*100)\
                               .alias('monthly_abs_perc_error'))            

    logger.info('MAE agg on monthly level for the next next month is ' +
                 str(round(agg_months_3m.agg({'monthly_abs_error': 'avg'}).collect()[0][0], 3)))
    logger.info('MAPE agg on monthly level for the next next month is ' +
                 str(round(agg_months_3m.agg({'monthly_abs_perc_error': 'avg'}).collect()[0][0], 3)) + ' %')
    

    
    feature_importances_1m = pd.DataFrame(rf_def_1m.feature_importances_,
                                   index = X_test_1m.columns,
                                   columns=['importance']).sort_values('importance', 
                                           ascending=False)

    logger.info('Feature importance for the current month model \n column' + str(feature_importances_1m))
    
    feature_importances_2m = pd.DataFrame(rf_def_2m.feature_importances_,
                                       index = X_test_2m.columns,
                                       columns=['importance']).sort_values('importance', 
                                               ascending=False)
    
    logger.info('Feature importance for the next month model \n column' + str(feature_importances_2m))
    
    feature_importances_3m = pd.DataFrame(rf_def_3m.feature_importances_,
                                       index = X_test_3m.columns,
                                       columns=['importance']).sort_values('importance', 
                                               ascending=False)
    
    logger.info('Feature importance for the next next month model \n column' + str(feature_importances_3m))