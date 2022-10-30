'''
Author: Liaw Yi Xian
Last Modified: 30th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import optuna
import joblib
import time
from tsxv.splitTrainValTest import split_train_val_test_forwardChaining
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error
from Application_Logger.logger import App_Logger
from Application_Logger.exception import CustomException

random_state=120

class model_trainer:


    def __init__(self, file_object):
        '''
            Method Name: __init__
            Description: This method initializes instance of model_trainer class
            Output: None

            Parameters:
            - file_object: String path of logging text file
        '''
        self.file_object = file_object
        self.log_writer = App_Logger()
        self.optuna_selectors = {
            'SARIMAX': {'obj': model_trainer.arima_objective,'modelname': 'SARIMAX'},
            'ExponentialSmoothing': {'obj': model_trainer.es_objective,'modelname': 'ExponentialSmoothing'},
            'Prophet': {'obj': model_trainer.prophet_objective,'modelname': 'Prophet'}
        }


    def arima_objective(trial, train_data, val_data=None, final_model=False):
        '''
            Method Name: arima_objective
            Description: This method sets the objective function for SARIMAX model (with no seasonal order) by setting various hyperparameters for different Optuna trials.
            Output: Single floating point value that represents AIC score of given model on validation set.

            Parameters:
            - trial: Optuna trial object
            - train_data: Data from training set
            - val_data: Data from validation set
            - final_model: Indicator of whether hyperparameter tuning is done for final model deployment (True or False)
        '''
        p = trial.suggest_int('p',0,24)
        q = trial.suggest_int('q',0,24)
        reg = SARIMAX(endog=train_data, order=(p,0,q), initialization='approximate_diffuse')
        reg = reg.fit(disp=0)
        val_aic = reg.aic
        if final_model == False:
            yhat = reg.forecast(len(val_data))
            val_mape = mean_absolute_percentage_error(val_data,yhat)
            trial.set_user_attr("val_mape", val_mape)
            trial.set_user_attr("val_aic", val_aic)
        return val_aic


    def es_objective(trial, train_data, val_data=None, final_model=False):
        '''
            Method Name: es_objective
            Description: This method sets the objective function for Exponential Smoothing model by setting various hyperparameters for different Optuna trials.
            Output: Single floating point value that represents AIC score of given model on validation set.

            Parameters:
            - trial: Optuna trial object
            - train_data: Data from training set
            - val_data: Data from validation set
            - final_model: Indicator of whether hyperparameter tuning is done for final model deployment (True or False)
        '''
        seasonal = trial.suggest_categorical('seasonal',['add','mul','None'])
        seasonal_periods = 0 if seasonal == 'None' else 96
        seasonal = None if seasonal == 'None' else seasonal
        reg = ExponentialSmoothing(
            endog=train_data, seasonal=seasonal, seasonal_periods=seasonal_periods)
        reg = reg.fit()
        val_aic = reg.aic
        if final_model == False:
            yhat = reg.forecast(len(val_data))
            val_mape = mean_absolute_percentage_error(val_data,yhat)
            trial.set_user_attr("val_mape", val_mape)
            trial.set_user_attr("val_aic", val_aic)
        return val_aic


    def prophet_objective(trial, train_data, val_data=None, final_model=False):
        '''
            Method Name: prophet_objective
            Description: This method sets the objective function for Prophet model by setting various hyperparameters for different Optuna trials.
            Output: Single floating point value that represents MAPE score of given model on validation set.

            Parameters:
            - trial: Optuna trial object
            - train_data: Data from training set
            - val_data: Data from validation set
            - final_model: Indicator of whether hyperparameter tuning is done for final model deployment (True or False)
        '''
        changepoint_prior_scale = trial.suggest_float(
            'changepoint_prior_scale', 0.001, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float(
            'seasonality_prior_scale', 0.01, 10, log=True)
        holidays_prior_scale = trial.suggest_float(
            'holidays_prior_scale', 0.01, 10, log=True)
        reg = Prophet(
            changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale, holidays_prior_scale=holidays_prior_scale)
        train_data_copy = train_data.reset_index().copy()
        train_data_copy.columns = ['ds', 'y']
        reg.fit(train_data_copy)
        if final_model == False:
            future = pd.DataFrame(val_data.reset_index()['Timestamp'])
            future.columns = ['ds']
            forecast = reg.predict(future)
            val_mape = mean_absolute_percentage_error(val_data,forecast['yhat'])
            trial.set_user_attr("val_mape", val_mape)
        else:
            future = pd.DataFrame(train_data.reset_index()['Timestamp'])
            future.columns = ['ds']
            forecast = reg.predict(future)
            val_mape = mean_absolute_percentage_error(
                train_data,forecast['yhat'])
        return val_mape


    def optuna_optimizer(self, obj, n_trials, fold):
        '''
            Method Name: optuna_optimizer
            Description: This method creates a new Optuna study object if the given Optuna study object doesn't exist or otherwise using existing Optuna study object and optimizes the given objective function. In addition, the following plots and results are also created and saved:
            1. Hyperparameter Importance Plot
            2. Optimization History Plot
            3. Optuna study object
            4. Optimization Results (csv format)
            
            Output: Single best trial object
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold: Fold number from nested cross-validation in outer loop
        '''
        try:
            if f"OptStudy_{obj.__name__}_Fold_{fold}.pkl" in os.listdir(self.folderpath+obj.__name__):
                study = joblib.load(
                    self.folderpath+obj.__name__+f"/OptStudy_{obj.__name__}_Fold_{fold}.pkl")
            else:
                sampler = optuna.samplers.TPESampler(
                    multivariate=True, seed=random_state)
                study = optuna.create_study(
                    direction='minimize',sampler=sampler)
            study.optimize(
                obj, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
            trial = study.best_trial
            if trial.number !=0:
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                opt_fig = optuna.visualization.plot_optimization_history(study)
                param_imp_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/HP_Importances_{obj.__name__}_Fold_{fold}.png')
                opt_fig.write_image(
                    self.folderpath+ obj.__name__ +f'/Optimization_History_{obj.__name__}_Fold_{fold}.png')
            joblib.dump(
                study, self.folderpath + obj.__name__ + f'/OptStudy_{obj.__name__}_Fold_{fold}.pkl')
            study.trials_dataframe().to_csv(
                self.folderpath + obj.__name__ + f"/Hyperparameter_Tuning_Results_{obj.__name__}_Fold_{fold}.csv",index=False)
            del study
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        return trial


    def residual_diagnostics(
            self, modelname, figtitle, plotname, actual_value, pred_value):
        '''
            Method Name: residual_diagnostics
            Description: This method performs residual diagnostics from the model and saves the following plots within the given model class folder
            - Residual plot
            - ACF plot for residuals
            - PACF plot for residuals

            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - modelname: String name of model
            - figtitle: String that represents part of title figure
            - plotname: String that represents part of image name
            - actual_value: Actual target values from dataset
            - pred_value: Predicted target values
        '''
        try:
            plt.style.use('seaborn-whitegrid')
            plt.scatter(x = pred_value, y= np.subtract(pred_value,actual_value))
            plt.axhline(y=0, color='black')
            plt.title(
                f'Residual plot for {modelname} {figtitle} (Mean: {np.round(np.subtract(pred_value,actual_value).mean(),6)})')
            plt.ylabel('Residuals')
            plt.xlabel('Predicted Value')
            plt.savefig(
                self.folderpath+modelname+f'/Residual_Plot_{modelname}_{plotname}.png',bbox_inches='tight')
            plt.clf()
            plot_acf(np.subtract(pred_value,actual_value), lags=24)
            plt.savefig(
                self.folderpath+modelname+f'/ACF_Plot_{modelname}_{plotname}.png',bbox_inches='tight')
            plt.clf()
            plot_pacf(np.subtract(pred_value,actual_value), lags=24)
            plt.savefig(
                self.folderpath+modelname+f'/PACF_Plot_{modelname}_{plotname}.png',bbox_inches='tight')
            plt.clf()
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)


    def time_series_plot(self, modelname, data, pred_values):
        '''
            Method Name: time_series_plot
            Description: This method plots time series for actual values vs values predicted by given model and saves the following plots within the given model class folder.
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - modelname: String name of model
            - data: Dataset in pandas dataframe format
            - pred_values: Predicted target values
        '''
        try:
            plt.style.use('seaborn-whitegrid')
            plt.figure(figsize=(60,18))
            plt.rc('font', size=20)      
            plt.rc('axes', titlesize=24)
            plt.rc('axes', labelsize=26)
            plt.rc('xtick', labelsize=24)
            plt.rc('ytick', labelsize=24)
            plt.rc('legend', fontsize=24)
            plt.rc('figure', titlesize=32)
            plt.plot(data, label='Actual')
            plt.title(
                'Actual vs Predicted 15 Minute Average of Power Consumption Levels', fontsize=34)
            pred_data = pd.DataFrame(
                {'Timestamp':data.reset_index()['Timestamp'],'allsum': pd.Series(pred_values)})
            pred_data['Timestamp'] = pd.to_datetime(pred_data['Timestamp'])
            pred_data = pred_data.set_index('Timestamp')
            plt.plot(pred_data, label='Predicted')
            plt.legend(loc='best')
            plt.ylabel("Power consumption level")
            plt.tight_layout()
            plt.savefig(
                self.folderpath+modelname+f'/Actual_vs_Predicted_Plot_{modelname}.png',bbox_inches='tight')
            plt.clf()
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)


    def learning_curve_plot(self, modelname, data, best_trial):
        '''
            Method Name: learning_curve_plot
            Description: This method plots learning curve and saves plot within the given model class folder.
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - modelname: String name of model
            - data: Dataset in pandas dataframe format
            - best_trial: Optuna's best trial object from hyperparameter tuning
        '''
        try:
            train_size, train_mape_list, test_mape_list = [], [], []
            for i in range(200,int(len(data) * 0.7)+1,200):
                train, test = data[0:i], data[i:]
                if modelname == 'SARIMAX':
                    model = SARIMAX(
                        endog = train, order = (best_trial.params['p'], 0, best_trial.params['q']),initialization='approximate_diffuse').fit(disp=0)
                    train_pred =  model.predict(start=0, end=len(train)-1)
                    test_pred =  model.forecast(len(test))
                elif modelname == 'Prophet':
                    train_copy = train.reset_index().copy()
                    train_copy.columns = ['ds', 'y']
                    model = Prophet(
                        changepoint_prior_scale = best_trial.params['changepoint_prior_scale'], seasonality_prior_scale = best_trial.params['seasonality_prior_scale'], holidays_prior_scale = best_trial.params['holidays_prior_scale']).fit(train_copy)
                    train_future = pd.DataFrame(train_copy['ds'])
                    train_pred =  model.predict(train_future)['yhat'].values
                    test_future = pd.DataFrame(test.reset_index()['Timestamp'])
                    test_future.columns = ['ds']
                    test_pred =  model.predict(test_future)['yhat'].values
                else:
                    seasonal = None if best_trial.params['seasonal'] == 'None' else best_trial.params['seasonal']
                    seasonal_periods = 0 if best_trial.params['seasonal']== 'None' else 96
                    model = ExponentialSmoothing(
                        endog=train, seasonal = seasonal, seasonal_periods = seasonal_periods).fit()
                    train_pred =  model.predict(start=0, end=len(train)-1)
                    test_pred =  model.forecast(len(test))
                train_mape = mean_absolute_percentage_error(train_pred,train)
                test_mape = mean_absolute_percentage_error(test_pred,test)
                train_mape_list.append(train_mape)
                test_mape_list.append(test_mape)
                train_size.append(i)
            plt.style.use('seaborn-whitegrid')
            plt.grid(True)
            plt.plot(
                train_size, train_mape_list, label = 'Training Score', marker='.',markersize=14)
            plt.plot(
                train_size, test_mape_list, label = 'Validation Score', marker='.',markersize=14)
            plt.ylabel('Score')
            plt.xlabel('Training instances')
            plt.title(f'Learning Curve for {modelname}')
            plt.legend(frameon=True, loc='best')
            plt.savefig(
                self.folderpath+modelname+f'/LearningCurve_{modelname}.png',bbox_inches='tight')
            plt.clf()
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)


    def model_training(
            self, modelname, obj, train_data, n_trials, fold_num, val_data=None, final_model=False):
        '''
            Method Name: model_training
            Description: This method performs Optuna hyperparameter tuning using day-forward cross validation on given dataset. The best hyperparameters with the best pipeline identified is used for model training.
            
            Output: 
            - model_copy: Trained model object
            - best_trial: Optuna's best trial object from hyperparameter tuning

            On Failure: Logging error and raise exception

            Parameters:
            - modelname: String name of model
            - obj: Optuna objective function
            - train_data: Data from training set
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - fold_num: Indication of fold number for model training (can be integer or string "overall")
            - val_data: Data from validation set
            - final_model: Indicator of whether hyperparameter tuning is done for final model deployment (True or False)
        '''
        try:
            func = lambda trial: obj(trial, train_data, val_data, final_model)
            func.__name__ = modelname
            self.log_writer.log(
                self.file_object, f"Start hyperparameter tuning for {modelname} for fold {fold_num}")
            best_trial = self.optuna_optimizer(func, n_trials, fold_num)
            self.log_writer.log(
                self.file_object, f"Hyperparameter tuning for {modelname} completed for fold {fold_num}")
            self.log_writer.log(
                self.file_object, f"Finish hyperparameter tuning for {modelname} for fold {fold_num}")
            if modelname == 'SARIMAX':
                model_copy = SARIMAX(
                    endog = pd.concat([train_data, val_data]), order = (best_trial.params['p'], 0, best_trial.params['q']),initialization='approximate_diffuse')
                model_copy = model_copy.fit(disp=0)
            elif modelname == 'Prophet':
                train_val_data = pd.concat([train_data, val_data])
                train_val_data_copy = train_val_data.reset_index().copy()
                train_val_data_copy.columns = ['ds', 'y']
                model_copy = Prophet(
                    changepoint_prior_scale = best_trial.params['changepoint_prior_scale'], seasonality_prior_scale = best_trial.params['seasonality_prior_scale'], holidays_prior_scale = best_trial.params['holidays_prior_scale'])
                model_copy.fit(train_val_data_copy)
            else:
                seasonal = None if best_trial.params['seasonal'] == 'None' else best_trial.params['seasonal']
                seasonal_periods = 0 if best_trial.params['seasonal']== 'None' else 96
                model_copy = ExponentialSmoothing(
                    endog = pd.concat([train_data, val_data]), seasonal = seasonal, seasonal_periods = seasonal_periods)
                model_copy = model_copy.fit()
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        return model_copy, best_trial


    def hyperparameter_tuning(self, obj, modelname, n_trials, data):
        '''
            Method Name: hyperparameter_tuning
            Description: This method performs day-forward cross Validation on the entire dataset, where the dataset is split into multiple training sets, validation sets and test sets. Training set and validation set is used for hyperparameter tuning, while test set is used for model evaluation to obtain overall generalization error of model. In addition, the following intermediate results are saved for a given model class:
            1. Model_Performance_Results_by_Fold (csv file)
            2. Overall_Model_Performance_Results (csv file)
            3. Multiple plots from residual_diagnostics function
            
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - modelname: String name of model
            - n_trials: Number of trials for Optuna hyperparameter tuning
            - data: Dataset in pandas dataframe format
        '''
        try:
            input_train, forecast_train, input_val, forecast_val, input_test, forecast_test = split_train_val_test_forwardChaining(
                data['Timestamp'], numInputs=672, numOutputs=96, numJumps=96)
            data = data.set_index('Timestamp')
            mape_val_cv, mape_test_cv = [], []
            actual_values, pred_values = [], []
            for fold in range(len(input_train)):
                input_sub_train_data = data.loc[list(input_train[fold].ravel())].drop_duplicates()
                input_sub_val_data = data.loc[list(input_val[fold].ravel())].drop_duplicates()
                input_sub_test_data = data.loc[list(input_test[fold].ravel())].drop_duplicates()
                model_copy, best_trial = self.model_training(
                    modelname, obj, input_sub_train_data, n_trials, fold+1, input_sub_val_data)
                if modelname != 'Prophet':
                    aic_outer_val_value = model_copy.aic
                    val_pred = model_copy.forecast(len(input_sub_test_data))
                else:
                    future = pd.DataFrame(
                        input_sub_test_data.reset_index()['Timestamp'])
                    future.columns = ['ds']
                    forecast = model_copy.predict(future)
                    val_pred = forecast['yhat']
                actual_values.extend(input_sub_test_data['allsum'])
                pred_values.extend(val_pred)
                mape_outer_val_value = mean_absolute_percentage_error(
                    np.array(input_sub_test_data['allsum']), val_pred)
                cv_lists = [mape_val_cv, mape_test_cv]
                metric_values = [best_trial.user_attrs['val_mape'], mape_outer_val_value]
                for cv_list, metric in zip(cv_lists, metric_values):
                    cv_list.append(metric)
                self.log_writer.log(
                    self.file_object, f"Evaluating model performance for {modelname} on validation set completed for fold {fold+1}")
                if modelname == 'Prophet':
                    best_trial.user_attrs['val_aic'] = None
                    aic_outer_val_value = None
                optimized_results = pd.DataFrame({
                    'Model': modelname, 'best_params': str(best_trial.params), 'Outer_fold': fold+1,'mape_val_cv': best_trial.user_attrs['val_mape'], 'mape_test_cv': [mape_outer_val_value],'aic_val_cv': best_trial.user_attrs['val_aic'], 'aic_test_cv': [aic_outer_val_value]})
                optimized_results.to_csv(
                    self.folderpath+'Model_Performance_Results_by_Fold.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Model_Performance_Results_by_Fold.csv'))
                self.log_writer.log(
                    self.file_object, f"Optimized results for {modelname} model saved for fold {fold+1}")
                time.sleep(10)
            average_results = pd.DataFrame({
                'Models': modelname, 'mape_val_cv_avg': np.mean(mape_val_cv), 'mape_val_cv_std': np.std(mape_val_cv), 'mape_test_cv_avg': np.mean(mape_test_cv), 'mape_test_cv_std': np.std(mape_test_cv)}, index=[0])
            average_results.to_csv(
                self.folderpath+'Overall_Model_Performance_Results.csv', mode='a', index=False, header=not os.path.exists(self.folderpath+'Overall_Model_Performance_Results.csv'))
            self.residual_diagnostics(
                modelname, 'from cv', 'CV', actual_values, pred_values)
            self.log_writer.log(
                self.file_object, f"Average optimized results for {modelname} model saved")                
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)


    def final_overall_model(self, obj, modelname, data, n_trials):
        '''
            Method Name: final_overall_model
            Description: This method performs hyperparameter tuning on best model algorithm identified on entire dataset. The best hyperparameters identified are then used to train the entire dataset before saving model for deployment.
            In addition, the following intermediate results are saved for a given model class:
            1. Multiple plots from residual_diagnostics function
            2. Learning Curve image
            3. Line curve that represents actual vs predicted values
            
            Output: None
            On Failure: Logging error and raise exception

            Parameters:
            - obj: Optuna objective function
            - modelname: String name of model
            - data: Dataset in pandas dataframe format
            - n_trials: Number of trials for Optuna hyperparameter tuning
        '''
        self.log_writer.log(
            self.file_object, f"Start final model training on all data for {modelname}")
        try:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data = data.set_index('Timestamp')
            overall_model, best_trial = self.model_training(
                modelname, obj, data, n_trials, 'overall', final_model=True)
            if modelname == 'SARIMAX':
                pred_values = overall_model.predict(start=0,end=len(data)-1).values
            elif modelname == 'Prophet':
                future = pd.DataFrame(data.reset_index()['Timestamp'])
                future.columns = ['ds']
                forecast = overall_model.predict(future)
                pred_values = forecast['yhat'].values
            else:
                pred_values = overall_model.predict(start=0,end=len(data)-1).values
            plt.rcdefaults()
            self.residual_diagnostics(
                modelname, '- final model', 'final_model', data['allsum'].values, pred_values)
            self.learning_curve_plot(modelname, data, best_trial)
            self.time_series_plot(modelname, data, pred_values)
            joblib.dump(overall_model,'Saved_Models/FinalModel.pkl')
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, f"Finish final model training on all data for {modelname}")
        

    def model_selection(self, data, num_trials, folderpath, model_names):
        '''
            Method Name: model_selection
            Description: This method performs model algorithm selection using Day-Forward Cross Validation.
            Output: None

            Parameters:
            - data: Dataset in pandas dataframe format
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
            - model_names: List of model names provided as input by user for model selection
        '''
        self.log_writer.log(
            self.file_object, 'Start process of model selection')
        self.data = data
        self.num_trials = num_trials
        self.folderpath = folderpath
        self.model_names = model_names
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        for selector in self.model_names:
            obj = self.optuna_selectors[selector]['obj']
            modelname = self.optuna_selectors[selector]['modelname']
            path = os.path.join(self.folderpath, modelname)
            if not os.path.exists(path):
                os.mkdir(path)
            self.hyperparameter_tuning(
                obj = obj, modelname = modelname, n_trials = self.num_trials, data = data)
            time.sleep(10)
        overall_results = pd.read_csv(
            self.folderpath + 'Overall_Model_Performance_Results.csv')
        self.log_writer.log(
            self.file_object, f"Best model identified based on MAPE is {overall_results.iloc[overall_results['mape_test_cv_avg'].idxmin()]['Models']} with the following test score: {np.round(overall_results.iloc[overall_results['mape_test_cv_avg'].idxmin()]['mape_test_cv_avg'],4)} ({np.round(overall_results.iloc[overall_results['mape_test_cv_avg'].idxmin()]['mape_test_cv_std'],4)})")
        self.log_writer.log(
            self.file_object, 'Finish process of model selection')


    def final_model_tuning(self, data, num_trials, folderpath, model_name):
        '''
            Method Name: final_model_tuning
            Description: This method performs final model training from best model algorithm identified on entire dataset.
            Output: None

            Parameters:
            - data: Dataset in pandas dataframe format
            - num_trials: Number of Optuna trials for hyperparameter tuning
            - folderpath: String path name where all results generated from model training are stored.
            - model_name: Name of model selected for final model training.
        '''
        self.data = data
        self.num_trials = num_trials
        self.folderpath = folderpath
        self.model_name = model_name
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        self.log_writer.log(
            self.file_object, f"Start performing hyperparameter tuning on best model identified overall: {self.model_name}")
        obj = self.optuna_selectors[self.model_name]['obj']
        modelname = self.optuna_selectors[self.model_name]['modelname']
        self.final_overall_model(
            obj = obj, modelname = modelname, data = data, n_trials = self.num_trials)
        self.log_writer.log(
            self.file_object, f"Finish performing hyperparameter tuning on best model identified overall: {self.model_name}")