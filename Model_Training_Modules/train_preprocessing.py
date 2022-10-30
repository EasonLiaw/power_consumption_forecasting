'''
Author: Liaw Yi Xian
Last Modified: 30th October 2022
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Application_Logger.logger import App_Logger
from Application_Logger.exception import CustomException
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import plotly.express as px
import plotly.figure_factory as ff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class train_Preprocessor:


    def __init__(self, file_object, datapath, result_dir):
        '''
            Method Name: __init__
            Description: This method initializes instance of train_Preprocessor class
            Output: None

            Parameters:
            - file_object: String path of logging text file
            - datapath: String path where compiled data is located
            - result_dir: String path for storing intermediate results from running this class
        '''
        self.file_object = file_object
        self.datapath = datapath
        self.result_dir = result_dir
        self.log_writer = App_Logger()


    def extract_compiled_data(self):
        '''
            Method Name: extract_compiled_data
            Description: This method extracts data from a csv file and converts it into a pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, "Start reading compiled data from database")
        try:
            data = pd.read_csv(self.datapath)
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish reading compiled data from database")
        return data


    def data_cleaning(self,data):
        '''
            Method Name: data_cleaning
            Description: This method performs initial data cleaning on a given pandas dataframe.
            Output: A pandas dataframe
            On Failure: Logging error and raise exception
        '''
        try:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data = data.set_index('Timestamp')
            data = data.resample('15min')['allsum'].mean()
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        return data


    def remove_duplicated_rows(self, data):
        '''
            Method Name: remove_duplicated_rows
            Description: This method removes duplicated rows from a pandas dataframe.
            Output: A pandas DataFrame after removing duplicated rows. In addition, duplicated records that are removed will be stored in a separate csv file labeled "Duplicated_Records_Removed.csv"
            On Failure: Logging error and raise exception

            Parameters:
            - data: Dataframe object
        '''
        self.log_writer.log(
            self.file_object, "Start handling duplicated rows in the dataset")
        if len(data[data.duplicated()]) == 0:
            self.log_writer.log(
                self.file_object, "No duplicated rows found in the dataset")
        else:
            try:
                data[data.duplicated()].to_csv(
                    self.result_dir+'Duplicated_Records_Removed.csv', index=False)
                data = data.drop_duplicates(ignore_index=True)
            except Exception as e:
                self.log_writer.log(
                    self.file_object, str(CustomException(e,sys)))
                raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, "Finish handling duplicated rows in the dataset")
        return data
    

    def eda(self):
        '''
            Method Name: eda
            Description: This method performs exploratory data analysis on the entire dataset, while generating various plots/csv files for reference.
            Output: None
            On Failure: Logging error and raise exception
        '''
        self.log_writer.log(
            self.file_object, 'Start performing exploratory data analysis')
        try:
            path = os.path.join(self.result_dir, 'EDA')
            if not os.path.exists(path):
                os.mkdir(path)
            data = self.extract_compiled_data()
            data = self.data_cleaning(data)
            # Extract basic information about dataset
            pd.DataFrame({"name": data.name, "non-nulls": len(data)-data.isnull().sum(), "type": data.dtype},index=[0]).to_csv(self.result_dir + "EDA/Data_Info.csv",index=False)
            # Extract summary statistics about dataset
            data.describe().T.to_csv(
                self.result_dir + "EDA/Data_Summary_Statistics.csv")
            col_path = os.path.join(path, data.name)
            if not os.path.exists(col_path):
                os.mkdir(col_path)
            # Plotting boxplot of features
            fig2 = px.box(data,x=data.name,title=f"{data.name} Boxplot")
            fig2.write_image(
                self.result_dir + f"EDA/{data.name}/{data.name}_Boxplot.png")
            # Plotting kdeplot of features
            fig3 = ff.create_distplot(
                [data], [data.name], show_hist=False,show_rug=False)
            fig3.layout.update(
                title=f'{data.name} Density curve (Skewness: {np.round(data.skew(),4)})')
            fig3.write_image(
                self.result_dir + f"EDA/{data.name}/{data.name}_Distribution.png")
            # ACF and PACF plot
            plt.style.use('seaborn-whitegrid')
            plot_acf(data, lags=24)
            plt.savefig(
                self.result_dir + f"EDA/{data.name}/{data.name}_ACF.png")
            plt.clf()
            plot_pacf(data, lags=24)
            plt.savefig(
                self.result_dir + f"EDA/{data.name}/{data.name}_PACF.png")
            plt.clf()
            # Time series plot for one month
            plt.figure(figsize=(72,18))
            plt.plot(data, label='Original')
            plt.title(
                '15 Minute Average of Power Consumption Levels for September', fontsize=34)
            plt.legend(loc='best', fontsize=28)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(
                self.result_dir + f"EDA/{data.name}/{data.name}_Time_Series.png")
            plt.clf()
            # Time series plot for one day
            plt.figure(figsize=(30,18))
            plt.plot(data.iloc[:96], label='Original')
            plt.title(
                '15 Minute Average of Power Consumption Levels for One Day', fontsize=34)
            plt.legend(loc='best', fontsize=28)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(
                self.result_dir + f"EDA/{data.name}/{data.name}_Time_Series_One_Day.png")
            plt.clf()
            # Stationarity test
            pd.Series(adfuller(data)[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']).to_csv(self.result_dir + "EDA/Stationarity_Test.csv")
        except Exception as e:
            self.log_writer.log(self.file_object, str(CustomException(e,sys)))
            raise CustomException(e,sys)
        self.log_writer.log(
            self.file_object, 'Finish performing exploratory data analysis')


    def data_preprocessing(self):
        '''
            Method Name: data_preprocessing
            Description: This method performs all the data preprocessing tasks for the data.
            Output: None
        '''
        self.log_writer.log(self.file_object, 'Start of data preprocessing')
        data = self.extract_compiled_data()
        data = self.data_cleaning(data)
        data = self.remove_duplicated_rows(data = data)
        data.to_csv(self.result_dir+'data.csv')
        self.log_writer.log(self.file_object, 'End of data preprocessing')