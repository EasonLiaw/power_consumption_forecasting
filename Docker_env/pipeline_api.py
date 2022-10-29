import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from Model_Training_Modules.validation_train_data import rawtraindatavalidation
from Model_Training_Modules.train_preprocessing import train_Preprocessor
from Model_Training_Modules.model_training import model_trainer
import os

DATABASE_LOG = "Training_Logs/Training_DB_Log.txt"
DATA_SOURCE = 'Training_Data_FromDB/Training_Data.csv'
PREPROCESSING_LOG = "Training_Logs/Training_Preprocessing_Log.txt"
RESULT_DIR = 'Intermediate_Train_Results/'
TRAINING_LOG = "Training_Logs/Training_Model_Log.txt"

def main():
    st.title("Power Consumption Forecasting")
    html_temp = """
    <div style="background-color:#7fc6ef;padding:3px">
    <h2 style="color:white;text-align:center;">Power Consumption Forecasting App </h2>
    <p></p>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    with st.expander("Model Training", expanded=True):
        if st.button("Training Data Validation"):
            trainvalidator = rawtraindatavalidation(
                dbname = 'energy', collectionname='consumption', file_object = DATABASE_LOG)
            folders = ['Training_Data_FromDB/','Intermediate_Train_Results/','Saved_Models/']
            trainvalidator.initial_data_preparation(
                folders = folders, datafilepath = "Training_Batch_Files/Energy_Consumption_Data.csv", compiledir= DATA_SOURCE)
            st.success(
                "This step of the pipeline has been completed successfully. Check the local files for more details.")
        if st.button("Exploratory Data Analysis"):
            if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
                st.error(
                    "Database has not yet inserted. Have u skipped Training Data Validation step?")
            else:
                preprocessor = train_Preprocessor(
                    file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
                preprocessor.eda()
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        if st.button("Training Data Preprocessing"):
            if 'Training_Data_FromDB' not in os.listdir(os.getcwd()):
                st.error(
                    "Database has not yet inserted. Have u skipped Training Data Validation step?")
            else:
                preprocessor = train_Preprocessor(
                    file_object= PREPROCESSING_LOG, datapath = DATA_SOURCE, result_dir= RESULT_DIR)
                preprocessor.data_preprocessing()
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        model_names = st.multiselect(
            "Select the following model you would like to train for model selection", options=['SARIMAX', 'ExponentialSmoothing', 'Prophet'])
        if st.button("Model Selection"):
            if not os.path.isdir(RESULT_DIR) or 'data.csv' not in os.listdir(RESULT_DIR):
                st.error(
                    "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
            else:
                trainer = model_trainer(file_object= TRAINING_LOG)
                data = pd.read_csv(RESULT_DIR + 'data.csv')
                trainer.model_selection(
                    data = data, num_trials = 20, folderpath = RESULT_DIR, model_names = model_names)
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
        model_name = st.selectbox(
            "Select the following model you would like to train for final model deployment", options=['SARIMAX', 'ExponentialSmoothing', 'Prophet'])
        if st.button("Final Model Training"):
            if not os.path.isdir(RESULT_DIR) or 'data.csv' not in os.listdir(RESULT_DIR):
                st.error(
                    "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
            elif not os.path.isdir(RESULT_DIR + model_name):
                st.error(
                    "Model algorithm selection has not been done. Have u skipped model selection step?")
            else:
                trainer = model_trainer(file_object= TRAINING_LOG)
                data = pd.read_csv(RESULT_DIR + 'data.csv')
                trainer.final_model_tuning(
                    data = data, num_trials = 20, folderpath = RESULT_DIR, model_name = model_name)
                st.success(
                    "This step of the pipeline has been completed successfully. Check the local files for more details.")
    with st.expander("Model Prediction"):
        if not os.path.isdir(RESULT_DIR) or 'data.csv' not in os.listdir(RESULT_DIR):
            st.error(
                "Data has not yet been preprocessed. Have u skipped Training Data Preprocessing step?")
        else:
            forecast_period = st.number_input('Enter number of periods you wish to forecast',min_value=1,max_value=96,step=1)
            if st.button('Forecast values'):
                if not os.path.isdir('Saved_Models/') or not os.listdir('Saved_Models/'):
                    st.error(
                        "No model has been saved yet. Have u skipped Final Model Training step?")
                else:
                    data = pd.read_csv(RESULT_DIR + 'data.csv')
                    model = joblib.load('Saved_Models/FinalModel.pkl')
                    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
                    data = data[-96:]
                    if type(model).__name__ != 'Prophet':
                        predicted_value = model.forecast(forecast_period)
                        predicted_value = pd.DataFrame(
                            predicted_value.reset_index())
                        predicted_value.columns = ['Timestamp','value']
                        predicted_value['Timestamp'] = pd.to_datetime(predicted_value['Timestamp'])
                    else:
                        future = model.make_future_dataframe(
                            forecast_period,freq='15T',include_history=False)
                        forecast = model.predict(future)
                        predicted_value = forecast[['ds','yhat']]
                        predicted_value.columns = ['Timestamp','value']
                    st.write(
                        "Predicted levels of power consumption for the next", str(forecast_period),"periods (15-minute interval):",predicted_value)
                    fig, ax = plt.subplots(figsize=(24,12))
                    plt.style.use('seaborn-whitegrid')
                    plt.rc('font', size=16)      
                    plt.rc('axes', titlesize=16)
                    plt.rc('axes', labelsize=18)
                    plt.rc('xtick', labelsize=16)
                    plt.rc('ytick', labelsize=16)
                    plt.rc('legend', fontsize=18)
                    plt.rc('figure', titlesize=24)
                    a = sns.lineplot(
                        data=data, x='Timestamp', y='allsum', ax=ax, label='actual',marker='o')
                    b = sns.lineplot(
                        data= predicted_value, x='Timestamp', y='value', ax=ax, label='forecast',linestyle='--',marker='o')
                    plt.title(
                        '15 Minute Average Forecast of Power Consumption Levels', fontsize=24)
                    plt.ylabel("Power consumption level")
                    plt.legend(loc='best')
                    st.pyplot(fig, clear_figure=True)

if __name__=='__main__':
    main()