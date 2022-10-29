# Power Consumption Forecasting Project

## Background
---

<img src="https://user-images.githubusercontent.com/34255556/198823730-39af6d71-f24f-432e-8d77-be1e6676875f.png" width="600">

In the current digital age, beverage manufacturing companies have an increase in demand for using more clean energy, which leads to a rise in demand for using IIOT (Industrial Internet of Things) for better efficiency of energy management and reducing operational costs. Sensors installed on various machines within a beverage manufacturing life cycle provide real-time data, which is critical for monitoring the power consumption of individual types of equipment and overall power consumption for future predictive maintenance instantaneously.

For this project, the main goal is to deploy initial time series models that help to monitor and forecast future power consumption levels of a beverage manufacturing factory. As real-time data flows in continuously from IoT sensors at specific small intervals, these time series models will need to be trained on a regular basis to capture fluctuations of overall power consumption for better monitoring and reducing energy waste of equipment. While power consumption data grows in size over time, it is more feasible to monitor the overall power consumption of a factory on a more coarse-grained level (i.e. on a 15-minute interval basis) from very granular data.

Dataset is provided in .csv format by client under Training_Batch_Files folder for model training. (not included in this repository due to data confidentiality reasons)

For model prediction, a web API is used (created using StreamLit) for user input. Note that results generated from model prediction along with user inputs can be stored in various formats (i.e. in CSV file format or another database).

## Contents
- [Code and Resources Used](#code-and-resources-used)
- [Model Training Setting](#model-training-setting)
- [Project Findings](#project-findings)
  - [EDA](#1-eda-exploratory-data-analysis)
  - [Best time series model](#2-best-time-series-model)
  - [Summary of model evaluation metrics from best time series model](#3-summary-of-model-evaluation-metrics-from-best-time-series-model)
  - [Residual diagnostics from final model trained](#4-residual-diagnostics-from-final-model-trained)
  - [Time series plot (Actual vs Predicted) from best time series model](#5-time-series-plot-actual-vs-predicted-from-best-time-series-model)
  - [Learning Curve Analysis](#6-learning-curve-analysis)
- [CRISP-DM Methodology](#crisp-dm-methodology)
- [Project Architecture Summary](#project-architecture-summary)
- [Project Folder Structure](#project-folder-structure)
- [MongoDB Atlas Setup](#mongodb-atlas-setup)
- [Project Instructions (Local Environment)](#project-instructions-local-environment)
- [Project Instructions (Docker)](#project-instructions-docker)
- [Project Instructions (Heroku with Docker)](#project-instructions-heroku-with-docker)
- [Initial Data Cleaning and Feature Engineering](#initial-data-cleaning-and-feature-engineering)
- [Legality](#legality)

## Code and Resources Used
---
- **Python Version** : 3.8.0 (Note that FBProphet library currently doesn't work for Python versions >=3.9.0)
- **Packages** : fbprophet, joblib, matplotlib, pymongo, numpy, optuna, pandas, plotly, scikit-learn, statsmodels, seaborn, streamlit, timeseries-cv
- **Dataset source** : From 360DIGITMG (For confidentiality reasons, dataset is not included here)
- **Database**: MongoDB Atlas
- **FBProphet documentation**: https://facebook.github.io/prophet/docs/quick_start.html
- **MongoDB documentation**: https://www.mongodb.com/docs/
- **Optuna documentation** : https://optuna.readthedocs.io/en/stable/
- **Scikit Learn documentation** : https://scikit-learn.org/stable/modules/classes.html
- **Numpy documentation**: https://numpy.org/doc/stable/
- **Pandas documentation**: https://pandas.pydata.org/docs/
- **Plotly documentation**: https://plotly.com/python/
- **Matplotlib documentation**: https://matplotlib.org/stable/index.html
- **Seaborn documentation**: https://seaborn.pydata.org/
- **Streamlit documentation**: https://docs.streamlit.io/
- **Statsmodels documentation**: https://www.statsmodels.org/stable/api.html

## Model Training Setting
---
For this project, day-forward cross validation is used for identifying the best model class to use for model deployment. Different training sets and validation sets are used with Optuna (TPE Multivariate Sampler with 20 trials) for hyperparameter tuning, while different test sets are used for model evaluation.

The diagram below shows how day-forward cross validation works:
<img src="https://user-images.githubusercontent.com/34255556/198823897-4effd7ba-f74a-4091-b988-3fcc363b2aec.png" width="800" height="350">

Given the dataset for this project is moderately large (less than 3000 samples), day-forward cross validation is the most suitable cross validation method to use for model algorithm selection to provide a more realistic generalization error of time series models.

The following list of time series models are tested in this project:
- SARIMAX (no seasonal component)
- Exponential Smoothing
- FBProphet

For model evaluation on time series models, the following metrics are used in this project:
- AIC (Main metric for Optuna hyperparameter tuning for SARIMAX and Exponential Smoothing)
- Mean Absolute Percentage Error (Main metric for Optuna hyperparameter tuning for FBProphet)

Note that AIC score is only used for hyperparameter tuning and not for direct comparison with other model classes, because calculation of AIC scores for different model classes (i.e. SARIMAX and Exponential Smoothing) require different initial conditions for computing likelihoods. Thus, MAPE is used instead for model selection.
More details about the use of AIC score can be referred here for reference: https://robjhyndman.com/hyndsight/aic/

## Project Findings
---

#### 1. EDA (Exploratory Data Analysis)

All plots generated from this section can be found in Intermediate_Train_Results/EDA folder.

##### i. Basic metadata of dataset
On initial inspection, the current dataset used in this project has a total of 71 features. Both "_id" and "ID.1 features represent unique identifier of a given record and the remaining features have  mix of "float", "int" and "object" data types. Upon closer inspection on data dictionary, there are several date-time related features where further information can be extracted and remaining features are considered as categorical variables.

##### ii. Target variable distribution
Given that there is no target variable, this project requires creating target variable manually (Wellbeing_Category_WMS - mainly based on variables related to Me and My Feelings Questionnaire. More details can be found in the coding file labeled "train preprocessing.py")

![Target_Class_Distribution](https://user-images.githubusercontent.com/34255556/196934993-fc9bbe23-81c3-459c-8263-2ff26a51b31f.png)

From the diagram above, there is a very clear indication of target imbalance between all 4 classes for multiclass classification. This indicates that target imbalancing needs to be addressed during model training.

##### iii. Missing values
![Proportion of null values](https://user-images.githubusercontent.com/34255556/196935092-1ba7c4e8-740f-49e7-bfc3-2247ee977b32.png)

From the diagram above, most features with missing values identified have missing proportions approximately less than 1%, except for "Method_of_keepintouch" feature with approximately 3% containing missing values.

---
#### 2. Best time series model

The following information below summarizes the configuration of the best model identified in this project:

  - <b>Best model class identified</b>: Exponential Smoothing

  - <b>Best model hyperparameters</b>: {'seasonal': 'additive', 'trend': None}
  
Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of inputs/number of jumps within day-forward cross validation (currently 8 folds in total).

For every type of time series model tested in this project, a folder is created for every model class within Intermediate_Train_Results folder with the following artifacts:

- HP_Importances for every fold (.png format)
- Hyperparameter tuning results for every fold (.csv format)
- Optimization history plot for every fold (.png format)
- Optuna study object for every fold (.pkl format)
- ACF Plot (.png format)
- PACF Plot (.png format)
- Residual Plot (.png format)

In addition, the following artifacts are also created for the best model class identified after final hyperparameter tuning on the entire dataset:

- HP_Importances (.png format)
- Hyperparameter tuning results (.csv format)
- Optimization history plot (.png format)
- Optuna study object (.pkl format)
- Learning curve plot (.png format)
- ACF Plot (.png format)
- PACF Plot (.png format)
- Residual Plot (.png format)
- Time Series Plot (Actual vs Predicted - .png format)

---
#### 3. Summary of model evaluation metrics from best time series model

The following information below summarizes the evaluation metric *(average (standard deviation)) from the best model identified in this project using day-forward cross validation: 

  - <b>MAPE score (Validation set)</b>: 0.11034% (1.29E-05)
  - <b>MAPE score (Test set)</b>: 0.1065588% (2.31E-05)

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of inputs/number of jumps within day-forward cross validation (currently 8 folds in total).

---
#### 4. Residual diagnostics from final model trained

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/198824370-d2caef08-2912-4eea-a0b9-dc9e0f84d8f0.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/198824376-1bb0317d-7241-47a1-bbb4-5fef8d981d93.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/198824390-8e87f3ca-7d26-4231-843e-9d0618401c09.png" width="400">
</p>

From the images above, both ACF and PACF plot shows there is no autocorrelation between residuals. The residual plot above shows that the mean amongst residuals is close to zero. Therefore, the assumption of no autocorrelation and zero mean amongst residuals remain valid for the best time series model identified.

---
#### 5. Time series plot (Actual vs Predicted) from best time series model

![Actual_vs_Predicted_Plot_ExponentialSmoothing](https://user-images.githubusercontent.com/34255556/198824441-4e0443a0-a64c-4dcf-ae4e-f4e9027fd6c8.png)

From the diagram above, exponential smoothing model shows a reasonable good fit on the actual power consumption levels, where the additive seasonality component is well captured.

---
#### 6. Learning Curve Analysis

![LearningCurve_ExponentialSmoothing](https://user-images.githubusercontent.com/34255556/198824593-e5264712-27b0-4115-aaf0-de776e26da1c.png)

From the diagram above, the gap between train and test MAPE scores gradually decreases as number of training sample size increases. Given the gap between both scores is considerably small, this indicates that adding more training data may not help to further improve model generalization.

## CRISP-DM Methodology
---
For any given Machine Learning projects, CRISP-DM (Cross Industry Standard Practice for Data Mining) methodology is the most commonly adapted methodology used.
The following diagram below represents a simple summary of the CRISP-DM methodology for this project:

<img src="https://www.datascience-pm.com/wp-content/uploads/2018/09/crisp-dm-wikicommons.jpg" width="450" height="400">

Note that an alternative version of this methodology, known as CRISP-ML(Q) (Cross Industry Standard Practice for Machine Learning and Quality Assurance) can also be used in this project. However, the model monitoring aspect is not used in this project, which can be considered for future use.

## Project Architecture Summary
---
The following diagram below summarizes the structure for this project:

![image](https://user-images.githubusercontent.com/34255556/197318105-4d4cd686-f6e5-43ed-8ad4-1cff1bbc2adf.png)

Note that all steps mentioned above have been logged accordingly for future reference and easy maintenance, which are stored in <b>Training_Logs</b> folder.

## Project Folder Structure
---
The following points below summarizes the use of every file/folder available for this project:
1. Application_Logger: Helper module for logging model training and prediction process
2. Intermediate_Train_Results: Stores results from EDA, data preprocessing and model training process
3. Model_Training_Modules: Helper modules for model training
4. Saved_Models: Stores best models identified from model training process for model prediction
5. Training_Batch_Files: Stores csv batch files to be used for model training
6. Training_Data_FromDB: Stores compiled data from MongoDB database for model training
7. Training_Logs: Stores logging information from model training for future debugging and maintenance
8. Dockerfile: Additional file for Docker project deployment
9. README.md: Details summary of project for presentation
10. requirements.txt: List of Python packages to install for project deployment
11. setup.py : Script for installing relevant python packages for project deployment
12. Docker_env: Folder that contains files that are required for model deployment without logging files or results.
13. pipeline_api.py: Main python file for running training pipeline process and performing model prediction.

## MongoDB Atlas Setup
---
![image](https://user-images.githubusercontent.com/34255556/197315546-b60b36b7-10e2-4b50-9eff-ae62ed44b17d.png)

For this project, data provided by the client in CSV format will be stored in MongoDB Atlas, which is a cloud database platform specially for MongoDB.

The following steps below shows the setup of MongoDB Atlas:

1. Register for a new MongoDB Atlas account for free using the following link: https://www.mongodb.com/cloud/atlas/register
2. After login, create a new database cluster (Shared option) and select the cloud provider and region of your choice:

<img src = "https://user-images.githubusercontent.com/34255556/197315198-8a65d44a-9e75-4d65-9de4-f3c10748b066.png" width="600">

3. Go to Database Access tab under Security section and add a new database user as follows:

<img src = "https://user-images.githubusercontent.com/34255556/197315308-c6c25139-528f-40f4-a3f1-55a6a269df68.png" width="600">

- Keep a record of username and password created for future use.

4. Go to Database tab under Deployment section and click on Connect button:

![image](https://user-images.githubusercontent.com/34255556/197315396-710bae00-c75d-4f69-b267-0ee9e217c819.png)

5. Select "Connect your application" option:

<img src = "https://user-images.githubusercontent.com/34255556/197315427-79eaf258-0ac7-4762-b3d5-4856d8474759.png" width="600">

6. <b>Important: Make a note of the connection string and replace username and password by its values from step 3.</b>

![image](https://user-images.githubusercontent.com/34255556/197315449-c077c899-97ed-4ce7-9d52-25c79bfaa217.png)

- Note that this connection string is required for connecting our API with MongoDB atlas.

The following sections below explains the three main approaches that can be used for deployment in this project after setting up MongoDB Atlas:
1. <b>Docker</b>
2. <b>Cloud Platform (Heroku with Docker)</b>
3. <b>Local environment</b>

## Project Instructions (Docker)
---

<img src="https://user-images.githubusercontent.com/34255556/195037066-21347c07-217e-4ecd-9fef-4e7f8cf3e098.png" width="600">

Deploying this project on Docker allows for portability between different environments and running instances without relying on host operating system.
  
<b>Note that docker image is created under Windows Operating system for this project, therefore these instructions will only work on other windows instances.</b>

<b> For deploying this project onto Docker, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

Docker Desktop needs to be installed into your local system (https://www.docker.com/products/docker-desktop/), before proceeding with the following steps:

1. Download and extract the zip file from this github repository into your local machine system.

<img src="https://user-images.githubusercontent.com/34255556/198825206-7e8e4483-2710-4862-b980-5bce29697b58.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/198824819-af31f11a-9985-4e7e-9e7a-6b4ff229cf72.png)

4. Build a new docker image on the project directory with the following command:
```
docker build -t api-name .
```

5. Run the docker image on the project directory with the following command: 
```
docker run -e PORT=8501 -p 8501:8501 api-name
```

6. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/198824842-82bad213-a460-43c3-9349-2c0ffca967fa.png" width="600">

Browser for the application can be opened from Docker Desktop by clicking on the specific button shown below:

![image](https://user-images.githubusercontent.com/34255556/197315936-2ea47b7a-9919-4010-b806-52f864966ea3.png)

7. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/198824850-374aac80-2693-47bc-8243-f73d81eb1f66.png" width="600">

8. After running all steps of the training pipeline, run the following command to extract files from a specific directory within the docker container to host machine for viewing:
```
docker cp <container-id>:<source-dir> <destination-dir>
```

9. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/198824862-f24eae53-c445-427f-bbf9-5dbc24a0cbf9.png" width="600">

10. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/198824873-3481021d-5a51-4516-9e43-8b71ab39325d.png" width="600">
<img src = "https://user-images.githubusercontent.com/34255556/198824878-a41e3275-f186-447f-b178-36b2192115ec.png" width="600">

## Project Instructions (Heroku with Docker)
---
<img src = "https://user-images.githubusercontent.com/34255556/195489080-3673ab77-833d-47f6-8151-0fed308b9eec.png" width="600">

A suitable alternative for deploying this project is to use docker images with cloud platforms like Heroku. 

<b> For deploying models onto Heroku platform, the following additional files are essential</b>:
- DockerFile
- requirements.txt
- setup.py

<b>Note that deploying this project onto other cloud platforms like GCP, AWS or Azure may have different additionnal files required.</b>

For replicating the steps required for running this project on your own Heroku account, the following steps are required:
1. Clone this github repository into your local machine system or your own Github account if available.

<img src="https://user-images.githubusercontent.com/34255556/198825206-7e8e4483-2710-4862-b980-5bce29697b58.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/198824819-af31f11a-9985-4e7e-9e7a-6b4ff229cf72.png)

5. From a new command prompt window, login to Heroku account and Container Registry by running the following commands:
```
heroku login
heroku container:login
```
Note that Docker needs to be installed on your local system before login to heroku's container registry.

6. Using the Dockerfile, push the docker image onto Heroku's container registry using the following command:
```
heroku container:push web -a app-name
```

7. Release the newly pushed docker images to deploy app using the following command:
```
heroku container:release web -a app-name
```

8. After successfully deploying docker image onto Heroku, open the app from the Heroku platform and you will see the following interface designed using Streamlit:

<img src = "https://user-images.githubusercontent.com/34255556/198824842-82bad213-a460-43c3-9349-2c0ffca967fa.png" width="600">

9. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/198824850-374aac80-2693-47bc-8243-f73d81eb1f66.png" width="600">

10. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/198824862-f24eae53-c445-427f-bbf9-5dbc24a0cbf9.png" width="600">

11. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/198824873-3481021d-5a51-4516-9e43-8b71ab39325d.png" width="600">
<img src = "https://user-images.githubusercontent.com/34255556/198824878-a41e3275-f186-447f-b178-36b2192115ec.png" width="600">

<b>Important Note</b>: 
- Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.

- Unlike stand-alone Docker containers, Heroku uses an ephemeral hard drive, meaning that files stored locally from running apps on Heroku will not persist when apps are restarted (once every 24 hours). Any files stored on disk will not be visible from one-off dynos such as a heroku run bash instance or a scheduler task because these commands use new dynos. Best practice for having persistent object storage is to leverage a cloud file storage service such as Amazon’s S3 (not part of project scope but can be considered)

## Project Instructions (Local Environment)
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/198825206-7e8e4483-2710-4862-b980-5bce29697b58.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. Add environment variable "MONGO_DB_URL" with connection string defined from last step of MongoDB Atlas setup section as value on your local system. The following link provides an excellent guide for setting up environment variables on your local system: https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/

4. Open anaconda prompt and create a new environment with the following syntax: 
```
conda create -n myenv python=3.8
```
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

5. After creating a new anaconda environment, activate the environment using the following command: 
```
conda activate myenv
```

6. Go to the local directory in Command Prompt where Docker_env folder is located and run the following commands in sequence to install all the python libraries : 
```
conda install libpython m2w64-toolchain -c msys2
pip install -r requirements.txt
pip install fbprophet
```

7. After installing all the required Python libraries, run the following command on your project directory: 
```
streamlit run pipeline_api.py
```

8. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/198824842-82bad213-a460-43c3-9349-2c0ffca967fa.png" width="600">

9. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/198824850-374aac80-2693-47bc-8243-f73d81eb1f66.png" width="600">

10. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/198824862-f24eae53-c445-427f-bbf9-5dbc24a0cbf9.png" width="600">

11. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/198824873-3481021d-5a51-4516-9e43-8b71ab39325d.png" width="600">
<img src = "https://user-images.githubusercontent.com/34255556/198824878-a41e3275-f186-447f-b178-36b2192115ec.png" width="600">

## Initial Data Cleaning and Feature Engineering
---
After performing Exploratory Data Analysis, the following steps are performed initially on the entire dataset before performing further data preprocessing and model training:

i) Filter data where respondent provides permission to use questionnaire (Use_questionnaire feature)

ii) Derive target variable based on features related to Me and My Feelings questionnaire.

iii) Reformat time related features (i.e Timestamp, Birthdate, Sleeptime_ytd and Awaketime_today) to appropriate form

iv) Removing list of irrelevant colummns identified from dataset (i.e. unique identifier features, features related to target variable to prevent target leakage and LSOA related features that have direct one to one relationship with WIMD related features)

v) Checking for duplicated rows and remove if exist

vi) Split dataset into features and target labels.

vii) Perform missing imputation on categorical variables based on highest frequency for every category.

viii) Save reduced set of features and target values into 2 different CSV files (X.csv and y.csv) for further data preprocessing with pipelines to reduce data leakage.

## Legality
---
This is an internship project made with 360DIGITMG for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
