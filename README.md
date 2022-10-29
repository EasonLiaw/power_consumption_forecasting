# Power Consumption Forecasting Project

## Background
---

<img src="https://user-images.githubusercontent.com/34255556/196916971-21406264-79e5-4f70-a418-08c94e4c1b91.png" width="600">

Ever since the start of the pandemic back in 2020, the government expects more lockdowns through the winter to slow down the spread of the virus. One of the concerns raised by the government is the wellbeing of young primary school children since they are removed from their friends and play environments.

For this project, the main goal is to deploy initial classification models that help to monitor a child’s wellbeing status during the pandemic period. By identifying factors that influence the status of a child’s wellbeing through a customized survey, the government will be able to make more informed decisions/new policies that reduce the risk of a child having behavioral or emotional difficulties or both.

Dataset is provided in .json format by client under Training_Batch_Files folder for model training. (not included in this repository due to data confidentiality reasons)

For model prediction, a web API is used (created using StreamLit) for user input. Note that results generated from model prediction along with user inputs can be stored in various formats (i.e. in CSV file format or another database).

## Contents
- [Code and Resources Used](#code-and-resources-used)
- [Model Training Setting](#model-training-setting)
- [Project Findings](#project-findings)
  - [EDA](#1-eda-exploratory-data-analysis)
  - [Best time series model](#2-best-time-series-model)
  - [Summary of model evaluation metrics from best classification model](#3-summary-of-model-evaluation-metrics-from-best-classification-model)
  - [Hyperparameter importances from Optuna (Final model)](#4-hyperparameter-importances-from-optuna-final-model)
  - [Hyperparameter tuning optimization history from Optuna](#5-hyperparameter-tuning-optimization-history-from-optuna)
  - [Overall confusion matrix and classification report from final model trained](#6-overall-confusion-matrix-and-classification-report-from-final-model-trained)
  - [Precision Recall Curve from best classification model](#7-precision-recall-curve-from-best-classification-model)
  - [Learning Curve Analysis](#8-learning-curve-analysis)
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
- **Python Version** : 3.8.0
- **Packages** : borutashap, feature-engine, featurewiz, imbalanced-learn, joblib, catboost, lightgbm, matplotlib, pymongo, numpy, optuna, pandas, plotly, scikit-learn, scipy, seaborn, shap, streamlit, tqdm, xgboost, yellowbrick
- **Dataset source** : From 360DIGITMG (For confidentiality reasons, dataset is not included here)
- **Database**: MongoDB Atlas
- **MongoDB documentation**: https://www.mongodb.com/docs/
- **Optuna documentation** : https://optuna.readthedocs.io/en/stable/
- **Feature Engine documentation** : https://feature-engine.readthedocs.io/en/latest/
- **Scikit Learn documentation** : https://scikit-learn.org/stable/modules/classes.html
- **Numpy documentation**: https://numpy.org/doc/stable/
- **Pandas documentation**: https://pandas.pydata.org/docs/
- **Plotly documentation**: https://plotly.com/python/
- **Matplotlib documentation**: https://matplotlib.org/stable/index.html
- **Seaborn documentation**: https://seaborn.pydata.org/
- **Streamlit documentation**: https://docs.streamlit.io/

## Model Training Setting
---
For this project, day-forward cross validation is used for identifying the best model class to use for model deployment. Different training sets and validation sets are used with Optuna (TPE Multivariate Sampler with 20 trials) for hyperparameter tuning, while different test sets are used for model evaluation.

The diagram below shows how day-forward cross validation works:
<img src="https://mlr.mlr-org.com/articles/pdf/img/nested_resampling.png" width="600" height="350">

Given the dataset for this project is moderately large (less than 2500 samples), day-forward cross validation is the most suitable cross validation method to use for model algorithm selection to provide a more realistic generalization error of time series models.

The following list of time series models are tested in this project:
- SARIMAX (no seasonal component)
- Exponential Smoothing
- FBProphet

For model evaluation on time series models, the following metrics are used in this project:
- AIC (Main metric for Optuna hyperparameter tuning for SARIMAX and Exponential Smoothing)
- Mean Absolute Percentage Error (Main metric for Optuna hyperparameter tuning for FBProphet)

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

  - <b>Best model hyperparameters</b>: {'seasonal': 'additive}
  
Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of inputs within day-forward cross validation.

For every type of time series model tested in this project, a folder is created for every model class within Intermediate_Train_Results folder with the following artifacts:

- Confusion Matrix from 5 fold cross validation (.png format)
- Classification Report from 5 fold cross validation (.png format)
- HP_Importances for every fold (.png format - 5 in total)
- Hyperparameter tuning results for every fold (.csv format - 5 in total)
- Optimization history plot for every fold (.png format - 5 in total)
- Optuna study object for every fold (.pkl format - 5 in total)
- Precision-Recall curve (.png format)

In addition, the following artifacts are also created for the best model class identified after final hyperparameter tuning on the entire dataset:

- Confusion matrix (.png format)
- Classification report (.png format)
- HP_Importances (.png format)
- Hyperparameter tuning results (.csv format)
- Optimization history plot (.png format)
- Optuna study object (.pkl format)
- Learning curve plot (.png format)
- Shap plots for feature importance from every class (.png format - 2 in total)
- Precision recall curve (.png format)

---
#### 3. Summary of model evaluation metrics from best classification model

The following information below summarizes the evaluation metrics *(average (standard deviation)) from the best model identified in this project along with the confusion matrix from nested cross validation (5 outer fold with 3 inner fold): 

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196926115-2c43b974-4a55-4624-9e17-8db399b9510c.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/196926153-0b2b1d2e-7e09-40f0-9db6-360c87085d1a.png" width="400">
</p>

  - <b>Balanced accuracy (Training set - 3 fold)</b>: 0.4975 (0.0381)
  - <b>Balanced accuracy (Validation set - 3 fold)</b>: 0.3367 (0.0267)
  - <b>Balanced accuracy (Test set - 5 fold)</b>: 0.3484 (0.0284)

  - <b>Precision (Training set - 3 fold)</b>: 0.5574 (0.1139)
  - <b>Precision (Validation set - 3 fold)</b>: 0.3854 (0.0504)
  - <b>Precision (Test set - 5 fold)</b>: 0.3346 (0.0384)

Note that the results above may differ by changing search space of hyperparameter tuning or increasing number of trials used in hyperparameter tuning or changing number of folds within nested cross validation

---
#### 4. Hyperparameter importances from Optuna (Final model)

![HP_Importances_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/196925529-e25eac89-ea69-4374-9d54-9951e331c90c.png)

From the image above, determining the contrast method for encoding ordinal data and method for handling imbalanced data as part of preprocessing pipeline for Linear SVC model provides the highest influence (0.22), followed by selecting hyperparameter value of "C", "class_weight" and feature selection method. Setting hyperparameter value of penalty and use of clustering as additional feature for Linear SVC model provides little to zero influence on results of hyperparameter tuning. This may suggest that both penalty hyperparameters of Linear SVC model and use of clustering as additional feature can be excluded from hyperparameter tuning in the future during model retraining to reduce complexity of hyperparameter tuning process.

---
#### 5. Hyperparameter tuning optimization history from Optuna

![Optimization_History_LinearSVC_Fold_overall](https://user-images.githubusercontent.com/34255556/196925946-56216317-c37c-4cb5-ad0b-632145be6386.png)

From the image above, the best objective value (average of F1 macro scores from 3 fold cross validation) is identified after 20 trials.

---
#### 6. Overall confusion matrix and classification report from final model trained

<p float="left">
<img src="https://user-images.githubusercontent.com/34255556/196926313-f89b556b-2cd6-4b95-9095-7c3f2733c7e7.png" width="400">
<img src="https://user-images.githubusercontent.com/34255556/196926276-512f430d-5aaa-4916-96af-c71154cdcc2a.png" width="400">
</p>

From the image above, the classification model performs better for cases where a child's wellbeing is either normal or emotional and behaviour significant with more samples being classified correctly. Given that the model evaluation criteria emphasize the costly impact of having both false positives and false negatives equally for all classes, the current classification model is optimized to improve F1 macro score.

---
#### 7. Precision Recall Curve from best classification model

![PrecisionRecall_Curve_LinearSVC_CV](https://user-images.githubusercontent.com/34255556/196927600-68a0119c-c961-4ad1-9cd0-3efa9d7c1258.png)

From the diagram above, precision-recall curve from best model class identified shows that the model performs best on identify wellbeing status of children that are normal (0.89), followed by emotional_significant (0.10), behaviour_significant (0.09) and emotional_and_behaviour_significant (0.06). 

---
#### 8. Learning Curve Analysis

![LearningCurve_LinearSVC](https://user-images.githubusercontent.com/34255556/196927116-575713c0-699f-4f23-bfd8-248341a22c48.png)

From the diagram above, the gap between train and test F1 macro scores (from 5-fold cross validation) gradually decreases as number of training sample size increases.
However, the gap between both scores remain large, which indicates that adding more training data may help to improve generalization of model.

---

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
6. Training_Data_FromDB: Stores compiled data from SQL database for model training
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

For this project, data provided by the client in JSON format will be stored in MongoDB Atlas, which is a cloud database platform specially for MongoDB.

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

<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/197315793-d676cd57-b2e3-4702-9c83-1fcd84efe6d8.png)

4. Build a new docker image on the project directory with the following command:
```
docker build -t api-name .
```

5. Run the docker image on the project directory with the following command: 
```
docker run -e PORT=8501 -p 8501:8501 api-name
```

6. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

Browser for the application can be opened from Docker Desktop by clicking on the specific button shown below:

![image](https://user-images.githubusercontent.com/34255556/197315936-2ea47b7a-9919-4010-b806-52f864966ea3.png)

7. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

8. After running all steps of the training pipeline, run the following command to extract files from a specific directory within the docker container to host machine for viewing:
```
docker cp <container-id>:<source-dir> <destination-dir>
```

9. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

10. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

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

<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.

3. Go to your own Heroku account and create a new app with your own customized name.
<img src="https://user-images.githubusercontent.com/34255556/160223589-301262f6-6225-4962-a92f-fc7ca8a0eee9.png" width="600" height="400">

4. On line 8 inside Dockerfile, set the environment variable MONGO_DB_URL as the connection string defined in the last step of MongoDB Atlas Setup section.

![image](https://user-images.githubusercontent.com/34255556/197315793-d676cd57-b2e3-4702-9c83-1fcd84efe6d8.png)

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
<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

9. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

10. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

11. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

<b>Important Note</b>: 
- Using "free" dynos on Heroku app only allows the app to run for a maximum of 30 minutes. Since the model training and prediction process takes a long time, consider changing the dynos type to "hobby" for unlimited time, which cost about $7 per month per dyno. You may also consider changing the dynos type to Standard 1X/2X for enhanced app performance.

- Unlike stand-alone Docker containers, Heroku uses an ephemeral hard drive, meaning that files stored locally from running apps on Heroku will not persist when apps are restarted (once every 24 hours). Any files stored on disk will not be visible from one-off dynos such as a heroku run bash instance or a scheduler task because these commands use new dynos. Best practice for having persistent object storage is to leverage a cloud file storage service such as Amazon’s S3 (not part of project scope but can be considered)

## Project Instructions (Local Environment)
---  
If you prefer to deploy this project on your local machine system, the steps for deploying this project has been simplified down to the following:

1. Download and extract the zip file from this github repository into your local machine system.
<img src="https://user-images.githubusercontent.com/34255556/197315695-5f19b123-22a3-4751-82cd-d6bbca13a3d9.png" width="600" height="200">

2. Copy Docker_env folder into a separate directory, before proceeding with subsequent steps which will use Docker_env folder as root directory.
  
3. Add environment variable "MONGO_DB_URL" with connection string defined from last step of MongoDB Atlas setup section as value on your local system. The following link provides an excellent guide for setting up environment variables on your local system: https://chlee.co/how-to-setup-environment-variables-for-windows-mac-and-linux/

4. Open anaconda prompt and create a new environment with the following syntax: 
```
conda create -n myenv python=3.10
```
- Note that you will need to install anaconda if not available in your local system: https://www.anaconda.com/

5. After creating a new anaconda environment, activate the environment using the following command: 
```
conda activate myenv
```

6. Go to the local directory in Command Prompt where Docker_env folder is located and run the following command to install all the python libraries : 
```
pip install -r requirements.txt
```

7. Overwrite both BorutaShap.py and _tree.py scripts in relevant directories (<b>env/env-name/lib/site-packages and env/env-name/lib/site-packages/shap/explainers</b>) where the original files are located.

8. After installing all the required Python libraries, run the following command on your project directory: 
```
streamlit run pipeline_api.py
```

9. A new browser will open after successfully running the streamlit app with the following interface:

<img src = "https://user-images.githubusercontent.com/34255556/197315976-fa90cc7a-a0b3-4c82-9c38-62072db71399.png" width="600">

10. From the image above, click on Training Data Validation first for initializing data ingestion into MongoDB Atlas, followed by subsequent steps from top to bottom in order to avoid potential errors with the model training/model prediction process. The image below shows an example of notification after the process is completed for Training Data Validation process:

<img src = "https://user-images.githubusercontent.com/34255556/197316040-748289f6-f509-4e29-aac0-1765de6d3167.png" width="600">

11. After performing model training, clicking on the Model Prediction section expands the following section that allows user input for model prediction:

<img src = "https://user-images.githubusercontent.com/34255556/197316098-ec71b7df-6819-4c46-944b-27596c6b262b.png" width="600">

12. The image below shows an example of output from model prediction after successfully completed all of the above steps:

<img src = "https://user-images.githubusercontent.com/34255556/197316193-d1cf6fb7-91be-4283-91d6-cced35c70e41.png" width="600">

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

For more details of which features have been initially removed from the dataset, refer to the following CSV file: <b>Columns_Drop_from_Original.csv</b>

In addition, the following pickle files (with self-explanatory names) have been created inside Intermediate_Train_Results folder during this stage which may be used later on during data preprocessing on test data:
- <b>CategoryImputer.pkl</b>

## Legality
---
This is an internship project made with 360DIGITMG for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
