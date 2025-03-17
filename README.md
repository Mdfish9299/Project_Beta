# Diabetes Early Detection Based on Lifestyle and Diet

![Diabetes](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/584/244/datas/original.png)

## Overview

The core of this project lies in a Jupyter Notebook, [Project.ipynb](Project.ipynb). This notebook utilizes several models, including the Random Forest Classifier (RFC) to make accurate predictions about diseases based on a set of symptoms.

## Table of Contents

1. [Introduction](#introduction)
2. [Exploratory Data Analysis (EDA)](#eda)
   - [1. Import Dependencies](#dependencies)
   - [2. Load the Dataset](#load-dataset)
   - [3. Statistical Details](#statistical-details)
   - [4. Convert Categorical Data](#convert-categorical)
   - [5. Handle Missing Values](#handle-missing)
   - [6. Symptoms Severity](#symptoms-severity)
   - [7. Split Dataset](#split-dataset)
3. [Model Selection](#model-selection)
   - [1. Random Forest Classifier](#random-forest)
   - [2. Neural Network (MLPClassifier)](#neural-network)
4. [Evaluate Models](#evaluate-models)
   - [1. Random Forest Classifier Evaluation](#evaluate-rfc)
   - [2. Neural Network (MLPClassifier) Evaluation](#evaluate-mlpc)
5. [Save Model](#save-model)
6. [Test the Model Manually](#test-manually)
7. [Symptom Input](#symptom-input)
8. [Usage with App.py](#usage-with-app)
9. [Implementation Image/Video](#implementation-image-video)

## Introduction <a name="introduction"></a>

Discover how machine learning can revolutionize Diabetes prediction. The RFC model achieves an outstanding 99% accuracy, making it a reliable tool for healthcare professionals and enthusiasts alike.

## Exploratory Data Analysis (EDA) <a name="eda"></a>

Explore the step-by-step process of preparing the data, training the model, and evaluating its performance.

## Dataset Information
The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has diabetes, is pre-diabetic, or healthy.

### Dataset Characteristics
- **Type:** Tabular, Multivariate
- **Subject Area:** Health and Medicine
- **Associated Tasks:** Classification
- **Feature Type:** Categorical, Integer
- **# Instances:** 253680
- **# Features:** 21

### Purpose
- **Created for:** To better understand the relationship between lifestyle and diabetes in the US
- **Funded by:** The CDC

### Instances
- **Representation:** Each row represents a person participating in this study.
- **Recommended Data Splits:** Cross-validation or a fixed train-test split could be used.
- **Sensitive Data:** Gender, Income, Education level
- **Data Preprocessing:** Bucketing of age

### Additional Information
- **Dataset link:** [CDC Diabetes Health Indicators](https://www.cdc.gov/brfss/annual_data/annual_2014.html)
- **Missing Values:** No

## Variables Table
| Variable Name          | Role     | Type     | Demographics |               Description                                    | Missing Values |
|------------------------|----------|----------|--------------|--------------------------------------------------------------|----------------|
| ID                     | ID       | Integer  |              | Patient ID                                                                  |       no       |
| Diabetes_binary        | Target   | Binary   |              | 0 = no diabetes, 1 = prediabetes or diabetes                                |       no       |
| HighBP                 | Feature  | Binary   |              | 0 = no high BP, 1 = high BP                                                 |       no       |
| HighChol               | Feature  | Binary   |              | 0 = no high cholesterol, 1 = high cholesterol                               |       no       |
| CholCheck              | Feature  | Binary   |              | 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years   |       no       |
| BMI                    | Feature  | Integer  |              | Body Mass Index                                                             |       no       |
| Smoker                 | Feature  | Binary   |              | Have you smoked at least 100 cigarettes in your entire life? 0 = no, 1 = yes|       no       |
| Stroke                 | Feature  | Binary   |              | (Ever told) you had a stroke. 0 = no, 1 = yes                               |       no       |
| HeartDiseaseorAttack   | Feature  | Binary   |              | coronary heart disease (CHD) or myocardial infarction (MI) 0 = no, 1 = yes  |       no       |
| PhysActivity           | Feature  | Binary   |              | physical activity in past 30 days - not including job 0 = no, 1 = yes       |       no       |
| Fruits                 | Feature  | Binary   |              | Consume Fruit 1 or more times per day 0 = no 1 = yes                        |       no       |
| Veggies                | Feature  | Binary   |              | Consume Vegetables 1 or more times per day 0 = no 1 = yes                   |       no       |
| HvyAlcoholConsump      | Feature  | Binary   |              | Heavy drinkers (adult men having more than 14 drinks/week and adult women having more than 7 drinks per week) 0 = no 1 = yes       |      no      |
| AnyHealthcare          | Feature  | Binary   |              | Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes                  |      no      |
| NoDocbcCost            | Feature  | Binary   |              | Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes               |      no      |
| GenHlth                | Feature  | Integer  |              | Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor                     |      no      |
| MentHlth               | Feature  | Integer  |              | Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days      | no        |
| PhysHlth               | Feature  | Integer  |              | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days      |    no             |
| DiffWalk               | Feature  | Binary   |              | Do you have serious difficulty walking or climbing stairs? 0 = no, 1 = yes                                                          |      no      |
| Sex                    | Feature  | Binary   | Sex          | 0 = female, 1 = male                                                                                                           |      no      |
| Age                    | Feature  | Integer  | Age          | 13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older (Every 5 years increases 1)                             |      no      |
| Education              | Feature  | Integer  | Education Level    | Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate) |       no             |
| Income              | Feature | Integer | Income             | Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more                     |       no             |



### Additional Variable Information
- **Diabetes diagnosis**
- **Demographics (race, sex)**
- **Personal information (income, education)**
- **Health history (drinking, smoking, mental health, physical health)**

### Class Labels
- **Diabetes**
- **Pre-diabetes**
- **Healthy**

## Installation
In a Jupyter notebook, install with the command 

    !pip3 install -U ucimlrepo 
    
Restart the kernel and import the module `ucimlrepo`.

## Example Usage

    from ucimlrepo import fetch_ucirepo, list_available_datasets
	
	# check which datasets can be imported
	list_available_datasets()
    
    # import dataset
    heart_disease = fetch_ucirepo(id=45)
    # alternatively: fetch_ucirepo(name='Heart Disease')
    
    # access data
    X = heart_disease.data.features
    y = heart_disease.data.targets
    # train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)
    
    # access metadata
    print(heart_disease.metadata.uci_id)
    print(heart_disease.metadata.num_instances)
    print(heart_disease.metadata.additional_info.summary)
    
    # access variable info in tabular format
    print(heart_disease.variables)



## `fetch_ucirepo`
Loads a dataset from the UCI ML Repository, including the dataframes and metadata information.

### Parameters
Provide either a dataset ID or name as keyword (named) arguments. Cannot accept both.
- **`id`**: Dataset ID for UCI ML Repository
- **`name`**: Dataset name, or substring of name

### Returns
- **`dataset`**
	- **`data`**: Contains dataset matrices as **pandas** dataframes
		- `ids`: Dataframe of ID columns
		- `features`: Dataframe of feature columns
		- `targets`: Dataframe of target columns
		- `original`: Dataframe consisting of all IDs, features, and targets
		- `headers`: List of all variable names/headers
	- **`metadata`**: Contains metadata information about the dataset
		- See Metadata section below for details
	- **`variables`**: Contains variable details presented in a tabular/dataframe format
		- `name`: Variable name
		- `role`: Whether the variable is an ID, feature, or target
		- `type`: Data type e.g. categorical, integer, continuous
		- `demographic`: Indicates whether the variable represents demographic data
		- `description`: Short description of variable
		- `units`: variable units for non-categorical data
		- `missing_values`: Whether there are missing values in the variable's column
   

## `list_available_datasets`
Prints a list of datasets that can be imported via `fetch_ucirepo`
### Parameters
- **`filter`**: Optional keyword argument to filter available datasets based on a category
	- Valid filters: `aim-ahead`
- **`search`**: Optional keyword argument to search datasets whose name contains the search query
### Returns
none


## Metadata 
- `uci_id`: Unique dataset identifier for UCI repository 
- `name`
- `abstract`: Short description of dataset
- `area`: Subject area e.g. life science, business
- `task`: Associated machine learning tasks e.g. classification, regression
- `characteristics`: Dataset types e.g. multivariate, sequential
- `num_instances`: Number of rows or samples
- `num_features`: Number of feature columns
- `feature_types`: Data types of features
- `target_col`: Name of target column(s)
- `index_col`: Name of index column(s)
- `has_missing_values`: Whether the dataset contains missing values
- `missing_values_symbol`: Indicates what symbol represents the missing entries (if the dataset has missing values)
- `year_of_dataset_creation`
- `dataset_doi`: DOI registered for dataset that links to UCI repo dataset page
- `creators`: List of dataset creator names
- `intro_paper`: Information about dataset's published introductory paper
- `repository_url`: Link to dataset webpage on the UCI repository
- `data_url`: Link to raw data file
- `additional_info`: Descriptive free text about dataset
	- `summary`: General summary 
	- `purpose`: For what purpose was the dataset created?
	- `funding`: Who funded the creation of the dataset?
	- `instances_represent`: What do the instances in this dataset represent?
	- `recommended_data_splits`: Are there recommended data splits?
	- `sensitive_data`: Does the dataset contain data that might be considered sensitive in any way?
	- `preprocessing_description`: Was there any data preprocessing performed?
	- `variable_info`: Additional free text description for variables
	- `citation`: Citation Requests/Acknowledgements
 - `external_url`: URL to external dataset page. This field will only exist for linked datasets i.e. not hosted by UCI


## Links
- [UCI Machine Learning Repository home page](https://archive.ics.uci.edu/)
- [PyPi repository for this package](https://pypi.org/project/ucimlrepo)
- [Submit an issue](https://github.com/uci-ml-repo/ucimlrepo-feedback/issues)




### 1. Import Dependencies <a name="dependencies"></a>

Import the necessary libraries and modules for data analysis and model implementation.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/80e5ff54-e301-45cc-bc92-da1809e02cf8)

### 2. Load the Dataset <a name="load-dataset"></a>

Load the dataset from a CSV file containing information about various diseases and their associated symptoms.

- [Dataset.csv](Dataset/dataset.csv) <-- (Take a look on data, used to build this model)

<div style="display: flex;">
<img src="images/original dataset.png" alt="Image 1";">
</div>

### 3. Statistical Details <a name="statistical-details"></a>

Understand the dataset better with statistical details, including mean, standard deviation, and quartiles.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/aa415040-b53a-4ccf-b23d-9878927b0123)

### 4. Handle Missing Values <a name="handle-missing"></a>

Handle missing values in the dataset by replacing them with zeros.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/74807e40-e02b-41e8-9efb-25cd2802fa0b)

### 5. Symptoms Severity <a name="symptoms-severity"></a>

Encode symptoms severity using a separate dataset for training the model. This Dataset assign the respective weight to each Symptoms.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/98f06197-79d5-468c-a7f3-3c1ba836a5fc)

With the help of this data, we will encode our original dataset.

### 6. Convert Categorical Data <a name="convert-categorical"></a>

Convert categorical data into numerical format using Severity dataset, ensuring consistency.
- Dataset after converting Categorical Data into Numerical Data.
- Now, its Ready to train the models.

<div style="display: flex;">
<img src="images/Encoded dataset.png" alt="Image 1";">
</div>

### 7. Split Dataset <a name="split-dataset"></a>
Firstly, Splits the Clean Dataset into Training data and Label data.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/b4de6af9-5fe5-41ef-9a79-b6e84063067b)

Then, it time to split the dataset into training and testing sets for model training and evaluation, using module `train_test_spilt`.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/1a502900-309b-42b2-baf3-a8d593f1a739)

## Model Selection <a name="model-selection"></a>

### 1. Random Forest Classifier (RFC) <a name="random-forest"></a>

Train and evaluate the RFC model, achieving an impressive accuracy of 99%.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/ec1b10c0-ab6d-4040-a4ed-0c62045bc3af)


### 2. Neural Network (MLPClassifier) <a name="neural-network"></a>

Implement and evaluate a Multi-Layer Perceptron (MLP) Classifier with a similar accuracy of 99%.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/d72d50f3-c3bc-466a-8642-9b9c17fe8fab)

## Evaluate Models <a name="evaluate-models"></a>

Detailed classification reports are provided for both the Random Forest Classifier and the MLPClassifier.

### 1. Random Forest Classifier Evaluation <a name="evaluate-rfc"></a>

Precision, recall, and accuracy scores showcase the effectiveness of the Random Forest Classifier.
 - Accuracy Proof:
<div style="display: flex;">
<img src="images/RFC accuracy.png" alt="Image 1";">
</div>

### 2. Neural Network (MLPClassifier) Evaluation <a name="evaluate-mlpc"></a>

Metrics calculated for the MLPClassifier demonstrate its comparable performance.
- Accuracy Proof:
<div style="display: flex;">
<img src="images/MLP classifier accuracy.png" alt="Image 1";">
</div>

## Save Model <a name="save-model"></a>

- Save the trained Random Forest Classifier model using the joblib library for future use.
- Used Python library `joblib`.

![image](https://github.com/amMistic/Diseases-Prediction-based-on-Symptoms/assets/134824444/49031765-7de6-4f69-a5ba-c066f0d9a87d)

## Test the Model Manually <a name="test-manually"></a>

Manually test the model with a sample set of symptoms, demonstrating its predictive capability.

<div style="display: flex;">
<img src="images/test_ manual.png" alt="Image 1";">
</div>

## Symptom Input <a name="symptom-input"></a>

- Capture user input for specific symptoms to predict the associated disease.

<div style="display: flex;">
  <img src="images/Prediction before.png" alt="Image 1" width=80%, height=80% ;">
</div>

## Usage with App.py <a name="usage-with-app"></a>

Utilize the [app.py](app.py) file to allow outsiders to use the model on their local machines. This file enables seamless interaction with the model, providing a user-friendly interface.

Prerequisites: 
- Load [dataset.csv](Dataset/dataset.csv)
- Load  [Severity-Symptom.csv](Dataset/Symptom-severity.csv) (Contain the weights of each Symptoms)
- Load model `model_RFC.joblib` (Replace 'model_RFC.joblib' with your model name)

Or You just Copy/Fork Repository To Have the file [app.py](app.py).

## Implementation Image/Video <a name="implementation-image-video"></a>
 Preview of app.py using Streamlit Library:

<div style="display: flex;">
  <img src="images/After prediction.png" alt="Image 1" width=80%, height=80% ;">
</div>

Explore the implementation of the model through an image provided in the project directory. Witness firsthand the capabilities of disease prediction using symptoms.

---
# Key Project Code

### Data Transformation

After loading the dataset, we transformed the data:

```python
cols_to_convert = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

for col in cols_to_convert:
    X[col] = X[col].astype(bool)
```

### Statistical Data Overview

```python
X.describe()
```

### Converted and Cleaned Data

We saved the cleaned dataset to CSV for further use:

```python
X.to_csv('X.csv', index=False)
pd.DataFrame(y, columns=['target']).to_csv('y.csv', index=False)
df_combined = pd.concat([X, pd.DataFrame(y, columns=['target'])], axis=1)
df_combined.to_csv('Cleaned_Data.csv', index=False)
```

### Splitting the Data for Training and Testing

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

### Training and Evaluating KNN Model

```python
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train_scores = []
test_scores = []

for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")

plt.plot(range(1, 20, 2), train_scores, marker='o')
plt.plot(range(1, 20, 2), test_scores, marker='x')
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.legend(["train", "test"])
plt.show()
```

### Model Performance Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

y_knn = knn.predict(X_test)
roc_auc_knn = roc_auc_score(y_test, y_knn)

print("KNN Performance Metrics")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_knn))
print("Classification Report:\n", classification_report(y_test, y_knn))
print('ROC AUC:', round(roc_auc_knn*100,2), "%")
```

### Testing the Model Manually

```python
input_data = [[1, 1, 1, 25, 0, 0, 1, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 9, 6, 2]]
manual_pred_knn = knn.predict(input_data)
print('KNN Prediction:', manual_pred_knn)
```

## Project Strategy

1. **Initial Commit** - Elliot's first commit started the project `(9e05e0f)`.
2. **Framework Setup** - Mehdi's commit `(0073d14)` introduced the Streamlit app (`streamlit run app.py`).
3. **Commit Issues** - James encountered push errors in commit `(862e628)`, later identified in `(0f475f8)`, causing commit `(1c2d1f0)` to be voided.
4. **Issue Resolution** - Professor Pet resolved the issue, allowing further development `(e6bdb50)`.
5. **New Strategy** - To avoid conflicts, team members coordinated Git pushes separately.
6. **Code Refinement** - Between `(57be890)` and `(25e0324)`, trial-and-error methods refined the model structure.
7. **Final Testing** - A small data anomaly was detected before the final version (Stopped adding commits notes due to Pete's feedback what are YOUR thoughts)

---

### How to Run the Streamlit App

1. Save the script as `app.py`.
2. Ensure all model files are in the same directory.
3. Run the command:
   ```bash
   streamlit run app.py
   ```

