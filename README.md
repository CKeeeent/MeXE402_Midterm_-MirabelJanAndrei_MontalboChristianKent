# MeXE402_Midterm_-MirabelJanAndrei_MontalboChristianKent-

# **Linear and Logistic Regression Analysis**

## Introduction 
<div align="justify"> This project demonstrates a step by step analyses of Student achievement in secondary education of two Portuguese schools and Heart disease dataset applied with the help of two most widely used machine learning algorithms, Linear Regression and Logistic Regression. Both are considered as common algorithms in predictive modeling. </div>

- **<div align="justify">Linear Regression** is a supervised learning algorithm used to predict a continuous dependent variable based on one or more independent variables. It involves prediction on the continuous dependent variable, using one or more independent variables. It takes into consideration the interdependence that exists between the dependent and independent variables and strives to establish the closest line (regression line) which minimizes the deviation of actual values from expected or predicted values Generally.</div>

- **<div align="justify">Logistic Regression**, on the other hand, is used to analyze categorical dependent variables. It determines the probability of a generally defined point in space belonging to some class. However, the model has a logistic function that determines the desired effects of a binary dependent variable, hence it is predominantly used in classification problems.</div>

## Dataset Description
The project utilizes two datasets:
1. **Student Performance Data Set**:
   - This dataset contains information on student achievement in secondary education from two Portuguese schools.
   - <div align="justify">Features include demographic, social, and academic attributes (e.g., age, study time, school support, and previous grades), with the aim to predict final exam scores.</div>
   - This dataset is used for **Linear Regression** to predict continuous outcomes like final grades.

2. **Heart Disease Dataset**:
   - <div align="justify">A public health dataset used to predict the presence of heart disease based on factors such as age, cholesterol levels, blood pressure, and more.</div>
   - The dataset contains various medical attributes, including both categorical and numerical variables.
   - This dataset is used for **Logistic Regression** to classify whether or not a person is likely to have heart disease.

## Project Objectives
The objective of this project is to:
1. <div align="justify">To create a linear regression model that would predict students' continuous performance indicators based on the features given. More specifically formulate the model in order to analyze the outcome using Mean Squared Error (MSE) and R² score.</div>
2. <div align="justify">Utilize the logistic regression analysis to be used in the data classification process based on the existence or absence of some feature, such as heart disease. </div>
3. <div align="justify">Evaluate model performance based on various factors using accuracy and correlation.</div>
4. <div align="justify">Analyze the performance of the models using scatter plots and graphs that can be used to visualize how well the models work.</div>


# Linear Regression Analysis: *Student achievement in secondary education of two Portuguese schools.*
## 1. Importing the required libraries, modules, and functions.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```
## 2. Loading and Inspecting the Dataset
- This step provides a preview of the data, helping identify the columns and understand the general structure.
```python
# Load the dataset
sdata = pd.read_csv('student-por.csv')
sdata.head(10)
```
> *Displaying 10 rows of the datasheet.*
>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school</th>
      <th>sex</th>
      <th>age</th>
      <th>address</th>
      <th>famsize</th>
      <th>Pstatus</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>...</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>G1</th>
      <th>G2</th>
      <th>G3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GP</td>
      <td>F</td>
      <td>18</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>1</td>
      <td>1</td>
      <td>at_home</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>12</td>
      <td>13</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GP</td>
      <td>F</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>4</td>
      <td>2</td>
      <td>health</td>
      <td>services</td>
      <td>...</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GP</td>
      <td>F</td>
      <td>16</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>3</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>11</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>4</td>
      <td>3</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>12</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GP</td>
      <td>M</td>
      <td>16</td>
      <td>U</td>
      <td>LE3</td>
      <td>T</td>
      <td>2</td>
      <td>2</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>13</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GP</td>
      <td>F</td>
      <td>17</td>
      <td>U</td>
      <td>GT3</td>
      <td>A</td>
      <td>4</td>
      <td>4</td>
      <td>other</td>
      <td>teacher</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>LE3</td>
      <td>A</td>
      <td>3</td>
      <td>2</td>
      <td>services</td>
      <td>other</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GP</td>
      <td>M</td>
      <td>15</td>
      <td>U</td>
      <td>GT3</td>
      <td>T</td>
      <td>3</td>
      <td>4</td>
      <td>other</td>
      <td>other</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>12</td>
      <td>12</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>

> *The dependent variable is the column "G3".*
## 3. Data Preprocessing
### 3.1 Handle Missing Values
- Check for missing values in each column.
```python
# Check for missing values
print(sdata.isnull().sum())
```
| Feature      | Missing Values |
|:------------:|:--------------:|
| school       | 0              |
| sex          | 0              |
| age          | 0              |
| address      | 0              |
| famsize      | 0              |
| Pstatus      | 0              |
| Medu         | 0              |
| Fedu         | 0              |
| Mjob         | 0              |
| Fjob         | 0              |
| reason       | 0              |
| guardian     | 0              |
| traveltime   | 0              |
| studytime    | 0              |
| failures     | 0              |
| schoolsup    | 0              |
| famsup       | 0              |
| paid         | 0              |
| activities   | 0              |
| nursery      | 0              |
| higher       | 0              |
| internet     | 0              |
| romantic     | 0              |
| famrel       | 0              |
| freetime     | 0              |
| goout        | 0              |
| Dalc         | 0              |
| Walc         | 0              |
| health       | 0              |
| absences     | 0              |
| G1           | 0              |
| G2           | 0              |
| G3           | 0              |



> *It shows that there are no missing values in the dataset.*
### 3.2 Encoding Categorical Variables
- Converting categorical variables into a format that can be provided to machine learning algorithms.
```python
# Convert categorical variables into a new binary column using one-hot encoding
sdata_encoded = pd.get_dummies(sdata, drop_first=True)
```

### 3.3 Outliers
- Calculating Z-scores for all columns to identify data points that are far from the mean.
```python
# Calculate Z-scores for all columns (numeric and one-hot encoded)
z_scores = np.abs((sdata_encoded - sdata_encoded.mean()) / sdata_encoded.std())

# Display the first 10 rows of Z-scores
z_scores.head(10)
```
> *Displaying the first 10 rows of Z-scores*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>...</th>
      <th>guardian_mother</th>
      <th>guardian_other</th>
      <th>schoolsup_yes</th>
      <th>famsup_yes</th>
      <th>paid_yes</th>
      <th>activities_yes</th>
      <th>nursery_yes</th>
      <th>higher_yes</th>
      <th>internet_yes</th>
      <th>romantic_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.030900</td>
      <td>1.309206</td>
      <td>1.539528</td>
      <td>0.576274</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>0.171514</td>
      <td>0.693250</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>2.920779</td>
      <td>1.258258</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>1.814644</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.209975</td>
      <td>1.335010</td>
      <td>1.187916</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>1.118885</td>
      <td>0.171514</td>
      <td>0.157259</td>
      <td>0.543136</td>
      <td>...</td>
      <td>1.530277</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>2.015947</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.431876</td>
      <td>1.335010</td>
      <td>1.187916</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>0.171514</td>
      <td>1.007768</td>
      <td>0.538138</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>2.920779</td>
      <td>1.258258</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.431876</td>
      <td>1.309206</td>
      <td>0.278768</td>
      <td>0.759446</td>
      <td>1.289120</td>
      <td>0.374017</td>
      <td>0.973785</td>
      <td>1.122905</td>
      <td>1.007768</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>1.028924</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>1.308754</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.610951</td>
      <td>0.427801</td>
      <td>0.630380</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>0.171514</td>
      <td>1.007768</td>
      <td>0.543136</td>
      <td>...</td>
      <td>1.530277</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>1.814644</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.610951</td>
      <td>1.309206</td>
      <td>0.630380</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>1.118885</td>
      <td>0.779877</td>
      <td>1.007768</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>1.028924</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.610951</td>
      <td>0.453605</td>
      <td>0.278768</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>0.779877</td>
      <td>0.693250</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>1.258258</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.209975</td>
      <td>1.309206</td>
      <td>1.539528</td>
      <td>0.576274</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>2.074296</td>
      <td>0.693250</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>2.920779</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>1.814644</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.431876</td>
      <td>0.427801</td>
      <td>0.278768</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>0.072550</td>
      <td>1.122905</td>
      <td>1.007768</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>0.970392</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.431876</td>
      <td>0.427801</td>
      <td>1.539528</td>
      <td>0.759446</td>
      <td>0.083588</td>
      <td>0.374017</td>
      <td>1.118885</td>
      <td>1.731268</td>
      <td>1.858278</td>
      <td>0.543136</td>
      <td>...</td>
      <td>0.652470</td>
      <td>0.259481</td>
      <td>0.341847</td>
      <td>0.793525</td>
      <td>0.252658</td>
      <td>1.028924</td>
      <td>0.495281</td>
      <td>0.344648</td>
      <td>0.550223</td>
      <td>0.762908</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 42 columns</p>
</div>

### 3.3.1 Heatmap of Z-scores to identify outliers
```python
plt.figure(figsize=(12, 8))
sns.heatmap(z_scores, cmap='coolwarm', cbar_kws={'label': 'Z-Score'})
plt.title("Heatmap of Z-Scores", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Data Points", fontsize=12)
plt.grid(False)
plt.show()
```
![image](https://github.com/user-attachments/assets/a80a018c-7b04-4f62-968b-b872b2b27dd5)
> *A heatmap of Z-scores was created to visually inspect potential outliers.*
### 3.3.2 Outliers per column
```python
# Outliers based on Z-score > 3
outliers = (z_scores > 3).sum()

print("Outliers per column:\n", outliers)
```
|                      |       |
|----------------------|-------|
| age                  | 3     |
| Medu                 | 0     |
| Fedu                 | 0     |
| traveltime           | 16    |
| studytime            | 0     |
| failures             | 14    |
| famrel               | 22    |
| freetime             | 0     |
| goout                | 0     |
| Dalc                 | 17    |
| Walc                 | 0     |
| health               | 0     |
| absences             | 11    |
| G1                   | 1     |
| G2                   | 7     |
| G3                   | 16    |
| school_MS            | 0     |
| sex_M                | 0     |
| address_U            | 0     |
| famsize_LE3          | 0     |
| Pstatus_T            | 0     |
| Mjob_health          | 48    |
| Mjob_other           | 0     |
| Mjob_services        | 0     |
| Mjob_teacher         | 0     |
| Fjob_health          | 23    |
| Fjob_other           | 0     |
| Fjob_services        | 0     |
| Fjob_teacher         | 36    |
| reason_home          | 0     |
| reason_other         | 0     |
| reason_reputation    | 0     |
| guardian_mother      | 0     |
| guardian_other       | 41    |
| schoolsup_yes        | 0     |
| famsup_yes           | 0     |
| paid_yes             | 39    |
| activities_yes       | 0     |
| nursery_yes          | 0     |
| higher_yes           | 0     |
| internet_yes         | 0     |
| romantic_yes         | 0     |

## 4. Model Implementation
### 4.1 Getting the inputs and output
```python
# Define your target variable (continuous) and independent variables
X = sdata_encoded.drop(columns='G3')
X
y = sdata_encoded['G3']
y
```
> *This shows the inputs of the dataset.*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>...</th>
      <th>guardian_mother</th>
      <th>guardian_other</th>
      <th>schoolsup_yes</th>
      <th>famsup_yes</th>
      <th>paid_yes</th>
      <th>activities_yes</th>
      <th>nursery_yes</th>
      <th>higher_yes</th>
      <th>internet_yes</th>
      <th>romantic_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>644</th>
      <td>19</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>645</th>
      <td>18</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>646</th>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>647</th>
      <td>17</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>648</th>
      <td>18</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>649 rows × 41 columns</p>
</div>

> *This shows the output of the dataset.*

| Index | G3  |
|-------|-----|
| 0     | 11  |
| 1     | 11  |
| 2     | 12  |
| 3     | 14  |
| 4     | 13  |
| ...   | ... |
| 644   | 10  |
| 645   | 16  |
| 646   | 9   |
| 647   | 10  |
| 648   | 11  |
### 4.2 Creating the Training Set and the Test Set
- Split the dataset into training (80%) and testing (20%) sets.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### 4.3 Normalize data
- Normalize the input features using ```StandardScaler```.
> *Normalizing the dataset since it has features that have different units and scales.*
```python
scaler = StandardScaler()

# Fit and transform on training data and transform on testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
> *Displaying the **normalized version of the input training dataset** (`X_train`)*
```python
X_train_scaled

array([[-0.61043136,  0.42368083, -0.27799922, ...,  0.35813527,
         0.55138018, -0.74111654],
       [ 0.20820914, -1.32148068, -1.18543063, ..., -2.79224107,
         0.55138018, -0.74111654],
       [ 1.02684965,  0.42368083, -0.27799922, ...,  0.35813527,
         0.55138018,  1.34931545],
       ...,
       [-0.61043136,  1.29626158, -0.27799922, ...,  0.35813527,
         0.55138018,  1.34931545],
       [ 0.20820914, -1.32148068, -0.27799922, ...,  0.35813527,
         0.55138018, -0.74111654],
       [ 1.02684965, -1.32148068, -2.09286204, ...,  0.35813527,
        -1.81363067, -0.74111654]])
```
> *Displaying the **normalized version of the input testing dataset** (`X_test`)*
```python
X_test_scaled

array([[-0.61043136, -0.44889993, -1.18543063, ...,  0.35813527,
         0.55138018,  1.34931545],
       [ 0.20820914, -2.19406143, -0.27799922, ...,  0.35813527,
         0.55138018, -0.74111654],
       [ 1.02684965,  0.42368083,  1.53686361, ...,  0.35813527,
         0.55138018, -0.74111654],
       ...,
       [ 1.02684965, -0.44889993, -0.27799922, ...,  0.35813527,
        -1.81363067, -0.74111654],
       [ 0.20820914, -0.44889993, -0.27799922, ..., -2.79224107,
        -1.81363067,  1.34931545],
       [ 1.02684965, -0.44889993, -1.18543063, ...,  0.35813527,
         0.55138018,  1.34931545]])
```


> *Displaying the **output training dataset** (`y_train`)*
```python
y_train

34     12
432     7
399    17
346    13
542    12
       ..
9      13
359    17
192    11
629     9
559    13
Name: G3
```
> *Displaying the **output testing dataset** (`y_test`)*
```python
y_test

532     8
375    15
306    16
625    10
480    10
       ..
403    15
266    14
641    15
558    10
242    11
Name: G3
```


### 4.4 Building the Model
```python
# Initialize the linear regression model
model = LinearRegression()
```
### 4.5 Training the Model
```python
# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)
```
### 4.6 Inference
```python
# Predict the target values for the test data
y_pred = model.predict(X_test_scaled)
y_pred
```
> *Displaying the **predicted target values***
```python
array([ 7.11839166, 15.2987607 , 16.51840561, 10.24362584,  8.88631813,
       12.40264525, 13.11431362, 18.58431904, 11.539958  , 11.22662697,
       10.86332591, 10.20475805, 13.46689866,  7.90904354, 18.52875113,
       12.28220793, 12.90826114, 12.43573679, 10.77504148,  9.94604434,
       12.10420184,  9.96779407, 17.30494134, 13.18451771, 12.64142506,
        0.46555942, 12.66670441, 13.46679886, 10.92486941, 12.76209961,
       14.0807856 , 16.53398922, 13.1430624 , 16.02934565, 12.7784794 ,
        9.03767647,  8.86226118, 11.36992111, 13.11354283, 11.28744116,
       15.54975067, 17.87695741, 11.34890768, 13.44009004, 12.35722513,
        9.19036267, 12.93904672,  8.54235792, 11.2706524 ,  9.28342951,
        5.31003205, 14.24479115,  8.91115527, 12.2377876 ,  6.6539997 ,
       11.6800828 , 11.69228473, 11.35474737, 14.51744483, 14.65246459,
       13.85454817,  6.87732712, 11.52813832,  8.8940211 , 13.15243389,
       12.35231133, 11.87723093, 12.93668168, 14.85414503,  6.40595837,
        8.05347897, 11.02559697, 13.93846354, 10.69657169, 14.42980161,
       13.83815397, 13.99634797, 11.57891697, 13.32203274, 12.13489736,
       14.52900954,  8.55483733, 10.05701427, 13.5473069 , 18.55515647,
       11.00386865, 11.04680672, 13.67212462, 12.90097445, 13.26617422,
       11.28592914, 15.94979713, 18.51247502, 11.89448638,  7.48188078,
       10.36704683, 13.44249519, 11.6493244 , 13.13071069, 14.47281931,
       12.470212  ,  9.46829343,  5.31283499, 10.92650604,  9.87709488,
       11.35712867, 16.90314565, 10.59765663, 10.12583098, 15.52766888,
       11.43589043, 13.25478   , 14.52004618, 13.4716261 , 11.74147721,
       10.2534332 , 15.80407921, 16.10581745, 10.53754751, 11.75796434,
        7.56688346, 10.49256986,  9.48829377, 10.45732009, 13.24003928,
       15.95107391, 15.20441861, 16.71540397, 11.97115601, 10.51095552])
```
> *Comparing the predicted values to the **output testing dataset** (y_test)*
```python
532     8
375    15
306    16
625    10
480    10
       ..
403    15
266    14
641    15
558    10
242    11
```

## 5. Evaluating the Model:
### 5.1 Mean Squared Error
```python
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
```
```python
Mean Squared Error (MSE): 1.02
```
> *The **Mean Squared Error(MSE)** is 1.02, providing insights into the average prediction errors.*

### 5.2 Mean Absolute Error
```python
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
```
```python
Mean Absolute Error (MAE): 0.75
```
> *The **Mean Absolute Error(MAE)** is 0.75, providing insights into the average prediction errors.*
### 5.3 R-squared
```python
# Calculate R-squared (r2)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.2f}")
```
```python
R-squared (R²): 0.86
```
> *The model achieved an **R²** score of 0.86, indicating that it explains 86% of the variance in student achievement.*

## 6. Interpretation:
```python
# Coefficients of the model
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
# Print the coefficients
coefficients.head(15)
```
> *A positive coefficient means that as the feature increases, the target variable increases. While a negative coefficient means that as the feature increases, the target variable decreases.*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>0.011230</td>
    </tr>
    <tr>
      <th>Medu</th>
      <td>-0.130062</td>
    </tr>
    <tr>
      <th>Fedu</th>
      <td>0.031043</td>
    </tr>
    <tr>
      <th>traveltime</th>
      <td>0.109396</td>
    </tr>
    <tr>
      <th>studytime</th>
      <td>0.039781</td>
    </tr>
    <tr>
      <th>failures</th>
      <td>-0.175474</td>
    </tr>
    <tr>
      <th>famrel</th>
      <td>-0.039511</td>
    </tr>
    <tr>
      <th>freetime</th>
      <td>-0.046087</td>
    </tr>
    <tr>
      <th>goout</th>
      <td>-0.015311</td>
    </tr>
    <tr>
      <th>Dalc</th>
      <td>-0.016356</td>
    </tr>
    <tr>
      <th>Walc</th>
      <td>-0.055358</td>
    </tr>
    <tr>
      <th>health</th>
      <td>-0.062430</td>
    </tr>
    <tr>
      <th>absences</th>
      <td>0.072488</td>
    </tr>
    <tr>
      <th>G1</th>
      <td>0.351990</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>2.662384</td>
    </tr>
  </tbody>
</table>
</div>


> *Features with the highest absolute coefficients, such as G1 and G2, appear to have the most significant influence on the target variable.*


###  6.1 Comparison of the actual vs. predicted values from the machine learning model.
- The scatter plot indicates that most predictions lie close to the ideal line, suggesting reasonable predictive accuracy.
```python
# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', s=70)  # Add alpha for transparency

# Add a diagonal line representing perfect predictions
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.fill_between([min_val, max_val], [min_val - 2, max_val - 2], [min_val + 2, max_val + 2], color='gray', alpha=0.2)
# Set labels and title
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Actual vs Predicted', fontsize=14)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Show the plot
plt.show()

```
> *The plot presents the evaluation of the performance of a regression model by showing how closely the model's predictions align with the true (actual) values.*

![image](https://github.com/user-attachments/assets/062fdf53-9361-4ba0-a91e-517fd26814c4)


Logistic Regression - HEART DISEASE DATASET

Importing Libraries and Dataset
The analysis began by importing the necessary libraries, with pandas being the primary library used for data manipulation. We loaded the dataset into a DataFrame using the pd.read_csv() function, which allows us to read data from a CSV file easily. This step is essential as it provides the foundation for further analysis.
![image](https://github.com/user-attachments/assets/1e9d521b-f5e9-4a87-a9cf-961c6c985a8b)

Overview of the Dataset
To understand the structure of the dataset, we used the dataset.info() method. This command provides a summary of the dataset, including the number of entries (rows), the types of data in each column, and any potential issues. This overview is vital for identifying what kind of preprocessing steps might be necessary.
![image](https://github.com/user-attachments/assets/28a5d7cc-6cbd-410f-bd58-3e6880cc9c98)

Handling Missing Values
To ensure the integrity of our dataset, we checked for any missing values using dataset.isna().sum(). This method shows the total number of missing values for each column. If any missing values were present, we would need to decide how to handle them (e.g., by filling them in with the mean or median of the column, or removing the affected rows). In our case, we confirmed there were no missing values, which simplified our preprocessing steps.
![image](https://github.com/user-attachments/assets/43e4e830-d3a6-4066-9537-ff9a345a43ae)

Separating Features and Target
After ensuring the data was clean, we separated the independent variables (features) from the dependent variable (target). The features, stored in X, consisted of all columns except the last one, while the target variable, stored in y, contained only the last column. This separation is crucial as it allows the model to learn from the features in order to predict the target.
![image](https://github.com/user-attachments/assets/f175e2ad-05df-4369-8210-830199560e04)

Train-Test Split
To evaluate the performance of our model accurately, we divided the dataset into two parts: training and testing. We used train_test_split to create these sets, with an 80-20 split, meaning 80% of the data was used for training the model, and 20% was reserved for testing it. This separation is important because it allows us to train the model on one set of data and then evaluate its performance on a different, unseen set.
![image](https://github.com/user-attachments/assets/06d771ed-7269-48b6-946f-8a32a2e7efa9)

Feature Scaling
To ensure that all features contribute equally to the model's predictions, we standardized the features using StandardScaler. This process adjusts the features so they have a mean of 0 and a standard deviation of 1. Feature scaling is particularly important for algorithms like logistic regression, as it can improve the model’s performance and convergence speed.
![image](https://github.com/user-attachments/assets/28083f26-62a0-4591-adae-4a130d186f1f)

Building and Training the Model
After preprocessing the data, we moved on to building and training the model.

Logistic Regression
We chose logistic regression as our classification algorithm because it is a widely used method for binary classification problems. We initialized the logistic regression model and trained it using the training dataset. During this training process, the model learned the relationships between the features and the target variable, allowing it to make predictions based on the input data.
![image](https://github.com/user-attachments/assets/edc3c966-c760-40de-87dc-0544f9ade71e)

Making Predictions
Once the model was trained, we used it to make predictions on the test set. This step is crucial for assessing how well our model generalizes to new data. We also demonstrated how to predict the outcome for a single data point to illustrate the model's practical application.
![image](https://github.com/user-attachments/assets/17a5865e-6db6-4350-9de5-bdda4d15eb66)
These correlations provide insights into which factors are most influential in predicting heart disease and can guide the development of predictive models

-age:

Positively correlated with trestbps (0.283), chol (0.207), and ca (0.302), indicating that older age is associated with higher resting blood pressure, higher cholesterol, and more major vessels colored by fluoroscopy. Negatively correlated with thalach (-0.395), indicating that older age is associated with lower maximum heart rate achieved. Negatively correlated with target (-0.221), indicating that older age is associated with a lower likelihood of having heart disease (though this might be counterintuitive and could indicate a complex relationship or influence of other factors).

-sex:

Negatively correlated with chol (-0.196) and target (-0.284), indicating that females are less likely to have higher cholesterol and heart disease compared to males. cp (chest pain type):

Positively correlated with thalach (0.293) and target (0.432), indicating that certain types of chest pain are associated with higher maximum heart rate achieved and a higher likelihood of heart disease. Negatively correlated with exang (-0.393), indicating that certain types of chest pain are less likely to be associated with exercise-induced angina.

-trestbps:

Positively correlated with oldpeak (0.195), indicating higher resting blood pressure is associated with higher ST depression induced by exercise. Negatively correlated with target (-0.146), indicating higher resting blood pressure is associated with a lower likelihood of heart disease (again, this could indicate a complex relationship or the influence of other factors).

-thalach:

Positively correlated with cp (0.293) and target (0.420), indicating higher maximum heart rate achieved is associated with certain types of chest pain and a higher likelihood of heart disease. Negatively correlated with age (-0.395), exang (-0.377), and oldpeak (-0.342), indicating that lower maximum heart rate achieved is associated with older age, exercise-induced angina, and higher ST depression induced by exercise.

-exang:

Positively correlated with oldpeak (0.287), indicating that exercise-induced angina is associated with higher ST depression induced by exercise. Negatively correlated with cp (-0.393), thalach (-0.377), and target (-0.436), indicating that exercise-induced angina is associated with certain types of chest pain, lower maximum heart rate achieved, and a lower likelihood of heart disease.

-oldpeak:

Positively correlated with exang (0.287) and ca (0.237), indicating higher ST depression induced by exercise is associated with exercise-induced angina and more major vessels colored by fluoroscopy. Negatively correlated with thalach (-0.342), slope (-0.576), and target (-0.429), indicating that higher ST depression induced by exercise is associated with lower maximum heart rate achieved, less favorable slope of the peak exercise ST segment, and a lower likelihood of heart disease.

-ca:

Positively correlated with age (0.302), exang (0.125), and oldpeak (0.237), indicating more major vessels colored by fluoroscopy are associated with older age, exercise-induced angina, and higher ST depression induced by exercise. Negatively correlated with thalach (-0.228) and target (-0.409), indicating that more major vessels colored by fluoroscopy are associated with lower maximum heart rate achieved and a lower likelihood of heart disease.

-thal:

Positively correlated with sex (0.211), indicating a gender difference in thalassemia. Negatively correlated with target (-0.343), indicating that certain thalassemia conditions are associated with a lower likelihood of heart disease.

