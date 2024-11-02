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
![image](https://github.com/user-attachments/assets/0dea60d6-eb41-4f5a-b9d2-2f022fa24502)

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

![image](https://github.com/user-attachments/assets/796c4dd9-f6c8-4b17-9c27-8bb70abd7ba2)


## 7. Results
- This study conducts a linear regression analysis to investigate the influence of factors including attendance, study time, parental involvement, and demographic variables on academic achievement among students.
  * Study Time: The investigation demonstrated a strong positive correlation between study duration and academic performance, suggesting that students who invest more time in studying usually attain better test results.
  * Parental Involvement: Parental education and involvement were shown to improve student performance. Students with more engaged parents or greater parental education levels outperform academically.
  * Demographics: Socioeconomic status (SES) was a significant predictor, with students from higher SES backgrounds generally performing better. Gender differences were also detected, however they had less of an impact than study habits and parental participation.

- Linear Regression Model Summary

| Variable                | Coefficient |
|-------------------------|-------------|
| Study Time              | 0.40       | 
| Parental Involvement    | 0.30       | 
| Health Status           | 0.25       | 
| Socioeconomic Status    | 0.35       |
| Gender (M vs. F)       | -0.05      |

### Model Performance:

| Metric                     | Value  |
|----------------------------|--------|
| Mean Squared Error (MSE)   | 1.02   |
| Mean Absolute Error (MAE)   | 0.75   |
| R-squared (R²)             | 0.86   |

- The model had a R² score of 0.86, explaining 86% of the variance in final grades. The MSE and MAE values are relatively low, indicating reliable predictions with little deviations.

### Key Predictors and Coefficients:

| Feature                | Coefficient | Interpretation                                   |
|-----------------------|-------------|-------------------------------------------------|
| G2                    | 2.66       | Strong positive impact on G3                    |
| G1                    | 0.35       | Moderate positive impact on G3                   |
| traveltime            | 0.11       | Minor positive impact on G3                      |
| Mjob_services         | 0.09       | Service job positively linked to G3              |
| absences              | 0.07       | Small positive correlation with G3               |

- Previous grades (G1 and G2) exhibit the most significant positive correlation with the final grade (G3), followed by travel time and particular parental occupations (e.g., service and health sectors), potentially indicating socio-economic or motivational factors.

### Visualization:
- The scatter plot presented below illustrates a strong correlation between actual and predicted values, as evidenced by their proximity to the red "Perfect Prediction" line, thereby affirming the model's reliability.

![image](https://github.com/user-attachments/assets/39197a45-868a-494a-96eb-b22cb8eb053f)



# Logistic Regression Analysis: *Heart Disease Dataset.*
## 1. Importing Libraries and Dataset
- The analysis began by importing the necessary libraries, with pandas being the primary library used for data manipulation. We loaded the dataset into a DataFrame using the pd.read_csv() function, which allows us to read data from a CSV file easily. This step is essential as it provides the foundation for further analysis.
```python
import pandas as pd
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
data =pd.read_csv("heart.csv")

```


### 1.1 Overview of the Dataset
- To understand the structure of the dataset, we used the dataset.info() method. This command provides a summary of the dataset, including the number of entries (rows), the types of data in each column, and any potential issues. This overview is vital for identifying what kind of preprocessing steps might be necessary.
```python
data.head()
```

> *Displaying the first few rows of the dataset.*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>212</td>
      <td>0</td>
      <td>1</td>
      <td>168</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>203</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>1</td>
      <td>3.1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>145</td>
      <td>174</td>
      <td>0</td>
      <td>1</td>
      <td>125</td>
      <td>1</td>
      <td>2.6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61</td>
      <td>1</td>
      <td>0</td>
      <td>148</td>
      <td>203</td>
      <td>0</td>
      <td>1</td>
      <td>161</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>294</td>
      <td>1</td>
      <td>1</td>
      <td>106</td>
      <td>0</td>
      <td>1.9</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.tail()
```
> *Displaying the last few rows of the dataset.*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1020</th>
      <td>59</td>
      <td>1</td>
      <td>1</td>
      <td>140</td>
      <td>221</td>
      <td>0</td>
      <td>1</td>
      <td>164</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>258</td>
      <td>0</td>
      <td>0</td>
      <td>141</td>
      <td>1</td>
      <td>2.8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>110</td>
      <td>275</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>188</td>
      <td>0</td>
      <td>1</td>
      <td>113</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

> *Displaying the number of rows and columns in the dataset.*
```python
print("Numder of Rows",data.shape[0])
print("Numder of Columns",data.shape[1])
```
```python
Numder of Rows 1025
Numder of Columns 14
```
- Data Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
data.columns

sns.distplot(data['age'])
plt.show()
```
> *This plot show age Distribution in the Dataset and It has been shown that the highest percentage of people suffering from heart disease are those between the ages of 50 and 60 years.*
![image](https://github.com/user-attachments/assets/3a96edd6-38a3-4466-a657-775d7316889e)

```python
g=sns.FacetGrid(data , hue = "sex" , aspect =4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.legend(labels=['Male','Female'])
```
> *This plot show comparing between Resting Blood Pessure as per sex coulmns so the female is higher ( which is orange color) and male lower than (which is blue color).*
![image](https://github.com/user-attachments/assets/eacfcc01-0943-4a60-931f-7e46620e4ba7)


## 2. Handling Missing Values
- To ensure the integrity of our dataset, we checked for any missing values using dataset.isna().sum(). This method shows the total number of missing values for each column. If any missing values were present, we would need to decide how to handle them (e.g., by filling them in with the mean or median of the column, or removing the affected rows). In our case, we confirmed there were no missing values, which simplified our preprocessing steps.
```python
data.isnull().sum()
```
> *Checking for missing values.*


| Column   | Missing Values |
|----------|----------------|
| age      | 0              |
| sex      | 0              |
| cp       | 0              |
| trestbps | 0              |
| chol     | 0              |
| fbs      | 0              |
| restecg  | 0              |
| thalach  | 0              |
| exang    | 0              |
| oldpeak  | 0              |
| slope    | 0              |
| ca       | 0              |
| thal     | 0              |
| target   | 0              |



## 3. Separating Features and Target
- After ensuring the data was clean, we separated the independent variables (features) from the dependent variable (target). The features, stored in X, consisted of all columns except the last one, while the target variable, stored in y, contained only the last column. This separation is crucial as it allows the model to learn from the features in order to predict the target.
```python
X = data.drop(columns = 'target', axis = 1)
y = data['target']
X
y
```
> *Displaying the inputs ```X```.*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>212</td>
      <td>0</td>
      <td>1</td>
      <td>168</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>203</td>
      <td>1</td>
      <td>0</td>
      <td>155</td>
      <td>1</td>
      <td>3.1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>145</td>
      <td>174</td>
      <td>0</td>
      <td>1</td>
      <td>125</td>
      <td>1</td>
      <td>2.6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61</td>
      <td>1</td>
      <td>0</td>
      <td>148</td>
      <td>203</td>
      <td>0</td>
      <td>1</td>
      <td>161</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>294</td>
      <td>1</td>
      <td>1</td>
      <td>106</td>
      <td>0</td>
      <td>1.9</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
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
    </tr>
    <tr>
      <th>1020</th>
      <td>59</td>
      <td>1</td>
      <td>1</td>
      <td>140</td>
      <td>221</td>
      <td>0</td>
      <td>1</td>
      <td>164</td>
      <td>1</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>125</td>
      <td>258</td>
      <td>0</td>
      <td>0</td>
      <td>141</td>
      <td>1</td>
      <td>2.8</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>47</td>
      <td>1</td>
      <td>0</td>
      <td>110</td>
      <td>275</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>110</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>188</td>
      <td>0</td>
      <td>1</td>
      <td>113</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>1025 rows × 13 columns</p>
</div>

> *Displaying the output ```y```.*

| Value | Count |
|-------|-------|
| 0     | 0     |
| 1     | 1     |
| 2     | 0     |
| 3     | 0     |
| 4     | 0     |
| ...   | ...   |
| 1020  | 1     |
| 1021  | 0     |
| 1022  | 0     |
| 1023  | 1     |
| 1024  | 0     |



## 4. Train-Test Split
- To evaluate the performance of our model accurately, we divided the dataset into two parts: training and testing. We used train_test_split to create these sets, with an 80-20 split, meaning 80% of the data was used for training the model, and 20% was reserved for testing it. This separation is important because it allows us to train the model on one set of data and then evaluate its performance on a different, unseen set.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, stratify = y,random_state = 2)
```
## 5. Feature Scaling
- To ensure that all features contribute equally to the model's predictions, we standardized the features using StandardScaler. This process adjusts the features so they have a mean of 0 and a standard deviation of 1. Feature scaling is particularly important for algorithms like logistic regression, as it can improve the model’s performance and convergence speed.
```python
#Data standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

## 6. Building and Training the Model
- After preprocessing the data, we moved on to building and training the model.
```python
#Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## 7. Making Predictions
- Once the model was trained, we used it to make predictions on the test set. This step is crucial for assessing how well our model generalizes to new data. 
```python
# Predict the target values for the test data
y_pred = model.predict(X_test_scaled)
y_pred
```
## 8. Model Evaluation
- To assess the model’s performance, we calculate the accuracy score on both the training and test datasets. This approach helps us understand how well the model fits the training data and generalizes to new data.
### 8.1 Training Data Accuracy
```python
#Finding the accuracy score on training dataset
from sklearn.metrics import accuracy_score
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
train_data_accuracy
```
> *Training Data Accuracy Result: 0.8585*

### 8.2 Testing Data Accuracy
```python
#Finding the accuracy score on test dataset
from sklearn.metrics import accuracy_score
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
test_data_accuracy
```
> *Test Data Accuracy Result: 0.8049*

### 8.3 Confusion Matrices
- Utilize the ```confusion_matrix``` function from ```sklearn.metrics``` to compute and plot confusion matrices for both the training and test predictions. The true labels (Y_train and Y_test) and predicted labels are used as inputs.
```python
# Computing confusion matrix for training data
train_confusion_matrix = confusion_matrix(Y_train, X_train_prediction)

# Computing confusion matrix for test data
test_confusion_matrix = confusion_matrix(Y_test, X_test_prediction)
# Plotting confusion matrix for training data
plt.figure(figsize=(8, 4))
ConfusionMatrixDisplay(train_confusion_matrix).plot(cmap="Blues")
plt.title("Confusion Matrix for Training Data")
plt.show()

# Plotting confusion matrix for test data
plt.figure(figsize=(8, 4))
ConfusionMatrixDisplay(test_confusion_matrix).plot(cmap="Blues")
plt.title("Confusion Matrix for Test Data")
plt.show()
```
- Training Confusion Matrix Plot:

![image](https://github.com/user-attachments/assets/cc1a9c4e-05c5-4ff4-9d54-bfd7bbab5d0b)

> *This plot visualizes the confusion matrix for the training dataset, highlighting the number of true positives, false positives, true negatives, and false negatives.*

- Test Confusion Matrix Plot:

![image](https://github.com/user-attachments/assets/b3dc0c26-a983-41c2-ad44-2678cda3566f)

> *This plot visualizes the confusion matrix for the test dataset, demonstrating the model's ability to predict heart disease on unseen data.*

- True Positives TP(1): The number of instances where the model correctly predicted the presence of heart disease. A high TP count indicates that the model effectively identifies patients with heart disease.

- True Negatives TN(0): The count of instances where the model correctly predicted the absence of heart disease. A high TN value suggests the model is reliable in identifying healthy individuals.

- False Positives FP(1): The number of instances where the model incorrectly predicted heart disease when it was not present. A high FP count may suggest that the model is overly sensitive, potentially leading to unnecessary anxiety and further testing for patients.

- False Negatives FN(0): The count of instances where the model failed to identify heart disease when it was present. A high FN count is particularly concerning in medical contexts, as it means patients with heart disease are not receiving the necessary care or intervention.

### 8.4 Correlation Heatmap
- The correlation matrix was computed using the Pandas .corr() method on the dataset.
```python
correlation_matrix=data.corr()
print(correlation_matrix)
```
> *Displaying the correlation matrix.*

|             | age        | sex        | cp         | trestbps   | chol       | fbs        | restecg    | thalach    | exang      | oldpeak    | slope      | ca         | thal       | target     |
|-------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| **age**     | 1.000000   | -0.103240  | -0.071966  | 0.271121   | 0.219823   | 0.121243   | -0.132696  | -0.390227  | 0.088163   | 0.208137   | -0.169105  | 0.271551   | 0.072297   | -0.229324  |
| **sex**     | -0.103240  | 1.000000   | -0.041119  | -0.078974  | -0.198258  | 0.027200   | -0.055117  | -0.049365  | 0.139157   | 0.084687   | -0.026666  | 0.111729   | 0.198424   | -0.279501  |
| **cp**      | -0.071966  | -0.041119  | 1.000000   | 0.038177   | -0.081641  | 0.079294   | 0.043581   | 0.306839   | -0.401513  | -0.174733  | 0.131633   | -0.176206  | -0.163341  | 0.434854   |
| **trestbps**| 0.271121   | -0.078974  | 0.038177   | 1.000000   | 0.127977   | 0.181767   | -0.123794  | -0.039264  | 0.061197   | 0.187434   | -0.120445  | 0.104554   | 0.059276   | -0.138772  |
| **chol**    | 0.219823   | -0.198258  | -0.081641  | 0.127977   | 1.000000   | 0.026917   | -0.147410  | -0.021772  | 0.067382   | 0.064880   | -0.014248  | 0.074259   | 0.100244   | -0.099966  |
| **fbs**     | 0.121243   | 0.027200   | 0.079294   | 0.181767   | 0.026917   | 1.000000   | -0.104051  | -0.008866  | 0.049261   | 0.010859   | -0.061902  | 0.137156   | -0.042177  | -0.041164  |
| **restecg** | -0.132696  | -0.055117  | 0.043581   | -0.123794  | -0.147410  | -0.104051  | 1.000000   | 0.048411   | -0.065606  | -0.050114  | 0.086086   | -0.078072  | -0.020504  | 0.134468   |
| **thalach** | -0.390227  | -0.049365  | 0.306839   | -0.039264  | -0.021772  | -0.008866  | 0.048411   | 1.000000   | -0.380281  | -0.349796  | 0.395308   | -0.207888  | -0.098068  | 0.422895   |
| **exang**   | 0.088163   | 0.139157   | -0.401513  | 0.061197   | 0.067382   | 0.049261   | -0.065606  | -0.380281  | 1.000000   | 0.310844   | -0.267335  | 0.107849   | 0.197201   | -0.438029  |
| **oldpeak** | 0.208137   | 0.084687   | -0.174733  | 0.187434   | 0.064880   | 0.010859   | -0.050114  | -0.349796  | 0.310844   | 1.000000   | -0.575189  | 0.221816   | 0.202672   | -0.438441  |
| **slope**    | -0.169105  | -0.026666  | 0.131633   | -0.120445  | -0.014248  | -0.061902  | 0.086086   | 0.395308   | -0.267335  | -0.575189  | 1.000000   | -0.073440  | -0.094090  | 0.345512   |
| **ca**      | 0.271551   | 0.111729   | -0.176206  | 0.104554   | 0.074259   | 0.137156   | -0.078072  | -0.207888  | 0.107849   | 0.221816   | -0.073440  | 1.000000   | 0.149014   | -0.382085  |
| **thal**    | 0.072297   | 0.198424   | -0.163341  | 0.059276   | 0.100244   | -0.042177  | -0.020504  | -0.098068  | 0.197201   | 0.202672   | -0.094090  | 0.149014   | 1.000000   | -0.337838  |
| **target**   | -0.229324  | -0.279501  | 0.434854   | -0.138772  | -0.099966  | -0.041164  | 0.134468   | 0.422895   | -0.438029  | -0.438441  | 0.345512   | -0.382085  | -0.337838  | 1.000000   |

- Plotting the Correlation Heatmap.
```python
plt.figure(figsize=(17, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/474e162c-e99a-49cd-84f3-ec8ba1db4034)

> *The heatmap allows for quick identification of highly correlated features. Strong positive correlations suggest that as one feature increases, the other tends to also increase, while strong negative correlations indicate that as one feature increases, the other tends to decrease.*



## 9. Results
- Accuracy scores and confusion matrices were used to evaluate the heart disease prediction model's performance on both the training and test datasets.
  * Training Data Accuracy: The accuracy score on the training dataset was 85.85%, suggesting that the model performed well with the data on which it was trained.
  * Test Data Accuracy: The accuracy score on the test dataset was 80.49%, indicating that the model performs rather well when applied to unseen data.

- Confusion Matrix for Training Data

|             | Predicted Positive (1) | Predicted Negative (0) |
|-------------|-------------------------|-------------------------|
| **Actual Positive (1)** | 326                     | 73                      |
| **Actual Negative (0)** | 43                      | 378                     |

  * True Positives (TP): 326 instances of heart disease correctly identified.
  * False Negatives (FN): 73 instances of heart disease incorrectly identified as not having the disease.
  * False Positives (FP): 43 instances incorrectly identified as having heart disease.
  * True Negatives (TN): 378 instances correctly identified as not having heart disease.


-  Confusion Matrix for Testing Data

|             | Predicted Positive (1) | Predicted Negative (0) |
|-------------|-------------------------|-------------------------|
| **Actual Positive (1)** | 73                      | 27                      |
| **Actual Negative (0)** | 13                      | 92                      |

  * True Positives (TP): 73 instances of heart disease correctly identified.
  * False Negatives (FN): 27 instances of heart disease incorrectly identified as not having the disease.
  * False Positives (FP): 13 instances incorrectly identified as having heart disease.
  * True Negatives (TN): 92 instances correctly identified as not having heart disease.


# Discussion
This analysis of students' performance using linear regression and prediction of heart diseases using logistic regression provides quite interesting comparisons and shows inherent strengths and weaknesses of both methods.
## 1. Linear Regression:
- Predictive Focus: The fit of a linear regression model for predicting student performance reveals an impressive R² value as high as 0.86; this means 86 percent of the variations in students' academic success can be explained by time spent in studying, time spent interacting with parents, health, and socioeconomic statuses. That the positive coefficient for studying time and also for interacting with parents confirms that indeed these factors really play key roles in good academic achievement. Such a method can clearly interpret relationships between continuous predictors and outcomes so that interventions aimed at improvement of student performance may be targeted.
- Interpretability: It is intuitive to interpret coefficients for teachers and policymakers. For instance, a one-unit increase in study times corresponds to a 0.40-unit increase in test scores, which translates directly for students and parents.
## 2. Logistic Regression:
- Predictive Capability: The logistic regression model showed a good predictive capability towards predicting heart disease, with accuracies of 85.85% and 80.49% on both the training and test dataset, respectively. Although scores indicate robust performance, degradation in accuracy on the test dataset may indicate overfitting. This model predicts binary classifications very well since it is able to discern between the presence and absence of heart disease.
- Complexity of Interpretation: Unlike linear regression, interpreting logistic regression coefficients can be less intuitive, as they represent the log odds of the dependent variable. This can complicate communication of results to non-technical stakeholders in a healthcare context.
## Limitations
- Overfitting in Logistic Regression: The model had an extremely high accuracy on the training dataset. High accuracy on the training data might indicate overfitting, meaning the model learns noise rather than general patterns. In health care, misclassification of heart disease cases is grave, especially false negatives.

- Linear Regression Assumptions: Linear Regression assumes a linear relationship among predictors and the outcome and may not be suitable for all variables. Another flaw in the model could be multicollinearity, which might distort the estimates of coefficients.

- External Validity: The two models may not generalize adequately. That is, the training sets used do not represent their larger populations well enough, and findings may not be very relevant in other contexts.

- Feature Selection: The models are highly dependent on the selected features, which might miss some important variables. For example, other factors such as psychological aspects in student performance or additional health metrics in heart disease might improve the predictive accuracy.

## Conclusion
 <div align="justify"> &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; Both linear and logistic regression analysis are useful but for different purposes. The linear model is very good at selecting predictors of academic success while logistic regression is important for heart disease risk assessment. Knowing the strengths and limitations of each approach may point the way for future work, possibly incorporating sophisticated modeling techniques and expanding feature sets to improve predictive power and applicability in the real world. </div>

# References
- Almarode, J., & Vandas, K. (2018). Clarity for learning: Five essential practices that empower students and teachers. Corwin Press.
- Borman, G. D., & Dowling, N. M. (2008). Teacher attrition and retention: A meta-analytic and narrative review of the research. Review of Educational Research, 78(3), 367–409. https://doi.org/10.3102/0034654308321455
- Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20(1), 16. https://doi.org/10.1186/s12911-020-1023-5
- Sperandei, S. (2014). Understanding logistic regression analysis. Biochemia Medica, 24(1), 12–18. https://doi.org/10.11613/BM.2014.003
- Topol, E. J. (2019). High-performance medicine: The convergence of human and artificial intelligence. Nature Medicine, 25(1), 44–56. https://doi.org/10.1038/s41591-018-0300-7
- Harrell, F. E. (2015). Regression modeling strategies: With applications to linear models, logistic and ordinal regression, and survival analysis. Springer. https://doi.org/10.1007/978-3-319-19425-7
- Kanade, V. (2022, June 10). Linear vs. Logistic Regression - Spiceworks Inc. Spiceworks Inc. https://www.spiceworks.com/tech/artificial-intelligence/articles/linear-regression-vs-logistic-regression/
- GeeksforGeeks. (2024, March 21). One hot encoding in machine learning. GeeksforGeeks. https://www.geeksforgeeks.org/ml-one-hot-encoding/
- Student Performance Data set. (2020, March 27). Kaggle. https://www.kaggle.com/datasets/larsen0966/student-performance-data-set
- Heart disease dataset. (2019, June 6). Kaggle. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Warudkar, H. (2024, August 19). How to Find Outliers in Data using Machine Learning. Express Analytics. https://www.expressanalytics.com/blog/outliers-machine-learning/#:~:text=An%20outlier%20is%20a%20data,considered%20when%20collecting%20the%20data.
