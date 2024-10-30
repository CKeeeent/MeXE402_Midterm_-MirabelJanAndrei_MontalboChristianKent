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
## Importing the required libraries, modules, and functions.
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
## Importing the dataset.
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
## Data Preprocessing:
### Handle Missing Values
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
### Encoding Categorical Variables
```python
# Convert categorical variables into a new binary column using one-hot encoding
sdata_encoded = pd.get_dummies(sdata, drop_first=True)
```
> *Converting categorical variables into a format that can be provided to machine learning algorithms.*

### Outliers
> *Calculating Z-scores for all columns*
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

### Heatmap of Z-scores to identify outliers
```python
plt.figure(figsize=(12, 8))
sns.heatmap(z_scores, cmap='coolwarm', cbar_kws={'label': 'Z-Score'})
plt.title("Heatmap of Z-Scores", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Data Points", fontsize=12)
plt.grid(False)
plt.show()
```
![image](https://github.com/user-attachments/assets/fa749839-9c29-493e-acf8-1689d9817c78)
> *This heatmap visualizes Z-scores for various features to spot rows with potential outliers.*
### Outliers per column
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

### Removing rows with outliers
```python
# Remove rows with outliers
sdata_clean = sdata[(z_scores < 3).all(axis=1)]
```
> *Removing rows with outliers to improve model performance.*

## Model Implementation:
### Getting the inputs and output
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
      <td>1</t
