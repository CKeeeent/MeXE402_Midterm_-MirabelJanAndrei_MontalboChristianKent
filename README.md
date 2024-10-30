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
### Creating the Training Set and the Test Set
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
> *Splitting the data (80% training, 20% testing)*

### Normalize data
```python
scaler = StandardScaler()

# Fit and transform on training data and transform on testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
> *Normalizing the dataset since it has features that have different units and scales.*

```python
X_train_scaled

array([[ 0.98793212, -0.43741772, -0.25633653, ...,  0.35125009,
         0.56919205, -0.75993921],
       [-0.62953403,  1.35013974,  1.59142265, ...,  0.35125009,
         0.56919205,  1.31589473],
       [ 0.98793212,  0.45636101, -0.25633653, ...,  0.35125009,
        -1.75687625, -0.75993921],
       ...,
       [-0.62953403,  1.35013974,  1.59142265, ...,  0.35125009,
         0.56919205, -0.75993921],
       [-1.4382671 , -1.33119645, -1.18021613, ...,  0.35125009,
         0.56919205, -0.75993921],
       [-1.4382671 ,  1.35013974,  1.59142265, ...,  0.35125009,
         0.56919205, -0.75993921]])
```
> *Displaying the **normalized version of the input training dataset** (`X_train`)*

```python
X_test_scaled

array([[ 0.98793212,  1.35013974,  1.59142265, ...,  0.35125009,
         0.56919205, -0.75993921],
       [-0.62953403,  0.45636101, -1.18021613, ...,  0.35125009,
         0.56919205, -0.75993921],
       [ 0.98793212,  1.35013974,  1.59142265, ...,  0.35125009,
         0.56919205, -0.75993921],
       ...,
       [-0.62953403,  0.45636101,  1.59142265, ...,  0.35125009,
        -1.75687625, -0.75993921],
       [ 0.17919904,  1.35013974,  0.66754306, ...,  0.35125009,
         0.56919205,  1.31589473],
       [ 0.98793212, -0.43741772, -0.25633653, ...,  0.35125009,
         0.56919205,  1.31589473]])
```

> *Displaying the **normalized version of the input testing dataset** (`X_test`)*

```python
y_train

332    18
29     12
302    12
286    13
554    10
       ..
71     10
106    10
270    15
435    10
102    12
Name: G3
```
> *Displaying the **output training dataset** (`y_train`)*

```python
y_test

636    19
220    12
594    18
429    11
72     11
       ..
514     7
374    17
444    11
244    12
601    10
Name: G3
```
> *Displaying the **output testing dataset** (`y_test`)*

### Building the Model
```python
# Initialize the linear regression model
model = LinearRegression()
```
### Training the Model
```python
# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)
```
### Inference
```python
# Predict the target values for the test data
y_pred = model.predict(X_test_scaled)
y_pred
```
> *Displaying the **predicted target values***
```python
array([18.40265613, 11.82720849, 18.56288575, 10.80969554, 11.74323992,
       16.52050538, 17.68642688,  9.20547185, 10.99263829, 10.52990179,
       18.68507485, 12.01170944, 12.51636949,  9.25637135, 10.92001471,
       13.89458333, 11.70284959,  7.75525634, 15.58827334, 14.91646892,
       15.48017743, 13.6711844 , 14.51763488, 12.14228197, 14.73663429,
       12.89956801,  8.386953  , 11.70486551, 11.30482308, 15.39946248,
       15.91856938, 13.007757  ,  7.94478881,  6.55967433, 17.82710293,
       15.76901234, 14.0460542 , 15.54431689, 13.26315587, 11.43204987,
       13.88933129, 11.05736646,  8.62073014, 11.85157458, 13.27989252,
       13.13368834, 17.89357593, 11.39508466, 12.06570598, 11.36425844,
       11.00499054, 11.16234464, 14.32597736,  9.8592132 , 10.79193041,
       18.01161782,  9.10909354, 10.35558102, 11.4316307 , 10.04193231,
        8.21272376, 11.30013557, 16.13851443, 12.36903381, 15.51853105,
       16.17553342,  9.9042642 ,  7.94076538,  9.50371636,  9.57917018,
       16.13197401, 15.88862171, 12.14110277, 16.71819814, 13.80215043,
       13.44212228, 12.85545806, 15.69804212, 12.46102076, 13.45123399,
       11.78788661, 11.6070244 , 18.07741723,  7.96914521, 12.35474266,
       18.76647297, 12.20141506,  8.79024424, 15.07839187, 12.5143866 ,
       15.58803867,  9.12198992, 11.73285092, 19.13536128,  8.87135955,
       14.71528738, 15.72214431,  9.63844169, 12.83536004,  9.82708007,
       12.02812807, 11.18463076, 11.55753676, 11.96583788, 12.96584937,
        9.81076629, 10.8074428 , 11.78069743,  9.02290603, 12.98379529,
       13.70886845,  8.33139453, 11.66278748, 10.61821796,  5.51091075,
        9.43945419, 11.10483796, 16.21379308, 15.73812692,  9.07104242,
       13.87280325,  0.52155643, 15.90638919, 14.38922905, 11.97730024,
        7.05447789, 18.58542536,  9.41108376, 13.2076544 ,  8.63315957])
```
> *Comparing the predicted values to the **output testing dataset** (y_test)*
```python
636    19
220    12
594    18
429    11
72     11
       ..
514     7
374    17
444    11
244    12
601    10
```

## Evaluating the Model:
### Mean Squared Error
```python
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
```
> *Calculated **Mean Squared Error***
```python
Mean Squared Error (MSE): 1.48
```
### Mean Absolute Error
```python
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
```
> *Calculated **Mean Absolute Error***
```python
Mean Absolute Error (MAE): 0.77
```

### R-squared
```python
# Calculate R-squared (r2)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.2f}")
```
> *Calculated **R-squared***
```python
R-squared (R²): 0.85
```
## Interpretation:
```python
# Coefficients of the model
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
# Print the coefficients
coefficients.head(15)
```
> *A positive coefficient means that as the feature increases, the target variable increases.*

> *A negative coefficient means that as the feature increases, the target variable decreases.*

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
      <td>0.007372</td>
    </tr>
    <tr>
      <th>Medu</th>
      <td>-0.166254</td>
    </tr>
    <tr>
      <th>Fedu</th>
      <td>0.041540</td>
    </tr>
    <tr>
      <th>traveltime</th>
      <td>0.083662</td>
    </tr>
    <tr>
      <th>studytime</th>
      <td>0.050082</td>
    </tr>
    <tr>
      <th>failures</th>
      <td>-0.124303</td>
    </tr>
    <tr>
      <th>famrel</th>
      <td>-0.040140</td>
    </tr>
    <tr>
      <th>freetime</th>
      <td>-0.113908</td>
    </tr>
    <tr>
      <th>goout</th>
      <td>0.020842</td>
    </tr>
    <tr>
      <th>Dalc</th>
      <td>-0.067851</td>
    </tr>
    <tr>
      <th>Walc</th>
      <td>0.035156</td>
    </tr>
    <tr>
      <th>health</th>
      <td>-0.079442</td>
    </tr>
    <tr>
      <th>absences</th>
      <td>0.053971</td>
    </tr>
    <tr>
      <th>G1</th>
      <td>0.460262</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>2.471735</td>
    </tr>
  </tbody>
</table>
</div>

##  Comparison of the actual vs. predicted values from the machine learning model.
```python
# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', s=70)  # Add alpha for transparency

# Add a diagonal line representing perfect predictions
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

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

![image](https://github.com/user-attachments/assets/c9437e6c-4205-4a47-a896-d7daf151d043)

