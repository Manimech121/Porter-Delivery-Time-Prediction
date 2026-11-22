# Porter Delivery Time Prediction

## Project Overview

This project focuses on predicting food delivery duration using machine
learning and neural networks. The objective is to understand the factors
influencing delivery time and build predictive models that can estimate
delivery duration based on order and operational information.

## Business Problem

Accurate delivery-time prediction is important for food delivery
platforms because it helps improve customer experience, delivery
planning, and operational efficiency.

This project uses historical delivery data to build regression models
that predict the actual delivery duration.

## Dataset Overview

The dataset contains approximately 175,777 delivery records and 14
original columns.

The features include:

-   Order information: Total items, subtotal, minimum and maximum item
    price.
-   Store information: Market ID and store primary category.
-   Delivery operations: Total on-shift dashers, busy dashers, and
    outstanding orders.
-   Delivery distance information: Estimated store-to-consumer driving
    duration.
-   Order timing: Order creation timestamp and delivery timestamp.

The target variable, `actual_delivery_duration`, is calculated as the
difference between the actual delivery timestamp and the order creation
timestamp, converted into minutes.

## Project Workflow

### 1. Data Cleaning and Feature Engineering

-   Inspected dataset dimensions, columns, and data types.
-   Converted timestamp columns into datetime format.
-   Created the actual delivery-duration target in minutes.
-   Extracted order hour and day of the week.
-   Applied one-hot encoding to the store primary category.

### 2. Exploratory Data Analysis (EDA)

-   Analyzed correlations between numerical variables.
-   Used scatterplots to understand relationships between order features
    and delivery duration.
-   Examined order volume across different days and hours.
-   Analyzed average delivery duration across different order hours.

### 3. Outlier Treatment

-   Applied the Interquartile Range (IQR) method to identify outliers
    across selected numerical features, including the target variable.
-   Removed 26,520 records identified as outliers (approximately 15.09%
    of the original dataset).
-   Retained 149,257 records and 87 columns for subsequent analysis and
    modelling.

### 4. Machine Learning -- Random Forest Regression

-   Split the cleaned dataset into training and testing sets using an
    80:20 ratio.
-   Built a baseline Random Forest Regression model.
-   Evaluated model performance using MAE, MSE, RMSE, and R².
-   Analyzed feature importance to identify influential predictors.

### 5. Neural Network Regression -- PyTorch

-   Applied StandardScaler for feature scaling.
-   Designed a feedforward neural network with two hidden layers (64 and
    32 neurons).
-   Used ReLU activation in the hidden layers and a linear output layer.
-   Used Mean Squared Error (MSE) as the loss function.
-   Used Adam optimizer with a learning rate of 0.001.
-   Trained the model and monitored training and validation loss to
    assess learning behaviour.

## Model Performance Comparison

The following results were obtained on the test dataset. The target
variable and error metrics are expressed in minutes (and squared minutes
for MSE).

  Evaluation Metric     Random Forest   Neural Network
  ------------------- --------------- ----------------
  MAE                          1.2247           0.4118
  MSE                          2.7555           0.3403
  RMSE                         1.6600           0.5833
  R² Score                     0.9592           0.9950

The Neural Network achieved lower prediction errors and a higher R²
score than the baseline Random Forest model on this test split. Its R²
score of approximately 0.995 indicates that it explains about 99.50% of
the variation in delivery duration in the test dataset.

## Key Findings

-   Estimated store-to-consumer driving duration was the most
    influential feature in the Random Forest model (approximately
    25.56%).
-   Total outstanding orders (approximately 24.74%) and total on-shift
    dashers (approximately 17.76%) were also important predictors.
-   These three features together accounted for approximately 68.06% of
    the Random Forest model's feature importance.
-   Order volume and average delivery duration varied across different
    hours of the day. Order volume was highest in the early morning
    hours, particularly around 1 AM to 2 AM, with another increase
    during the evening.
-   The Neural Network demonstrated stronger predictive performance than
    the baseline Random Forest model on the test dataset.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   PyTorch
-   Jupyter Notebook

## Repository Structure

``` text
Porter-Delivery-Time-Prediction/
│
├── NN_Porter_Business_case_Regression_Scaler.ipynb
├── Porter.csv
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates an end-to-end regression workflow, starting
from data preparation and exploratory analysis to classical machine
learning and neural network modelling. It provided practical experience
in feature engineering, outlier treatment, model evaluation, and
implementing neural networks using PyTorch.

The Neural Network outperformed the baseline Random Forest model on the
test dataset, achieving an MAE of approximately 0.41 minutes and an R²
score of approximately 0.995. These results should be interpreted in the
context of the selected features and the train/test split; checking that
all predictors are available at the intended prediction time remains
important before using the model in a real-world setting.
