# Porter Delivery Time Prediction

## Project Overview

This project focuses on predicting food delivery duration using machine learning and neural networks. The objective is to understand the factors influencing delivery time and build predictive models that can estimate delivery duration based on order and operational information.

## Business Problem

Accurate delivery-time prediction is important for food delivery platforms because it helps improve customer experience, delivery planning, and operational efficiency.

This project uses historical delivery data to build regression models that predict the actual delivery duration.

## Dataset Overview

The dataset contains approximately 175,777 delivery records and 14 original columns.

The features include:

- Order information: Total items, subtotal, minimum and maximum item price.
- Store information: Market ID and store primary category.
- Delivery operations: Total on-shift dashers, busy dashers, and outstanding orders.
- Delivery distance information: Estimated store-to-consumer driving duration.
- Order timing: Order creation timestamp and delivery timestamp.

## Project Workflow

### 1. Data Cleaning and Feature Engineering

- Inspected dataset dimensions, columns, and data types.
- Converted timestamp columns into datetime format.
- Created delivery-duration and order-time features.
- Extracted order hour and day of the week.
- Applied one-hot encoding to the store primary category.

### 2. Exploratory Data Analysis (EDA)

- Analyzed correlations between numerical variables.
- Used scatterplots to understand relationships between order features and delivery duration.
- Examined order volume across different days and hours.
- Analyzed average delivery duration across different order hours.

### 3. Outlier Treatment

- Applied the Interquartile Range (IQR) method to identify outliers.
- Removed records containing outliers in the selected numerical features.
- Compared visualizations before and after outlier treatment.

### 4. Machine Learning – Random Forest Regression

- Split the dataset into training and testing sets using an 80:20 ratio.
- Built a baseline Random Forest Regression model.
- Evaluated model performance using MAE, MSE, RMSE, and R².
- Analyzed feature importance to identify influential predictors.

### 5. Neural Network Regression – PyTorch

- Applied StandardScaler for feature scaling.
- Designed a feedforward neural network with two hidden layers.
- Used ReLU activation in the hidden layers and a linear output layer.
- Used Mean Squared Error (MSE) as the loss function.
- Used Adam optimizer with a learning rate of 0.001.
- Trained the model for 50 epochs.
- Monitored training and validation loss to assess overfitting.

## Model Performance Comparison

The following results were obtained on the test dataset:

| Evaluation Metric | Random Forest | Neural Network |
|---|---:|---:|
| MAE | 0.02039 | 0.00688 |
| MSE | 0.000763 | 0.000108 |
| RMSE | 0.02763 | 0.01039 |
| R² Score | 0.95935 | 0.99425 |

The Neural Network achieved lower prediction errors and a higher R² score than the baseline Random Forest model on this test split.

*Note: The target variable's unit should be verified before interpreting these error values in real-world time units.*

## Key Findings

- Estimated store-to-consumer driving duration was the most influential feature in the Random Forest model.
- Total outstanding orders and total on-shift dashers were also important predictors.
- Order volume and average delivery duration varied across different hours of the day.
- The Neural Network demonstrated stronger predictive performance than the baseline Random Forest model on the test dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Jupyter Notebook

## Repository Structure

```text
Porter-Delivery-Time-Prediction/
│
├── NN_Porter_Business_case_Regression_Scaler.ipynb
├── Porter.csv
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates an end-to-end regression workflow, starting from data preparation and exploratory analysis to classical machine learning and neural network modelling. It provided practical experience in feature engineering, outlier treatment, model evaluation, and implementing neural networks using PyTorch.
