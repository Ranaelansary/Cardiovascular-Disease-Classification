# Cardiovascular Disease Classification

## Project Overview

This project uses **Machine Learning** to predict cardiovascular disease based on patient medical and lifestyle information.

## Dataset

The dataset contains **70,000 patient records** with features such as:

* Age
* Gender
* Height
* Weight
* Blood Pressure
* Cholesterol
* Glucose
* Smoking
* Alcohol
* Physical Activity

The target variable is `cardio`:

* `0` = No cardiovascular disease
* `1` = Cardiovascular disease

## Data Preprocessing

The dataset was cleaned by:

* Removing duplicate records
* Converting age from days to years
* Removing unrealistic values
* Removing the `id` column
* Creating new features such as BMI and blood pressure features

## Machine Learning Models

We used four models:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest

The models were tuned using **RandomizedSearchCV**.

## Results

The models were evaluated using **Accuracy, Precision, Recall, and F1-Score**.

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   73.12% |    77.85% | 63.91% |   70.19% |
| KNN                 |   72.93% |    75.21% | 67.65% |   71.23% |
| Decision Tree       |   73.03% |    75.33% | 67.74% |   71.33% |
| Random Forest       |   73.49% |    76.36% | 67.31% |   71.55% |

## Evaluation

An **Accuracy Comparison** plot is generated to compare the four models.

## Tools

* Python
* Pandas
* Scikit-learn
* Matplotlib
