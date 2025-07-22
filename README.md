## PREDICTIVE ANALYTICS APPROACH TO UNDERSTANDING CUSTOMER RETENTION

##### TABLE OF CONTENTS
* [Business Understanding](#1-business-understanding)
    * [Stakeholders](#11-stakeholders)
    * [Objectives](#2-objectives)
* [Data Understanding](#3-data-understanding)
    * [Data Source](#31-source-of-data)
    * [Loading Data](#31-loading-data-set)
    * [Data Features](#32-data-features)
* [Data Preparation](#40-data-preparation)
* [Modeling](#50-modeling)
    * [Logistic Regrassion](#51-logistic-regression)
    * [Decision Tree](#52-decision-tree)
    * [Random Forest](#53-random-forest)
* [Model Evaluation](#60-model-evaluation)
    * [Model Selection](#61-final-model-selection)
    * [Feature Importance](#62-feature-importance)
* [Recommendations](#60-recommendations)
* [Non Technical Presentation](#70-non-technical-presentation)
* [License](#80-license)

##### 1.0 BUSINESS UNDERSTANDING

Customers leaving a business is a challenge concern as it affects the profits being made and cuts growth opportunities. This is an even greater problem in the telecommunications sector as it is always much cheaper to maintain existing users than to recruit new ones. Knowing the reasons behind customer churn and being able to make accurate predictions on how many customers are likely to leave enables the firm to take appropriate proactive retention measures. The goal of this project is to examine the customers’ data to determine the characteristics that surround churn in order to create a predictive model to preemptively manage customer attrition.

Most naturally, the audience of this project will be the telecom business itself, interested in reducing how much money is lost because of customers who don't stick around very long. The question then would be: are there any predictable patterns do determine this major elephant challenge faced by telecommunication sector?

#### 1.1 Stakeholders

This project will be useful to the company as various departments will be able to make meaningful decisions :
* **Marketing Department:** It will be useful in performing customer segmentation.
* **Sales Department:** Contributes with information about new clients and their expectations, why clients disengage, and aids in formulating a more retention aligned sales approach.
* **Customer Service Department:** Helps understand common pain points, service gaps, and general client satisfaction. The customer's perception alongside complaints is instrumental.
* **Product Development/Management Team:** Their role is to ascertain whether some attributes, services, or products lead to churn. Their perspectives assist in enhancing and developing products to address customer concerns.
* **Chief executives/Senior Management:** They need an overview of churn from a business perspective such as revenue loss as well as customer lifetime value. The senior management will be able to define customer retention KPIs(Key performance Indicators) based on business object

#### 1.2 Objectives

The primary objectives of this project were to:

* Determine which features or behaviors are most indicative of customer churn.
* Develop a robust predictive model that can accurately identify customers at high risk of churning.
* Provide actionable insights that can help the business develop effective strategies to reduce churn, such as personalized offers, improved customer service, or proactive outreach/marketing.
* Advise the company on resource allocation by targeting retention efforts towards high-risk customers

##### 2.0 DATA UNDERSTANDING

#### 2.1 Source of Data
This data has been sourced from Kaggle and represents data from SyriaTel, a telecommunications company, the data takes the form of binary classification.

#### 2.2 Loading Data 
This was achieved by use of important libraries and classes in python.
The libraries shared below was used to load the data into a dataframe and read through the output on the featurs of both the target and predictor variables.
* Pandas
* Numpy
* Matplotlib for Visualizations
* Classes in Scikit-Learn for Modeling

* I**mporting the neccessary libraries**
The python built in libraries below are imported to allow data wrangling and data understanding.

```
# Import the necessary packages
# Data handling and numerical operations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For displaying plots nicely in Jupyter notebooks
%matplotlib inline
# Machine learning tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score,auc,RocCurveDisplay


# Ignore warnings (optional for cleaner output)
import warnings
warnings.filterwarnings('ignore')
```

#### 2.3 Data features

The code below give the general information about the data that we are working with, this will help in getting to know whether there are features with misssing values or not. From the output there are no missing values but some features are categorical which need to be converted to numerical for purposes of building a machine learnig model to predict whether the customers will churn or not.

The dataset contains various features related to customer behavior and account information, along with a churn column indicating whether a customer has churned.

The dataset has:

* 3333 entries with 21 features
* The churn which is our target variable is a boolean indicatiing whether the customer churned or not.
* 16 Numerical features and 4 objects

`matplotlib` and `seaborn` were extensively used to visualize trends, distribution,correlations of key features across the datasets. Key visualizations included:

* **Understanding the distribution of each feature**
``` 
# Set the style
sns.set(style="whitegrid")

# Plot histograms for numerical columns
tele_data.hist(bins=30, figsize=(15, 12), edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()
```
![Understanding the Distribution of Each Feature](./Images/Distribution%20of%20numerical%20Features.png)

* Several features show bell-shaped, symmetric distributions, suggesting they are normally distributed
* The plot assumed area code to be numerical whie it is not.
* From this distribution this project will begin its modeling from logistic regression with is good with linear relatioships and classification.

* **Analyzing Correlation Between Features and the Churn Variable**
```# Compute correlation matrix
correlation = tele_data.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Numerical Features")
plt.show()
```
![Correlation Between Features and Churn variable](./Images/Correlation%20Matrix.png)

* Positive correlation — more service calls leads higher churn. Likely due to dissatisfaction.
* Customers using more daytime minutes may have higher usage, leading to higher churn.
* Total international charge is slightly positive, could indicate higher-cost international users.
* Area code is this case is categorical which implies and therefore will be converted to object and not interger for better predictions.
* Total day minutes is perfectly correlated with total day minutes, as it’s likely derived from it.
* Other features that exhibit multicollinearity are, Total international charge and total international minutes,total evening charge and total evening minutes, total night charge and total night minutes.
* The features that exhibit perfect muilticolleniarity can be dropped from modeling the prdictive model.
I this project i worked with all the features except phone number because it is an identifier and not relevant for modeling.

* **Exploring Categorical Features vs. Churn**
```# Grouped churn rates
for col in ['international plan', 'voice mail plan']:
    churn_rate = tele_data.groupby(col)['churn'].mean()
    print(f"\nChurn Rate by {col}:\n", churn_rate)

    churn_rate.plot(kind='bar', color='skyblue')
    plt.ylabel('Churn Rate')
    plt.title(f"Churn Rate by {col}")
    plt.show()
 ```
 ![International plan vs Churn](./Images/Churn%20Rate%20by%20International%20Plan.png)
 ![VoiceMail plan vs Churn](./Images/Churn%20Rate%20by%20Voice%20Mail%20plan.png)
 
 
* Customers who have an international calling plan churn at a significantly higher rate than those who don't — almost 4 times more likely.It is therefore a strong categorical predictor of churn
* This is therefore an important feature in making predictions of customers who churned.
* Voice mail plan on the other hand has a lower correlation with churns, meaning it is less important but it can still be included among the the features. In this case it was not dropped.

#### 3.0 DATA PREPARATION
From the data information given, there are no data features with missing values.
The steps below were taken to prepare the data for modeling:
* There are no missing values in this dataset.
* Converted churn (target variable) to numerical (0 or 1).
* Converted 'international plan' and 'voice mail plan' from 'yes'/'no' to numerical (1 or 0).
* Dropped phone number because it is an identifier and not relevant for modeling
* Applied One-Hot Encoding to categorical features: state and area code
* Separate features (X) the predictor variable and (y) the target variable.
* Split the data into training and testing sets.
* Feature Scaling(standardization) to the numerical features to have a mean of 1 and standard deviation of 0

#### 4.0 MODELING
In this predictive modeling ,logistic Regression is used as the baseline model since most of the features demostrated normality and logistic regression can be used for both linear relashionship and classification. Logistic Regression is a statistical model that, in its basic form, uses a logistic function to model a binary dependent(Target) variable, allowing it to estimate the probability of a binary outcome (in our case, churn or no churn). 

The other models build from logistic regression subsequently are Decision tree and Random Forest.

Based on the model metric performance the best robust model was choosen for further features importance analysis and recommendations made based on this best predictive model.

#### 4.1 Logistic Regression
``` # Model 1: Logistic Regression
# Define the Logistic Regression model with 'liblinear' solver which supports L1 and L2 penalties.
log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)

# Define the parameter grid for regularization tuning (C is inverse of regularization strength)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Stratified K-Fold for cross-validation to maintain class distribution in folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Setting up GridSearchCV to find the best regularization parameters using ROC AUC as the scoring metric
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=1, n_jobs=-1)

# Fit GridSearchCV to the training data to find the best model
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score from cross-validation: {grid_search.best_score_:.4f}\n")

# Get the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# --- Evaluation of the Best Model on the Test Set ---
print("Evaluation of the Best Logistic Regression Model \n")

#  Predictions and probabilities for the positive class (churn)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation Matrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nInterpretation of Confusion Matrix:")
print(f"True Negatives (TN): {conf_matrix[0, 0]} (Correctly predicted non-churners)")
print(f"False Positives (FP): {conf_matrix[0, 1]} (Incorrectly predicted churners - Type I error)")
print(f"False Negatives (FN): {conf_matrix[1, 0]} (Incorrectly predicted non-churners - Type II error)")
print(f"True Positives (TP): {conf_matrix[1, 1]} (Correctly predicted churners)")

# Confusion Matrix 

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
print("Confusion Matrix plot generated successfully.")

# ROC Curve 
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator(best_model, X_test, y_test, name='Logistic Regression', ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("ROC Curve generated successfully.")
```
![Confusion Matrix](./Images/CM-LR.png)
![ROC-AUC Curve](./Images/ROC-AUC%20LR.png)

##### Logistic Regression Model Performance
* Has demostrated good descriminative power through ROC-AUC of 0.82
* It's ability to predict churn customers is poor. The model only identified approximately 23%(Recall/Sensitivity) of all actual churners. This is a significant limitation, as a large portion of customers who will churn are being missed by the model.
* The F1-Score(0.2781), which balances precision and recall, is relatively low. This reflects the poor recall performance

The model is too conservative in its predictions, leading to many missed opportunities for proactive retention and therefore tried a more complex model (Decision tree) for comparison.

#### 4.2 Decision Tree 
```# Model 2-Decision tree
# Decision Tree Classifier with Cross-Validation and Hyperparameter Tuning
print("Modeling: Decision Tree Classifier\n")

dt_classifier = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search_dt = GridSearchCV(estimator=dt_classifier, param_grid=param_grid_dt, cv=cv, scoring='roc_auc', verbose=0, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

best_dt_model = grid_search_dt.best_estimator_

print("Decision Tree Model Training Complete.\n")

# Evaluation and Visualization 
y_pred_dt = best_dt_model.predict(X_test)
y_pred_proba_dt = best_dt_model.predict_proba(X_test)[:, 1]
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC-AUC Curve
plt.figure(figsize=(8, 6))
roc_display_dt = RocCurveDisplay.from_estimator(best_dt_model, X_test, y_test, name='Decision Tree', ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.title('ROC Curve for Decision Tree Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```
![Confusion Matrix](./Images/CM-DT.png)
![ROC-AUC Curve](./Images/ROC-AUC%20DT.png)

* Decision tree model has better recall value that Logistic regression hence better in prediction.
* The F1 score has greating improved compared to our baseline model.
* The area under the curve has also slightly improved.

##### 4.3 Random Forest
```# Model 3- Random Forest
#  Random Forest Classifier with Cross-Validation and Hyperparameter Tuning 
print(" Modeling: Random Forest Classifier with Cross-Validation and Hyperparameter Tuning\n")

# Define the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100, 150], # Number of trees in the forest
    'max_depth': [5, 10, None], # Max depth of the tree
    'min_samples_split': [2, 5], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2], # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy'] # Function to measure the quality of a split
}

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for hyperparameter tuning
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=cv, scoring='roc_auc', verbose=1, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters found for Random Forest: {grid_search_rf.best_params_}")
print(f"Best ROC AUC score from Random Forest cross-validation: {grid_search_rf.best_score_:.4f}\n")

# Get the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_

# Evaluation of the Random Forest Model on the Test Set 
print("Evaluation of the Best Random Forest Model\n")

# Predictions and Probabilities
y_pred_rf = best_rf_model.predict(X_test)
y_pred_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Accuracy (Random Forest): {accuracy_rf:.4f}")
print(f"Precision (Random Forest): {precision_rf:.4f}")
print(f"Recall (Random Forest): {recall_rf:.4f}")
print(f"F1-Score (Random Forest): {f1_rf:.4f}")
print(f"ROC AUC Score (Random Forest): {roc_auc_rf:.4f}")
print("\nConfusion Matrix (Random Forest):")
print(conf_matrix_rf)

print("\nInterpretation of Confusion Matrix (Random Forest):")
print(f"True Negatives (TN): {conf_matrix_rf[0, 0]} (Correctly predicted non-churners)")
print(f"False Positives (FP): {conf_matrix_rf[0, 1]} (Incorrectly predicted churners - Type I error)")
print(f"False Negatives (FN): {conf_matrix_rf[1, 0]} (Incorrectly predicted non-churners - Type II error)")
print(f"True Positives (TP): {conf_matrix_rf[1, 1]} (Correctly predicted churners)")

# --- Visualization of Confusion Matrix (Random Forest) ---
print("\n Visualizing Confusion Matrix (Random Forest) ")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve Plotting (Random Forest) 
plt.figure(figsize=(8, 6))
roc_display_rf = RocCurveDisplay.from_estimator(best_rf_model, X_test, y_test, name='Random Forest', ax=plt.gca())
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.title('ROC Curve for Random Forest Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```
![Confusion Matrix](./Images/CM-RF.png)
![ROC-AUC Curve](./Images/ROC-AUC%20RF.png)

* The confusion matrix for the Random Forest classifier shows a strong performance with high true positives and true negatives, and very few false positives and false negatives.
* The ROC Curve for the Random Forest classifier stays close to the top-left corner, and the high ROC AUC score of 0.9029 indicates that the model has excellent discriminatory power.

#### 5.0 MODEL EVALUATION
The code shared below will combine the overall metrics for the three models, visualize the metrics and plot a combined ROC-AUC curve for comparison anf choice of the final model for analysis and recommendation.

```# DataFrame for metrics
metrics_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy, accuracy_dt, accuracy_rf],
    'Precision': [precision, precision_dt, precision_rf],
    'Recall': [recall, recall_dt, recall_rf],
    'F1-Score': [f1, f1_dt, f1_rf],
    'ROC AUC': [roc_auc, roc_auc_dt, roc_auc_rf]
}
metrics_df = pd.DataFrame(metrics_data)
```
```# Melting the DataFrame for easier plotting with seaborn
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 7))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_melted, palette='viridis')
plt.title('Comparison of Model Performance Metrics')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.ylim(0, 1) 
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show() ```

```#  Combined ROC-AUC Curves
print("\nGenerating Combined ROC-AUC Curves Plot...")

plt.figure(figsize=(10, 8))

# ROC curve for Logistic Regression
RocCurveDisplay.from_estimator(best_model, X_test, y_test, name=f'Logistic Regression (AUC = {roc_auc:.2f})', ax=plt.gca())

# ROC curve for Decision Tree
RocCurveDisplay.from_estimator(best_dt_model, X_test, y_test, name=f'Decision Tree (AUC = {roc_auc_dt:.2f})', ax=plt.gca())

#  ROC curve for Random Forest
RocCurveDisplay.from_estimator(best_rf_model, X_test, y_test, name=f'Random Forest (AUC = {roc_auc_rf:.2f})', ax=plt.gca())

# Random classifier line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')

plt.title('Combined ROC Curves for Logistic Regression, Decision Tree, and Random Forest')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show() 
print("Combined ROC-AUC Curves Plot displayed.")
```

![Comparison Model Performance](./Images/Model%20Performance%20Comparison%20Metrics.png)
![Combined ROC-AUC Curve](./Images/Combined%20ROC-AUC.png)

#### Consolidated Model Comparison:

| Metric            | Logistic Regression | Decision Tree | Random Forest | Interpretation (Higher is Better)                                                                                                    |
| :---------------- | :------------------ | :------------ | :------------ | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Accuracy** | 0.8650              | 0.9400        | **0.9360** | Decision Tree and Random Forest show significantly higher overall accuracy compared to Logistic Regression.                                                                 |
| **Precision** | 0.6190              | 0.9126        | **0.9263** | Random Forest has the highest precision, meaning when it predicts churn, it's most likely to be correct.                          |
| **Recall** | 0.1793              | **0.6483** | 0.6069 | Decision Tree has the highest recall, making it best at identifying actual churners among all models. |
| **F1-Score** | 0.2781              | **0.7581** | 0.7333 | Decision Tree provides the best balance between precision and recall, slightly outperforming Random Forest.                     |
| **ROC AUC Score** | 0.8156              | 0.8617        | **0.9053** | Random Forest shows the strongest discriminative power, followed closely by Decision Tree.      |

---

#### Confusion Matrix - Consolidated View:

| Outcome          | Logistic Regression | Decision Tree | Random Forest | Interpretation (Desired Trend)                                                                                             |
| :--------------- | :------------------ | :------------ | :------------ | :------------------------------------------------------------------------------------------------------------------------- |
| **True Negatives (TN)** | 839                 | 844           | **848** | Random Forest correctly identifies the most non-churners.                                        |
| **False Positives (FP)** | 16                  | 11            | **7** | Random Forest has the fewest false alarms (incorrectly predicting churn), meaning fewer wasted resources.     |
| **False Negatives (FN)** | 119                 | 77            | **57** | Random Forest has significantly reduced missed churners, which is critical for proactive retention. |
| **True Positives (TP)** | 26                  | 81            | **88** | Random Forest identifies the most actual churners, showing a substantial improvement over Logistic Regression.                                 |

---

#### 5.1 Final Model Selection:

Considering all evaluated metrics and the specific objective of churn prediction (which often prioritizes identifying as many actual churners as possible while maintaining good overall performance), the **Random Forest Classifier** is the most suitable model for this task.

Random Forest has been selected as the best model for prediction because of:

1.  **Highest ROC AUC Score (0.9053):** This indicates that Random Forest has the best overall discriminatory power, meaning it's most effective at distinguishing between churners and non-churners.
2.  **Lowest False Positives (7):** Random Forest has the fewest instances where it incorrectly predicts a non-churner as a churner. This is crucial for efficient resource allocation, as it minimizes wasted efforts on retention campaigns for customers who wouldn't have churned anyway.
3.  **High True Positives (88) and Lower False Negatives (57):** While the Decision Tree has a slightly higher recall (0.6483 vs 0.6069) and F1-score (0.7581 vs 0.7333), Random Forest's ability to identify more actual churners (88 vs 81 for Decision Tree) with fewer false positives makes it slightly more robust. The difference in recall is marginal, but Random Forest's higher precision makes its positive predictions more reliable.
4.  **Overall Robustness:** Random Forest, being an ensemble method, generally provides more stable and robust predictions by averaging out the biases of individual decision trees. This often leads to better generalization on unseen data.

While the Decision Tree showed very strong performance, especially in Recall and F1-score, the Random Forest's superior ROC AUC and significantly lower False Positives (which can be costly in a business context) make it the preferred choice for deployment. The slight trade-off in Recall for Random Forest compared to Decision Tree is compensated by its higher precision and better overall discriminative power.

#### 5.2 Random Forest Model Visualization
From our selected model of Random Forest the next step was to  analyse feature importance, check churn probability distribution and compare probabilities verses actual outcome ,make conclusions on the model performance and share recommendations to the senior management.
* #### Feature Importance

```# Feature importances from the Random Forest model
feature_importances = best_rf_model.feature_importances_

# Feature names
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sorting the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Top 20 most important features
n_top_features = 20
print(f"Top {n_top_features} Most Important Features for Churn Prediction:\n")
print(importance_df.head(n_top_features))

# --- Visualization of Feature Importance ---
print("\nVisualizing Feature Importance")
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(n_top_features), palette='viridis')
plt.title(f'Top {n_top_features} Feature Importances for Random Forest Model (Customer Churn)')
plt.xlabel('Importance (Mean Decrease in Impurity)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("Feature Importance plot generated successfully.")
```

![Feature Importance](./Images/Feature%20Importance.png)

##### **Findings From Feature Importance
* Total day charge and total day minutes are far the most important features(Customers with higher daytime usage and corresponding charges are much more likely to churn.)
* The number of customer service calls is another highly influential factor. A higher number of calls could indicate customer dissatisfaction or recurring issues, leading to churn.
* Features related to international plans and usage (minutes, calls, charge) are also important, suggesting that these services or their associated costs contribute to churn.
* Similar to daytime usage, evening usage and charges also play a significant role, though slightly less impactful than daytime.

```# Export to CSV for Tableau
importance_df.to_csv("feature_importances_rf.csv", index=False)
```

The code above saves features importance dataframe to a csv file

The code cell below creates a DataFrame combining:

* Input features (X_test)

* True values (y_test)

* Predicted labels (y_pred_rf)

* Predicted churn probabilities (y_pred_proba_rf)

and saves the data into a csv file.

```# Combined predictions with test features and true labels
results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = y_pred_rf
results_df["Probability_Churn"] = y_pred_proba_rf

# Export to CSV for Tableau
results_df.to_csv("rf_predictions_results.csv", index=False)
```
* #### Churn Probability Distribution
```# Churn Probability Distribution
plt.figure(figsize=(8, 5))
sns.histplot(results_df["Probability_Churn"], bins=20, kde=True, color='orange')
plt.title("Distribution of Predicted Churn Probabilities")
plt.xlabel("Predicted Probability of Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
```
![Churn Probability Distribution](./Images/Distribution%20of%20predicted%20Churn%20Probabilities.png)

The histogram above displays the distribution of predicted churn probabilities from the Random Forest model:
* Most customers have a low predicted probability of churn (between 0.0 and 0.2), indicating the model is   confident that these customers are unlikely to churn.
* The distribution is right-skewed, suggesting that churn is relatively rare in the dataset.
* A smaller portion of customers fall within the medium to high churn probability range (above 0.4), which represents a critical segment for potential targeted retention efforts.
* The smooth line (KDE) helps to visualize the overall trend in predicted churn risk.

* #### Probability Vs Actual Outcome
```# Probability Vs Actual Outcome
plt.figure(figsize=(8, 5))
sns.boxplot(x="Actual", y="Probability_Churn", data=results_df)
plt.title("Predicted Churn Probability by Actual Class")
plt.xlabel("Actual Churn")
plt.ylabel("Predicted Churn Probability")
plt.tight_layout()
plt.show()
```

![Churn Prbability Vs Actual outcome](./Images/Predicted%20Churn%20probability%20by%20Actual%20class.png)

This boxplot compares the predicted churn probabilities between the two actual classes.
**Class 0 (Non-Churners)**
* The predicted probabilities are mostly low (centered around 0.05–0.1).
**Class 1 (Churners)**
* These customers have significantly higher predicted probabilities, with a median around 0.55–0.6
* The model is generally assigning higher risk to actual churners.

The Random Forest model effectively differentiates between churners and non-churners. Higher predicted probabilities align well with actual churners, supporting the model's use for risk scoring and targeted retention strategies.

* #### Risk Categories and Distribution of risk Levels
```# Creating risk categories
def assign_risk(prob):
    if prob >= 0.6:
        return "High Risk"
    elif prob >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

# Apply function
results_df["Churn_Risk_Level"] = results_df["Probability_Churn"].apply(assign_risk)
````
```# Distribution of risk Levels
plt.figure(figsize=(8, 5))
sns.countplot(x="Churn_Risk_Level", data=results_df, order=["Low Risk", "Medium Risk", "High Risk"], palette="Set2")
plt.title("Customer Count by Churn Risk Level")
plt.xlabel("Churn Risk Level")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
```
![Customer Count by Churn Risk](./Images/Customer%20Count%20by%20Churn%20Risk.png)

The chart displays how customers are distributed across three churn risk categories based on predicted probabilities from the Random Forest model:

***Low Risk (0.00–0.29)***
The majority of customers fall into this category. These customers are unlikely to churn and may only require standard retention strategies.

***Medium Risk (0.30–0.59)***
A smaller segment of customers is at moderate risk. These individuals should be monitored and could benefit from personalized engagement or loyalty incentives.

***High Risk (0.60–1.00)***
This is a critical group with a high likelihood of churning. They should be prioritized for immediate intervention, such as targeted offers, service improvement calls, or personalized support.

This segmentation enables the organization to prioritize customer retention efforts efficiently by focusing resources on customers with the highest predicted risk of churn.

#### 6.0 RECOMMENDATIONS
* Total day minutes,customer service calls,international plan, total evening charges are the most important features in determining the customer churn.
* Random Forest is recommended as the most robust predictive model that accurately identified customers at risk but need to be retrained periodically using latest data to mentain prediction accuracy.
* Assign customer success managers or special support channels to high-risk accounts to reduce frustration and enhance satisfaction for high risk customers
* Use churn probabilities to create automated alerts for sales or customer care teams when a customer’s risk rises.
* Develop a marketing strategy by offering exclusive benefits, discounts, or loyalty points low-risk customers. Reward them for their loyalty with personalized offers that acknowledge their high consumption.
* Track engagement signals (e.g., reduced usage, late payments) to detect early signs of churn.
* Launch urgent, personalized outreach campaigns (e.g., call center, SMS, or email) to address issues like poor service, billing complaints, or unmet expectations for high risk customers

 #### 7.0 NON TECHNICAL PRESENTATION
Check out the link to [Non Technical Presentation ](https://drive.google.com/file/d/13FaGBUY1_cSDVW9zyNU4OD_JsmadL_Gu/view?usp=sharing)
 #### 8.0 LICENSE
This project is licensed under the [GNU General Public Licence v3.0](https://github.com/MTirop/DSF-PT11-Phase3-Project/blob/master/LICENSE)
