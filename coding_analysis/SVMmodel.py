# Import the required libraries and dependencies

# Import the required libraries and dependencies
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC

# Read the data from the online_gaming_behavior_dataset.csv file into a Pandas DataFrame
gaming_df = pd.read_csv("/Users/cw/Desktop/Working Class Projects/Project-2/Resources/online_gaming_behavior_dataset.csv")
# Review the DataFrame
gaming_df.head()

### DATA CLEANING
custom_mapping = [
    ['Easy', 'Medium', 'Hard'],  # Custom order for 'GameDifficulty'
    ['Low', 'Medium', 'High']  # Custom order for 'EngagementLevel'
]

oe_gender = OrdinalEncoder(categories=custom_mapping)
encodings = oe_gender.fit_transform(gaming_df[['GameDifficulty','EngagementLevel']])
gaming_df[['GameDifficulty','EngagementLevel']] = encodings

ohe = OneHotEncoder(sparse_output=False, dtype='int')
ohe_df = pd.DataFrame(data=ohe.fit_transform(gaming_df[['Gender','Location','GameGenre']]), columns=ohe.get_feature_names_out())
gaming_df = pd.concat([gaming_df, ohe_df], axis=1)

# Drop unnecessary columns
gaming_df = gaming_df.drop(columns=['PlayerID', 'PlayTimeHours', 'GameGenre', 'Location', 'Gender'])

# Add a new feature
gaming_df['AvgMinutesPerWeek'] = gaming_df['SessionsPerWeek'] * gaming_df['AvgSessionDurationMinutes']

### END DATA CLEANING ###

### START OF ML
# Define features and target
X = gaming_df.drop(columns='InGamePurchases')
y = gaming_df['InGamePurchases']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Initialize and fit GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict with the optimized model
y_pred_optimized = best_model.predict(X_test)

# Evaluate the optimized model
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
optimized_classification_report = classification_report(y_test, y_pred_optimized)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_optimized)

# Plotting the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Purchase', 'Purchase'], 
            yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualizing the accuracy for different hyperparameter combinations
results = pd.DataFrame(grid_search.cv_results_)
scores_matrix = results.pivot("param_C", "param_gamma", "mean_test_score")

plt.figure(figsize=(12, 8))
sns.heatmap(scores_matrix, annot=True, cmap="YlGnBu")
plt.title('GridSearchCV Results: Accuracy Scores')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()

# Output the results
print("Optimized Accuracy:", optimized_accuracy)
print("\nOptimized Classification Report:\n", optimized_classification_report)
print("\nBest Parameters Found:\n", best_params)