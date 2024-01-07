#!/usr/bin/env python
# coding: utf-8

# # HYPOTHESIS GENERATION

# In[43]:


#Here are four (04) hypothesis that we would want to test after the EDA:
#On basis of item:
#Item visibility in store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
#Product Frequency: More frequent products will have high Sales.
#On basis of store:

#City type: Stores located in urban cities should have higher sales because of the higher income levels of people there.
#Store capacity: Stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place


# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[45]:


df = pd.read_csv(r"C:\Users\Kesha Patel\Desktop\Technocolabs\9961_14084_bundle_archive\Train.csv")


# In[46]:


df #Train data


# In[47]:


df1 = pd.read_csv(r"C:\Users\Kesha Patel\Desktop\Technocolabs\9961_14084_bundle_archive\Test.csv")


# In[48]:


df1 #test data


# In[ ]:





# In[ ]:





# # Explore the data

# In[49]:


print(df.head())   # Display the first few rows
print(df.info())   # Display information about the dataset
print(df.describe())  # Display summary statistics


# In[75]:


print("\nSummary Statistics:")
print(df1.describe())


# In[76]:


df.isnull().sum()


# In[77]:


# Replace NaN values with median for each column
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())


# In[78]:


df['Item_Weight']


# In[82]:


# Replace NaN values with median for each column
df1['Item_Weight'] = df1['Item_Weight'].fillna(df1['Item_Weight'].median())


# In[83]:


df1['Item_Weight']


# In[85]:


df.outletsize = 'Outlet_Size'
df.dropna(subset=[df.outletsize], inplace=True)


# In[86]:


df.isnull().sum()


# In[87]:


df1.outletsize = 'Outlet_Size'
df1.dropna(subset=[df1.outletsize], inplace=True)


# In[88]:


df1.itemweight = 'Item_Weight'
df1.dropna(subset=[df1.itemweight], inplace=True)


# In[89]:


df1.isnull().sum()


# # # Convert categorical variable to numerical

# In[90]:


from sklearn.preprocessing import LabelEncoder

# Sample categorical data
categorical_data = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical data to numerical codes
numerical_codes = label_encoder.fit_transform(categorical_data)

# Display the mapping between categories and numerical codes
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Category Mapping:", category_mapping)

# Display the numerical codes for the original categorical data
print("Numerical Codes:", numerical_codes)


# In[91]:


df


# In[92]:


df['Item_Fat_Content'] ## low fat = 1 and regular = 2


# In[93]:


df['Item_Type'] ## 1 = Bread,2 = Breakfast, 3 = Canned,4 = Dairy,5 = Frozen foods,6 = Fruits&Veg,7 = Hard Drinks,8 = Health&Higeine,9 = Household,10 = Meat, 11 = Others, 12 = Seafood,13 = Snack Foods,14 = Soft Drinks,15 = Starchy Foods 


# In[94]:


df['Outlet_Type']  ## SuperMarket = 1,Supermarket = 2, Grocery Store = 0,Supermarket = 3


# In[95]:


# Sample categorical data
categorical_data = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical data to numerical codes
numerical_codes = label_encoder.fit_transform(categorical_data)

# Display the mapping between categories and numerical codes
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Category Mapping:", category_mapping)

# Display the numerical codes for the original categorical data
print("Numerical Codes:", numerical_codes)


# In[96]:


df1


# In[97]:


df1['Item_Fat_Content']


# In[98]:


df1['Item_Type']


# In[99]:


df1['Outlet_Type']  


# In[106]:


print("test mode, train mode\n",[df['Outlet_Size'].mode().values[0], df1['Outlet_Size'].mode().values[0]])


# In[107]:


num = df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = df.select_dtypes('object').columns.to_list()

#numeric df
BM_num =  df[num]
#categoric df
BM_cat = df[cat]

[df[category].value_counts() for category in cat[1:]]


# In[108]:


df.head()


# In[110]:


#ADDING THE NEW COLUMN OUTLET_AGE
df['Outlet_Age'], df1['Outlet_Age']= df['Outlet_Establishment_Year'].apply(lambda year: 2020 - year), df1['Outlet_Establishment_Year'].apply(lambda year: 2020 - year)

df['Outlet_Age'].head
df1['Outlet_Age'].head


# In[111]:


plt.figure(figsize=(4, 4))
sns.countplot(x='Item_Fat_Content', data=df, palette='mako')
plt.xlabel('Item_Fat_Content', fontsize=12)
plt.show()


# In[112]:


fig, axes = plt.subplots(3, 2, figsize=(12, 20))
columns = ["Item_MRP", "Item_Weight", "Item_Visibility"]
print("Visualizing Quantitative Data Spread")
for i, label in enumerate(columns):
    row = i
    ax1 = sns.histplot(df[label], ax=axes[row, 0])
    ax1.set_xlabel(label)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"df Data - {label} Frequency")
    for container in ax1.containers:
        ax1.bar_label(container, label_type="edge")
    ax2 = sns.histplot(df1[label], ax=axes[row, 1])
    ax2.set_xlabel(label)
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"df1 Data - {label} Frequency")
    for container in ax2.containers:
        ax2.bar_label(container, label_type="edge")
plt.tight_layout()
plt.show()


# # Univariate analysis

# In[100]:


# Univariate analysis for a single variable 
import seaborn as sns
column_name = 'Outlet_Establishment_Year'

# Descriptive statistics
print(df[column_name].describe())

# Histogram for distribution
plt.figure(figsize=(10, 6))
plt.hist(df[column_name], bins=20, color='Red', edgecolor='black')
plt.title(f'Univariate Analysis: {column_name}')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.show()


# In[101]:


# Univariate analysis for a single variable 
import seaborn as sns
column_name = 'Outlet_Type'

# Descriptive statistics
print(df1[column_name].describe())

# Histogram for distribution
plt.figure(figsize=(10, 6))
plt.hist(df1[column_name], bins=20, color='indigo', edgecolor='black')
plt.title(f'Univariate Analysis: {column_name}')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.show()


# In[102]:


# Bivariate analysis for two variables (e.g., 'column1' and 'column2')
column1 = 'Item_MRP'
column2 = 'Item_Outlet_Sales'

# Scatter plot for two numerical variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[column1], y=df[column2])
plt.title(f'Bivariate Analysis: {column1} vs {column2}')
plt.xlabel(column1)
plt.ylabel(column2)
plt.show()

# Pair plot for multiple numerical variables
sns.pairplot(df[['Item_MRP', 'Item_Outlet_Sales']])
plt.suptitle('Pair Plot for Bivariate Analysis', y=1.02)
plt.show()


# In[103]:


# Bivariate scatter plot
column1 = 'Item_MRP'
column2 = 'Item_Outlet_Sales'

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[column1], y=df[column2], color='skyblue')
plt.title(f'Bivariate Analysis: {column1} vs {column2}')
plt.xlabel(column1)
plt.ylabel(column2)
plt.show()

# Pair plot for multiple numerical variables
sns.set(style="darkgrid", color_codes=True)
pair_plot = sns.pairplot(df[['Item_MRP', 'Item_Outlet_Sales']], palette="husl")
pair_plot.fig.suptitle('Pair Plot for Bivariate Analysis', y=1.02)
plt.show()


# # ##Feature Engineering

# In[113]:


##Encoding Categorical Variables:


# In[116]:


from sklearn.preprocessing import LabelEncoder

# Assuming 'train' is your DataFrame
Label = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']

le = LabelEncoder()

for i in Label:
    df[i] = le.fit_transform(df[i])
    df1[i] = le.fit_transform(df1[i])

df.head()


# In[ ]:


#DROPING IRRELEVANT COLUMNS


# In[117]:


df = df.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Type'],axis=1)
df1 = df1.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Type'],axis=1)


# In[122]:


df


# In[121]:


df1.head()


# In[124]:


#Applying ML algo


# In[125]:


from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection  import cross_val_score as CVS


# In[126]:


y = df['Item_Outlet_Sales']
X = df.drop('Item_Outlet_Sales', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)


# In[127]:


def cross_val(model_name,model,X,y,cv):

    scores = CVS(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model_name} score: {round(scores.mean(),4)}')


# In[128]:


#LINEAR REGRESSION


# In[129]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# In[130]:


##SPLITTING TRAINING AND TESTING


# In[131]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


# In[134]:


## Applying LinearRegression


# In[135]:


LR = LinearRegression()


# In[136]:


##MODEL TRAINING


# In[137]:


LR.fit(X_train, y_train)


# In[138]:


y_predict = LR.predict(X_test)


# In[139]:


LR_MAE = mean_absolute_error(y_test, y_predict)
LR_MSE = mean_squared_error(y_test, y_predict)
LR_R2 = r2_score(y_test, y_predict)
LR_CV = round(cross_val_score(LR, X, y, cv=5).mean(), 4)


# In[140]:


print(f" Mean Absolute Error: {round(LR_MAE, 2)}")
print(f" Mean Squared Error: {round(LR_MSE, 2)}")
print(f" R^2 Score: {round(LR_R2, 4)}")
print(f" Cross-Validation Score: {LR_CV}")


# In[141]:


cv_scores = cross_val_score(LR, X, y, cv=5)
average_cv_score = cv_scores.mean()

print(f"\n Cross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f" Fold {i}: {round(score, 2)}")
print(f"\n Average Cross-Validation Score: {round(average_cv_score, 4)}")


# In[142]:


##LASSO REGRESSION


# In[143]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# In[144]:


# Create a Lasso model with a specified alpha (regularization strength)
LS = Lasso(alpha=0.05)


# In[145]:


# Fit the model to the training data
LS.fit(X_train, y_train)


# In[146]:


# Make predictions on the test set
y_predict = LS.predict(X_test)


# In[147]:


# Calculate evaluation metrics
LS_MAE = round(mean_absolute_error(y_test, y_predict), 2)
LS_MSE = round(mean_squared_error(y_test, y_predict), 2)
LS_R2 = round(r2_score(y_test, y_predict), 4)


# In[148]:


# Print the evaluation metrics
print(f"Mean Absolute Error: {LS_MAE}")
print(f"Mean Squared Error: {LS_MSE}")
print(f"R^2 Score: {LS_R2}")


# In[149]:


# Cross-validation
LS_CS = round(cross_val_score(LS, X, y, cv=5).mean(), 4)
print(f"Cross-validated Score: {LS_CS}")


# In[150]:


##RIDGE REGRESSION


# In[151]:


from sklearn.linear_model import Ridge

# Create a Ridge model with a specified alpha (regularization strength)
RD = Ridge(alpha=0.1)

# Fit the Ridge model to the training data
RD.fit(X_train, y_train)

# Make predictions on the test set
y_predict_ridge = RD.predict(X_test)

# Calculate evaluation metrics for Ridge regression
RD_MAE = round(mean_absolute_error(y_test, y_predict_ridge), 2)
RD_MSE = round(mean_squared_error(y_test, y_predict_ridge), 2)
RD_R2 = round(r2_score(y_test, y_predict_ridge), 4)

# Print the evaluation metrics for Ridge regression
print("\nRidge Regression Metrics:")
print(f"Mean Absolute Error: {RD_MAE}")
print(f"Mean Squared Error: {RD_MSE}")
print(f"R^2 Score: {RD_R2}")

# Cross-validation for Ridge regression
RD_CS = round(cross_val_score(RD, X, y, cv=5).mean(), 4)
print(f"Cross-validated Score (Ridge): {RD_CS}")


# In[152]:


##RANDOM FOREST


# In[153]:


import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define functions
def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def cross_val(model_name, model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i, 2))
    print(f'Average {model_name} score: {round(scores.mean(), 4)}')

# Assuming you have your data loaded into X and y
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
RFR = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4, random_state=101)

# Fit the model to the training data
RFR.fit(X_train, y_train)

# Make predictions on the test set
y_predict = RFR.predict(X_test)

# Calculate evaluation metrics
RFR_MAE = round(MAE(y_test, y_predict), 2)
RFR_MSE = round(MSE(y_test, y_predict), 2)
RFR_R_2 = round(R2(y_test, y_predict), 4)
RFR_CS = round(cross_val_score(RFR, X, y, cv=5).mean(), 4)

# Print the evaluation metrics
print(f"Random Forest Regressor Metrics:")
print(f"Mean Absolute Error: {RFR_MAE}")
print(f"Mean Squared Error: {RFR_MSE}")
print(f"R^2 Score: {RFR_R_2}")
print(f"Cross-validated Score: {RFR_CS}")


# In[154]:


##ACTUAL AND PREDICTED VALUES


# In[155]:


Random_Forest_Regressor=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest_Regressor.to_csv("Random Forest Regressor.csv")
Random_Forest_Regressor


# In[156]:


##ADABOOST REGRESSOR


# In[157]:


from sklearn.ensemble import AdaBoostRegressor

# Create an AdaBoostRegressor with specified hyperparameters
ABR = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=101)

# Fit the model to the training data
ABR.fit(X_train, y_train)

# Make predictions on the test set
y_predict_abr = ABR.predict(X_test)

# Calculate evaluation metrics for AdaBoost Regressor
ABR_MAE = round(mean_absolute_error(y_test, y_predict_abr), 2)
ABR_MSE = round(mean_squared_error(y_test, y_predict_abr), 2)
ABR_R2 = round(r2_score(y_test, y_predict_abr), 4)
ABR_CS = round(cross_val_score(ABR, X, y, cv=5).mean(), 4)

# Print the evaluation metrics for AdaBoost Regressor
print("\nAdaBoost Regressor Metrics:")
print(f"Mean Absolute Error: {ABR_MAE}")
print(f"Mean Squared Error: {ABR_MSE}")
print(f"R^2 Score: {ABR_R2}")
print(f"Cross-validated Score (AdaBoost): {ABR_CS}")

# Create a DataFrame to store actual and predicted values for AdaBoost Regressor
ABR_Regressor = pd.DataFrame({'y_test': y_test, 'prediction': y_predict_abr})

# Save the DataFrame to a CSV file
ABR_Regressor.to_csv("AdaBoost Regressor.csv")

# Display the DataFrame
ABR_Regressor


# In[158]:


##XGBOOST REGRESSOR


# In[159]:


import xgboost as xgb

# Create an XGBoost Regressor with specified hyperparameters
XGBR = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=101)

# Fit the model to the training data
XGBR.fit(X_train, y_train)

# Make predictions on the test set
y_predict_xgbr = XGBR.predict(X_test)

# Calculate evaluation metrics for XGBoost Regressor
XGBR_MAE = round(mean_absolute_error(y_test, y_predict_xgbr), 2)
XGBR_MSE = round(mean_squared_error(y_test, y_predict_xgbr), 2)
XGBR_R2 = round(r2_score(y_test, y_predict_xgbr), 4)
XGBR_CS = round(cross_val_score(XGBR, X, y, cv=5).mean(), 4)

# Print the evaluation metrics for XGBoost Regressor
print("\nXGBoost Regressor Metrics:")
print(f"Mean Absolute Error: {XGBR_MAE}")
print(f"Mean Squared Error: {XGBR_MSE}")
print(f"R^2 Score: {XGBR_R2}")
print(f"Cross-validated Score (XGBoost): {XGBR_CS}")

# Create a DataFrame to store actual and predicted values for XGBoost Regressor
XGBR_Regressor = pd.DataFrame({'y_test': y_test, 'prediction': y_predict_xgbr})

# Save the DataFrame to a CSV file
XGBR_Regressor.to_csv("XGBoost Regressor.csv")

# Display the DataFrame
XGBR_Regressor


# In[160]:


##GRADIENT BOOSTING REGRESSOR


# In[161]:


from sklearn.ensemble import GradientBoostingRegressor

# Create a GradientBoostingRegressor with specified hyperparameters
GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=101)

# Fit the model to the training data
GBR.fit(X_train, y_train)

# Make predictions on the test set
y_predict_gbr = GBR.predict(X_test)

# Calculate evaluation metrics for Gradient Boosting Regressor
GBR_MAE = round(mean_absolute_error(y_test, y_predict_gbr), 2)
GBR_MSE = round(mean_squared_error(y_test, y_predict_gbr), 2)
GBR_R2 = round(r2_score(y_test, y_predict_gbr), 4)
GBR_CS = round(cross_val_score(GBR, X, y, cv=5).mean(), 4)

# Print the evaluation metrics for Gradient Boosting Regressor
print("\nGradient Boosting Regressor Metrics:")
print(f"Mean Absolute Error: {GBR_MAE}")
print(f"Mean Squared Error: {GBR_MSE}")
print(f"R^2 Score: {GBR_R2}")
print(f"Cross-validated Score (Gradient Boosting): {GBR_CS}")

# Create a DataFrame to store actual and predicted values for Gradient Boosting Regressor
GBR_Regressor = pd.DataFrame({'y_test': y_test, 'prediction': y_predict_gbr})

# Save the DataFrame to a CSV file
GBR_Regressor.to_csv("Gradient Boosting Regressor.csv")

# Display the DataFrame
GBR_Regressor


# In[162]:


##SGD REGRESSOR


# In[163]:


from sklearn.linear_model import SGDRegressor

# Create an SGDRegressor with specified hyperparameters
SGDR = SGDRegressor(max_iter=1000, tol=1e-3, random_state=101)

# Fit the model to the training data
SGDR.fit(X_train, y_train)

# Make predictions on the test set
y_predict_sgdr = SGDR.predict(X_test)

# Calculate evaluation metrics for SGD Regressor
SGDR_MAE = round(mean_absolute_error(y_test, y_predict_sgdr), 2)
SGDR_MSE = round(mean_squared_error(y_test, y_predict_sgdr), 2)
SGDR_R2 = round(r2_score(y_test, y_predict_sgdr), 4)
SGDR_CS = round(cross_val_score(SGDR, X, y, cv=5).mean(), 4)

# Print the evaluation metrics for SGD Regressor
print("\nSGD Regressor Metrics:")
print(f"Mean Absolute Error: {SGDR_MAE}")
print(f"Mean Squared Error: {SGDR_MSE}")
print(f"R^2 Score: {SGDR_R2}")
print(f"Cross-validated Score (SGD): {SGDR_CS}")

# Create a DataFrame to store actual and predicted values for SGD Regressor
SGDR_Regressor = pd.DataFrame({'y_test': y_test, 'prediction': y_predict_sgdr})

# Save the DataFrame to a CSV file
SGDR_Regressor.to_csv("SGD Regressor.csv")

# Display the DataFrame
SGDR_Regressor


# In[164]:


##HYPERTUNING


# In[165]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

# Assuming you have your data loaded into X and y
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipelines for standard scaling and robust scaling
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# Define the models
xgb_model = XGBRegressor()
gbr_model = GradientBoostingRegressor()

# Create pipelines with hyperparameter tuning
xgb_pipeline = Pipeline([
    ('scaler', standard_scaler),
    ('xgb', XGBRegressor())
])

gbr_pipeline = Pipeline([
    ('scaler', robust_scaler),
    ('gbr', GradientBoostingRegressor())
])

# Define hyperparameter grids for randomized search
xgb_param_dist = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': [int(x) for x in range(50, 200, 10)],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

gbr_param_dist = {
    'n_estimators': [int(x) for x in range(50, 200, 10)],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Randomized search for XGBoost
xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
xgb_random_search.fit(X_train, y_train)

# Randomized search for Gradient Boosting Regressor
gbr_random_search = RandomizedSearchCV(estimator=gbr_model, param_distributions=gbr_param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
gbr_random_search.fit(X_train, y_train)

# Get the best models
best_xgb_model = xgb_random_search.best_estimator_
best_gbr_model = gbr_random_search.best_estimator_



# In[166]:


##FINAL PREDICTIONS ON TESTDATA


# In[167]:


# Make final predictions on the test dataset
xgb_test_preds = best_xgb_model.predict(X_test)
gbr_test_preds = best_gbr_model.predict(X_test)

# Calculate evaluation metrics on the test dataset
xgb_test_mse = mean_squared_error(y_test, xgb_test_preds)
gbr_test_mse = mean_squared_error(y_test, gbr_test_preds)

# Print evaluation metrics
print(f"XGBoost Test MSE: {xgb_test_mse}")
print(f"Gradient Boosting Regressor Test MSE: {gbr_test_mse}")


# # SAVING THE MODEL

# In[168]:


# Save the final models
joblib.dump(best_xgb_model, 'xgb_final_model.joblib')
joblib.dump(best_gbr_model, 'gbr_final_model.joblib')


# In[ ]:




