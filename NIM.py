# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data 
df = pd.read_excel('/Users/mimi/Documents/Tech_data.xlsx', sheet_name='Data', usecols=('Name', 'Year','Earning_asset',
                    'Net_revenue', 'NPL', 'CAR', 'Credit_insti_lending', 'Credit_insti_allowance','Cust_lending','Cust_allowance',
                    'Operating_cost', 'Total_asset', 'GDP', 'CPI'))

df[['CPI', 'GDP']] = df[['CPI', 'GDP']]/100



# Insert some new columns
df['NIM'] = df['Net_revenue']/df['Earning_asset']
print(df['NIM'][0])
df['Credit_risk'] = (df['Credit_insti_allowance'] + df['Cust_allowance'])/(df['Credit_insti_lending'] + df['Cust_lending'])
df['OCR'] = df['Operating_cost']/df['Total_asset']

# Count missing data
missing_data = df.isna().sum()
print(missing_data)

# Pivot table
pivot_year = pd.pivot_table(df,
                       columns=('Year'),
                       values=('NIM'),
                       aggfunc=np.mean)
print(pivot_year)

pivot_name = pd.pivot_table(df,
                       columns=('Name'),
                       values=('NIM'),
                       aggfunc=np.mean)
print(pivot_name)

Plot 1
# Year_NIM
df_summary_year = df[['Year', 'NIM']].groupby('Year').mean()
x, y = list(df_summary_year.index), df_summary_year['NIM'].values
plt.figure(figsize=(16, 8))
plt.plot(x, y, marker='o', markersize=10, linestyle='-.', linewidth=2)
plt.xlabel('Year', fontsize=16)
plt.ylabel('NIM', fontsize=16)
plt.title("Average of NIM by year", fontsize=18)
plt.show()

# CPI
df_summary_CPI= df[['Year', 'CPI']].groupby('Year').mean()
x1, y1 = list(df_summary_CPI.index), df_summary_CPI['CPI'].values
plt.figure(figsize=(16, 8))
plt.plot(x1, y1, marker='o', markersize=10, linestyle='-.', linewidth=2)
plt.xlabel('Year', fontsize=16)
plt.ylabel('CPI', fontsize=16)
plt.title("CPI by year", fontsize=18)
plt.show()

# GDP
df_summary_GDP= df[['Year', 'GDP']].groupby('Year').mean()
x1, y1 = list(df_summary_GDP.index), df_summary_GDP['GDP'].values
plt.figure(figsize=(16, 8))
plt.plot(x1, y1, marker='o', markersize=10, linestyle='-.', linewidth=2)
plt.xlabel('Year', fontsize=16)
plt.ylabel('GDP', fontsize=16)
plt.title("GDP by year", fontsize=18)
plt.show()

# Plot 2
name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# Filter the DataFrame for multiple names
filtered_df = df[df['Name'].isin(name_filter)]

# Configure the plot
colors = ['red', 'blue', 'green', 'coral', 'brown']

# Create the line graph
fig, ax = plt.subplots()

for i, name in enumerate(name_filter):
    name_df = filtered_df[filtered_df['Name'] == name]
    years = name_df['Year'].tolist()
    nim_values = name_df['NIM'].tolist()
    ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('NIM')
ax.set_title('NIM for top banks')

# Add legend
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()

# Plot 3
name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# Filter the DataFrame for multiple names
filtered_df = df[df['Name'].isin(name_filter)]

# Configure the plot
colors = ['red', 'blue', 'green', 'coral', 'brown']

# Create the line graph
fig, ax = plt.subplots()

for i, name in enumerate(name_filter):
    name_df = filtered_df[filtered_df['Name'] == name]
    years = name_df['Year'].tolist()
    nim_values = name_df['NPL'].tolist()
    ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('NPL')
ax.set_title('NPL for top banks')

# Add legend
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()

Plot 4
name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# Filter the DataFrame for multiple names
filtered_df = df[df['Name'].isin(name_filter)]

# Configure the plot
colors = ['red', 'blue', 'green', 'coral', 'brown']

# Create the line graph
fig, ax = plt.subplots()

for i, name in enumerate(name_filter):
    name_df = filtered_df[filtered_df['Name'] == name]
    years = name_df['Year'].tolist()
    nim_values = name_df['CAR'].tolist()
    ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# Add labels and title
ax.set_xlabel('Year')
ax.set_ylabel('CAR')
ax.set_title('CAR for top banks')

# Add legend
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()


# Independent variables distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Independent variables list
inde_cols = df[['NPL', 'CAR', 'GDP', 'CPI', 'OCR']].columns

# Grid layout
row = 2
col = 3

# Grid layout subplot
fig, axes = plt.subplots(row, col, figsize=(12, 8))

# Histogram
for i, col_name in enumerate(inde_cols):
    ax = axes[i // col, i % col]
    sns.histplot(data=df, x=col_name, kde=True, ax=ax)

# Delete an empty chart
if len(inde_cols) % col != 0:
    fig.delaxes(axes.flatten()[-1])

# Plot 5
plt.tight_layout()
plt.show()

# Correlation between independent variables
corr_matrix =  df[['NPL', 'CAR', 'GDP', 'CPI', 'OCR', 'Credit_risk', 'NIM']].corr()
print(corr_matrix)

# Choosing hyperparameters
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Split dataset
X = df[['NPL', 'CAR', 'GDP', 'CPI', 'OCR', 'Credit_insti_lending', 'Cust_lending',
        'Cust_allowance', 'Credit_insti_allowance', 'Operating_cost', 'Total_asset', 'Credit_risk' ]]

y = df['NIM']

idx = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, test_size=0.33, random_state=42)

# OCR and NIM
X_train_df = pd.DataFrame(X_train)
OCR_values = X_train_df['OCR'].values
print(OCR_values)
OCR_index = df.columns.get_loc('OCR')
print(OCR_index)
print(y_train)
plt.scatter(OCR_values, y_train)
plt.xlabel('OCR')
plt.ylabel('NIM')
plt.title('Relationship between NIM and OCR')
plt.show()

# Lasso Regression
# Mark train_data as -1 and mark test_data as 0
split_index = [-1 if i in idx_train else 0 for i in idx]
ps = PredefinedSplit(test_fold=split_index)

# Pipeline
pipeline = Pipeline([
                     ('scaler', StandardScaler()),
                     ('model', Lasso())
])
# GridSearch
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(1, 10, 1)}, # Tham số alpha từ 1->10 huấn luyện mô hình
                      cv = ps, # validation trên tập kiểm tra
                      scoring="neg_mean_squared_error", # trung bình tổng bình phương phần dư
                      verbose=3
                      )
# Tuning paramaters
search.fit(X, y)
print(search.best_estimator_)
print('Best core: ', search.best_score_)

# Training 
reg_lasso = Lasso(alpha = 1)
reg_lasso.fit(X_train, y_train)

y_test_pred = reg_lasso.predict(X_test)
mse_lasso_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error of Lasso regression:", mse_lasso_test)

y_train_pred = reg_lasso.predict(X_train)
mse_lasso_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error of Lasso regression:", mse_lasso_train)

print(reg_lasso.score(X_train, y_train))
print(reg_lasso.coef_)
print(reg_lasso.intercept_)

# Split the data into train, validation, and test sets
X = df[['NPL', 'OCR']]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the random forest regressor
rf = RandomForestRegressor(n_estimators= 10, random_state=42)

# Fit the random forest on the training set
rf.fit(X_train_scaled, y_train)

# Evaluate the model on the validation set
y_val_pred = rf.predict(X_val_scaled)
val_loss = mean_squared_error(y_val, y_val_pred)

# Evaluate the model on the test set (unseen data)
y_test_pred = rf.predict(X_test_scaled)
test_loss = mean_squared_error(y_test, y_test_pred)

# Result
r2 = r2_score(y_test, y_test_pred)
print(r2)
print("Validation Loss:", val_loss)
print("Test Loss:", test_loss)

# Print tree
from sklearn.tree import export_graphviz
import pydotplus
tree = rf.estimators_[0]
dot_data = export_graphviz(tree, out_file=None, feature_names=X.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("/Users/mimi/random_forest_tree.pdf")

