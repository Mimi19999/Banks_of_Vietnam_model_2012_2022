# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data 
df = pd.read_excel('/Users/mimi/Documents/Tech_data.xlsx', sheet_name='Data', usecols=('Name', 'Year','Earning_asset',
                    'Net_revenue', 'NPL', 'CAR', 'Credit_insti_lending', 'Credit_insti_allowance','Cust_lending','Cust_allowance',
                    'Operating_cost', 'Total_asset', 'GDP', 'CPI'))
print(df)

# Count missing data
missing_data = df.isna().sum()
print(missing_data)

# Insert some new columns
df['NIM'] = df['Net_revenue']/df['Earning_asset']
df['Credit_risk'] = (df['Credit_insti_allowance'] + df['Cust_allowance'])/(df['Credit_insti_lending'] + df['Cust_lending'])
df['OCR'] = df['Operating_cost']/df['Total_asset']

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

# Plot 1
# Year_NIM
# df_summary_year = df[['Year', 'NIM']].groupby('Year').mean()
# x, y = list(df_summary_year.index), df_summary_year['NIM'].values
# plt.figure(figsize=(16, 8))
# plt.plot(x, y, marker='o', markersize=10, linestyle='-.', linewidth=2)
# plt.xlabel('Year', fontsize=16)
# plt.ylabel('%NIM', fontsize=16)
# plt.title("Average of NIM by year", fontsize=18)
# plt.show()

# # CPI
# df_summary_CPI= df[['Year', 'CPI']].groupby('Year').mean()
# x1, y1 = list(df_summary_CPI.index), df_summary_CPI['CPI'].values
# plt.figure(figsize=(16, 8))
# plt.plot(x1, y1, marker='o', markersize=10, linestyle='-.', linewidth=2)
# plt.xlabel('Year', fontsize=16)
# plt.ylabel('%CPI', fontsize=16)
# plt.title("CPI by year", fontsize=18)
# plt.show()

# # GDP
# df_summary_GDP= df[['Year', 'GDP']].groupby('Year').mean()
# x1, y1 = list(df_summary_GDP.index), df_summary_GDP['GDP'].values
# plt.figure(figsize=(16, 8))
# plt.plot(x1, y1, marker='o', markersize=10, linestyle='-.', linewidth=2)
# plt.xlabel('Year', fontsize=16)
# plt.ylabel('%GDP', fontsize=16)
# plt.title("GDP by year", fontsize=18)
# plt.show()

# Plot 2
# name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# # Filter the DataFrame for multiple names
# filtered_df = df[df['Name'].isin(name_filter)]

# # Configure the plot
# colors = ['red', 'blue', 'green', 'coral', 'brown']

# # Create the line graph
# fig, ax = plt.subplots()

# for i, name in enumerate(name_filter):
#     name_df = filtered_df[filtered_df['Name'] == name]
#     years = name_df['Year'].tolist()
#     nim_values = name_df['NIM'].tolist()
#     ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# # Add labels and title
# ax.set_xlabel('Year')
# ax.set_ylabel('NIM %')
# ax.set_title('NIM for top banks')

# # Add legend
# ax.legend()

# # Display the chart
# plt.tight_layout()
# plt.show()

# # Plot 3
# name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# # Filter the DataFrame for multiple names
# filtered_df = df[df['Name'].isin(name_filter)]

# # Configure the plot
# colors = ['red', 'blue', 'green', 'coral', 'brown']

# # Create the line graph
# fig, ax = plt.subplots()

# for i, name in enumerate(name_filter):
#     name_df = filtered_df[filtered_df['Name'] == name]
#     years = name_df['Year'].tolist()
#     nim_values = name_df['NPL'].tolist()
#     ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# # Add labels and title
# ax.set_xlabel('Year')
# ax.set_ylabel('NPL %')
# ax.set_title('NPL for top banks')

# # Add legend
# ax.legend()

# # Display the chart
# plt.tight_layout()
# plt.show()

# Plot 4
# name_filter = ['Tech', 'VIB', 'VP', 'Vietcom', 'Vietin']

# # Filter the DataFrame for multiple names
# filtered_df = df[df['Name'].isin(name_filter)]

# # Configure the plot
# colors = ['red', 'blue', 'green', 'coral', 'brown']

# # Create the line graph
# fig, ax = plt.subplots()

# for i, name in enumerate(name_filter):
#     name_df = filtered_df[filtered_df['Name'] == name]
#     years = name_df['Year'].tolist()
#     nim_values = name_df['CAR'].tolist()
#     ax.plot(years, nim_values, color=colors[i], label=name, marker='o', markersize=5, linewidth=2)

# # Add labels and title
# ax.set_xlabel('Year')
# ax.set_ylabel('CAR %')
# ax.set_title('CAR for top banks')

# # Add legend
# ax.legend()

# # Display the chart
# plt.tight_layout()
# plt.show()




