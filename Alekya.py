import pandas as pd

# Set the file path
file_path = "API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, skiprows=4)

# Display the DataFrame
df.head()


import pandas as pd

# Set the file path
file_path = "API_NV.IND.MANF.ZS_DS2_en_csv_v2_5358349.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, skiprows=4)

# Display the DataFrame
df.head()


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv", skiprows=4)

# Select relevant columns and drop missing values
df = df[['Country Name', '2021']].dropna()

# Rename columns
df.columns = ['Country', 'GDP_per_capita']

# Set index to country name
df.set_index('Country', inplace=True)

# Remove rows with invalid GDP values (negative or zero)
df = df[df['GDP_per_capita'] > 0]

# Log-transform the GDP values to reduce skewness
df['GDP_per_capita'] = np.log(df['GDP_per_capita'])

# Standardize the data using z-score normalization
df = (df - df.mean()) / df.std()

# Save the cleaned dataset to a new file
df.to_csv('clustering_dataset.csv', index=True)


import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
df = pd.read_csv('fitting_data.csv')

# Filter the data for the selected countries
countries = ['China', 'Indonesia', 'Tanzania', 'United Kingdom', 'Nepal', 'Nicaragua']
df_countries = df[df['Country'].isin(countries)]

# Create a dictionary to hold the country names and their respective values
values = {}
for country in countries:
    df_country = df_countries[df_countries['Country'] == country]
    value = df_country['Value'].iloc[-1]  # Get the latest value
    values[country] = value

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(values.values(), labels=values.keys(), autopct='%1.1f%%')
ax.set_title('Agriculture, forestry, and fishing value added (% of GDP)')

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load cleaned dataset
df = pd.read_csv("clustering_dataset.csv", index_col="Country")

# Extract GDP per capita column and normalize
X = df['GDP_per_capita'].values.reshape(-1,1)
X_norm = (X - X.mean()) / X.std()

# Range of number of clusters to try
n_clusters_range = range(2, 11)

# Iterate over number of clusters and compute within-cluster sum of squares (WCSS)
wcss = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_norm)
    wcss.append(kmeans.inertia_)

# Plot WCSS values
plt.plot(n_clusters_range, wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Curve for Optimal Number of Clusters")
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# load cleaned dataset
df = pd.read_csv('clustering_dataset.csv')

# perform k-means clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(df[['GDP_per_capita']])
df['Cluster'] = kmeans.labels_

# plot results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue', 'orange']
for i in range(4):
    cluster_data = df[df['Cluster']==i]
    scatter = ax.scatter(cluster_data.index, cluster_data['GDP_per_capita'], 
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('GDP per capita', fontsize=14)
plt.title('K-Means Clustering Results', fontsize=16)
ax.legend(fontsize=12)

# add annotation for the cluster centers
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(1, center[0]), xytext=(6, 0), 
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()



# print countries in each cluster in a table
for i in range(4):
    print(f'Cluster {i+1}:')
    cluster_data = df[df['Cluster']==i]
    cluster_table = pd.DataFrame({'Country': cluster_data['Country'].values})
    display(cluster_table)


import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_NV.IND.MANF.ZS_DS2_en_csv_v2_5358349.csv', skiprows=4)

# Select only the necessary data for fitting analysis

df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-30:-1]]]

# Rename columns to simpler names

df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2019)]

# Melt the DataFrame to transform the columns into rows

df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value')

# Drop rows with missing values

df_cleaned = df_melted.dropna()

# Save the cleaned data to a new CSV file

df_cleaned.to_csv('fitting_data.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('fitting_data.csv')

# Filter for the selected countries
countries = ['China', 'Indonesia', 'Tanzania', 'United Kingdom', 'Nepal', 'Nicaragua']
df = df[df['Country'].isin(countries)]

# Pivot the DataFrame to have the years as columns and the values as rows
df_pivot = df.pivot(index='Country', columns='Year', values='Value')

# Plot a line graph for each country
for country in countries:
    plt.plot(df_pivot.loc[country], label=country)

# Add a legend and axis labels
plt.legend()
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from errors import err_ranges

# Load the cleaned dataset
df = pd.read_csv('fitting_data.csv')

# Select the data for a specific country and indicator
country = 'United States'
indicator = 'Manufacturing, value added (% of GDP)'
data = df[(df['Country'] == country) & (df['Indicator'] == indicator)]

# Define the function to fit
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

# Fit the function to the data using curve_fit
popt, pcov = curve_fit(quadratic, data['Year'], data['Value'])

# Get the lower and upper limits of the confidence interval
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_ranges(data['Year'], quadratic, popt, sigma)

# Make predictions for the next 10 years
future_years = np.arange(2019, 2029)
predictions = quadratic(future_years, *popt)

# Plot the data, the best fitting function, and the confidence range
plt.figure(figsize=(12,8))
plt.plot(data['Year'], data['Value'], 'o', label='Data')
plt.plot(data['Year'], quadratic(data['Year'], *popt), label='Best Fit')
plt.fill_between(data['Year'], lower, upper, alpha=0.3, label='Confidence Range')
plt.plot(future_years, predictions, label='Predictions')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Manufacturing, value added (% of GDP)')
plt.title(f'{country}: {indicator}')
plt.show()
