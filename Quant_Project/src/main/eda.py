import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import coint
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster



file_path = 'E:/Quant_Project/data/features_data.csv' 
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = data.sort_values(by='Date')

def plot_cumulative_returns_companies(data):
    plt.figure(figsize=(12, 6))
    for company in data['Company'].unique():
        company_data = data[data['Company'] == company]
        company_data = company_data.sort_values(by='Date')
        company_data['Daily_Return'] = company_data['Close/Last'].pct_change()
        company_data['Cumulative_Return'] = (1 + company_data['Daily_Return']).cumprod() - 1
        plt.plot(company_data['Date'], company_data['Cumulative_Return'], label=company)
    
    plt.title('Cumulative Returns Comparison Across Companies')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()


def plot_risk_return_companies(data):
    risk_return_data = []
    
    for company in data['Company'].unique():
        company_data = data[data['Company'] == company]
        company_data['Daily_Return'] = company_data['Close/Last'].pct_change()
        avg_return = company_data['Daily_Return'].mean() 
        volatility = company_data['Daily_Return'].std()  
        risk_return_data.append((company, avg_return, volatility))
    
    # Convert to DataFrame
    risk_return_df = pd.DataFrame(risk_return_data, columns=['Company', 'Avg_Return', 'Volatility'])
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(risk_return_df['Volatility'], risk_return_df['Avg_Return'], color='blue')
    
    for _, row in risk_return_df.iterrows():
        plt.text(row['Volatility'], row['Avg_Return'], row['Company'], fontsize=10)
    
    plt.title('Risk-Return Analysis Across Companies')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Average Daily Return')
    plt.grid()
    plt.show()

pivot_data = data.pivot(index='Date', columns='Company', values='Close/Last')


pivot_data = pivot_data.apply(pd.to_numeric, errors='coerce')

# Function to plot the correlation matrix
def plot_correlation_matrix(data):

    corr_matrix = data.corr()


    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Stock Prices')
    plt.show()

# Function to plot dynamic correlation using rolling windows
def plot_rolling_correlation(data, stock1, stock2, window=30):
    rolling_corr = data[stock1].rolling(window=window).corr(data[stock2])
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, rolling_corr, label=f'Rolling Correlation ({stock1} & {stock2})')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Rolling Correlation Between {stock1} and {stock2} (Window = {window})')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid()
    plt.show()

# Calculate features: returns, volatility, and average volume for each company
def calculate_features(data, original_data):
    feature_data = []
    for company in data.columns:
        daily_return = data[company].pct_change()
        avg_return = daily_return.mean()
        volatility = daily_return.std()
        avg_volume = original_data[original_data['Company'] == company]['Volume'].mean()
        feature_data.append([company, avg_return, volatility, avg_volume])
    
    feature_df = pd.DataFrame(feature_data, columns=['Company', 'Avg_Return', 'Volatility', 'Avg_Volume'])
    return feature_df.set_index('Company')

features = calculate_features(pivot_data, data)

# Standardize the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Hierarchical Clustering
def hierarchical_clustering(scaled_features, features):
    linkage_matrix = linkage(scaled_features, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=features.index.tolist(), leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Stocks')
    plt.ylabel('Distance')
    plt.show()

    # Return cluster assignments
    return fcluster(linkage_matrix, t=3, criterion='maxclust')  # Adjust 't' for the number of clusters

# K-means Clustering
def kmeans_clustering(scaled_features, features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    
    features['Cluster'] = kmeans.labels_
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(features['Volatility'], features['Avg_Return'], c=features['Cluster'], cmap='viridis', s=100)
    for i, txt in enumerate(features.index):
        plt.annotate(txt, (features['Volatility'][i], features['Avg_Return'][i]), fontsize=10)
    plt.title('K-means Clustering')
    plt.xlabel('Volatility')
    plt.ylabel('Average Return')
    plt.grid()
    plt.show()

# Cointegration Testing using the Engle-Granger Two-Step Method
def test_cointegration(data, stock1, stock2):

    series1 = data[stock1].dropna()
    series2 = data[stock2].dropna()

    # Align the two series
    combined = pd.concat([series1, series2], axis=1).dropna()
    series1, series2 = combined[stock1], combined[stock2]

    # Perform the cointegration test
    coint_t, p_value, crit_value = coint(series1, series2)

  
    print(f"Cointegration Test Results for {stock1} and {stock2}:")
    print(f"  Test Statistic: {coint_t:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Critical Values: {crit_value}")
    return p_value


def test_all_pairs(data):

    results = []
    stocks = data.columns
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            stock1, stock2 = stocks[i], stocks[j]
            p_value = test_cointegration(data, stock1, stock2)
            results.append((stock1, stock2, p_value))


    results_df = pd.DataFrame(results, columns=['Stock1', 'Stock2', 'P-Value'])
    return results_df

results_df = test_all_pairs(pivot_data)


print("\nCointegration Test Results for All Pairs:")
print(results_df)


significant_pairs = results_df[results_df['P-Value'] < 0.05]
print("\nSignificant Cointegrated Pairs (P-Value < 0.05):")
print(significant_pairs)

# Standardize the data for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Function to plot explained variance
def plot_explained_variance(explained_variance):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()


plot_explained_variance(explained_variance)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=pca_result, index=pivot_data.index, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Display the PCA components' loadings
loadings = pd.DataFrame(pca.components_, columns=pivot_data.columns, index=[f'PC{i+1}' for i in range(len(pca.components_))])

print("PCA Loadings:")
print(loadings)

# Plot the loadings for the first two principal components
def plot_pca_loadings(loadings):
    plt.figure(figsize=(12, 6))
    loadings.iloc[:2].T.plot(kind='bar', figsize=(14, 6))
    plt.title('PCA Loadings: PC1 and PC2')
    plt.ylabel('Loading Value')
    plt.xlabel('Stocks')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(title='Principal Components')
    plt.show()

# Plot the loadings for PC1 and PC2
plot_pca_loadings(loadings)


hierarchical_clusters = hierarchical_clustering(scaled_features, features)


kmeans_clustering(scaled_features, features, n_clusters=3)

plot_correlation_matrix(pivot_data)


plot_rolling_correlation(pivot_data, stock1='QCOM', stock2='AMD', window=30)

plot_cumulative_returns_companies(data)


plot_risk_return_companies(data)
