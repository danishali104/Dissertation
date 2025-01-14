import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df_db_clustering = pd.read_csv('results_db_clustering.csv')
df_cs_clustering = pd.read_csv('results_cs_clustering.csv')
df_mongo_clustering = pd.read_csv('results_mongo_clustering.csv')  # Assuming you have this file

df_db_classification = pd.read_csv('results_db_classification.csv')
df_cs_classification = pd.read_csv('results_cs_classification.csv')
df_mongo_classification = pd.read_csv('results_mongo_classification.csv')  # Assuming you have this file

# Add a source column to each DataFrame
df_db_clustering['Source'] = 'SQL'
df_cs_clustering['Source'] = 'CSV'
df_mongo_clustering['Source'] = 'MongoDB'

df_db_classification['Source'] = 'SQL'
df_cs_classification['Source'] = 'CSV'
df_mongo_classification['Source'] = 'MongoDB'

# Concatenate DataFrames
df_clustering = pd.concat([df_db_clustering, df_cs_clustering, df_mongo_clustering], ignore_index=True)
df_classification = pd.concat([df_db_classification, df_cs_classification, df_mongo_classification], ignore_index=True)

# Plotting function with saving capability
def plot_results(df, x_col, y_col, title, y_label, save_as=None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue='Source', marker='o')
    plt.title(title)
    plt.xlabel('Data Size (rows)')
    plt.ylabel(y_label)
    plt.legend(title='Source')
    plt.grid(True)
    
    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()

# Plotting and saving
plot_results(df_clustering, 'Data Size (rows)', 'Cleaning Time (s)', 'Cleaning Time Comparison', 'Cleaning Time (s)', 'cleaning_time_comparison.png')
plot_results(df_clustering, 'Data Size (rows)', 'Clustering Time (s)', 'Clustering Time Comparison', 'Clustering Time (s)', 'clustering_time_comparison.png')
plot_results(df_clustering, 'Data Size (rows)', 'Silhouette Score', 'Clustering Accuracy (Silhouette Score) Comparison', 'Silhouette Score', 'silhouette_score_comparison.png')
plot_results(df_classification, 'Data Size (rows)', 'Classification Time (s)', 'Classification Time Comparison', 'Classification Time (s)', 'classification_time_comparison.png')
plot_results(df_classification, 'Data Size (rows)', 'Accuracy', 'Classification Accuracy Comparison', 'Accuracy', 'classification_accuracy_comparison.png')