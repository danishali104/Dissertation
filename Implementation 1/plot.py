import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df_db_kmeans = pd.read_csv('kmeans_sql.csv')
df_cs_kmeans = pd.read_csv('kmeans_csv.csv')
df_mongo_kmeans = pd.read_csv('kmeans_mongo.csv')

df_db_decision_tree = pd.read_csv('decision_tree_sql.csv')
df_cs_decision_tree = pd.read_csv('decision_tree_csv.csv')
df_mongo_decision_tree = pd.read_csv('decision_tree_mongo.csv')

# Add a source column to each DataFrame
df_db_kmeans['Source'] = 'SQL'
df_cs_kmeans['Source'] = 'CSV'
df_mongo_kmeans['Source'] = 'MongoDB'

df_db_decision_tree['Source'] = 'SQL'
df_cs_decision_tree['Source'] = 'CSV'
df_mongo_decision_tree['Source'] = 'MongoDB'

# Concatenate DataFrames
df_kmeans = pd.concat([df_db_kmeans, df_cs_kmeans, df_mongo_kmeans], ignore_index=True)
df_decision_tree = pd.concat([df_db_decision_tree, df_cs_decision_tree, df_mongo_decision_tree], ignore_index=True)

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

# Plotting and saving for KMeans clustering
plot_results(df_kmeans, 'rows', 'cleaning_time', 'Cleaning Time Comparison (KMeans Clustering)', 'Cleaning Time (s)', 'Graphs1/kmeans_cleaning_time_comparison.png')
plot_results(df_kmeans, 'rows', 'kmeans_time', 'Clustering Time Comparison (KMeans)', 'Clustering Time (s)', 'Graphs1/kmeans_time_comparison.png')
plot_results(df_kmeans, 'rows', 'kmeans_inertia', 'Clustering Accuracy (ARI) Comparison (KMeans)', 'adjusted rand score', 'Graphs1/kmeans_ari_comparison.png')

# Plotting and saving for Decision Tree classification
plot_results(df_decision_tree, 'rows', 'cleaning_time', 'Cleaning Time Comparison (Decision Tree Classification)', 'Cleaning Time (s)', 'Graphs1/decision_tree_cleaning_time_comparison.png')
plot_results(df_decision_tree, 'rows', 'dt_time', 'Classification Time Comparison (Decision Tree)', 'Classification Time (s)', 'Graphs1/decision_tree_time_comparison.png')
plot_results(df_decision_tree, 'rows', 'dt_accuracy', 'Classification Accuracy Comparison (Decision Tree)', 'Accuracy', 'Graphs1/decision_tree_accuracy_comparison.png')
