# imports
import pandas as pd
import sqlite3
import pymongo
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from time import time

# Initialize SQL database
def initialize_sql():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS synthetic_data (
        feature_1 REAL,
        feature_2 REAL,
        feature_3 REAL,
        feature_4 REAL,
        feature_5 REAL,
        target INTEGER
    )
    ''')
    conn.commit()
    conn.close()

# Initialize MongoDB
def initialize_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["example_db"]
    collection = db["synthetic_data"]
    collection.drop()

# Generate synthetic data
def generate_data(n_samples):
    X, y = make_classification(n_samples=n_samples, n_features=5, n_classes=2, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
    df['target'] = y
    return df

# Load data into SQL
def load_sql(df):
    conn = sqlite3.connect('example.db')
    df.to_sql('synthetic_data', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

# Load data into MongoDB
def load_mongo(df):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["example_db"]
    collection = db["synthetic_data"]
    collection.delete_many({})
    collection.insert_many(df.to_dict('records'))

# Load data into CSV
def load_csv(df):
    df.to_csv('synthetic_data.csv', index=False)

# Preprocess SQL data
def preprocess_sql():
    start_time = time()
    conn = sqlite3.connect('example.db')
    df = pd.read_sql('SELECT * FROM synthetic_data', conn)
    df = clean_data(df)
    df.to_sql('synthetic_data', conn, if_exists='replace', index=False)
    conn.close()
    elapsed_time = time() - start_time
    return elapsed_time

# Preprocess MongoDB data
def preprocess_mongo():
    start_time = time()
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["example_db"]
    collection = db["synthetic_data"]
    df = pd.DataFrame(list(collection.find({}, {'_id': 0})))  # Exclude '_id' field
    df = clean_data(df)
    collection.delete_many({})
    collection.insert_many(df.to_dict('records'))
    elapsed_time = time() - start_time
    return elapsed_time

# Preprocess CSV data
def preprocess_csv():
    start_time = time()
    df = pd.read_csv('synthetic_data.csv')
    df = clean_data(df)
    df.to_csv('synthetic_data.csv', index=False)
    elapsed_time = time() - start_time
    return elapsed_time

def clean_data(df):
    # Handle missing values
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical column
            df[column].fillna(df[column].mode()[0], inplace=True)  # Fill with mode
        else:  # Numeric column
            df[column].fillna(df[column].mean(), inplace=True)  # Fill with mean
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Detect and handle outliers
    numeric_cols = df.select_dtypes(include=['number']).columns
    z_scores = stats.zscore(df[numeric_cols])
    abs_z_scores = abs(z_scores)
    filter_outliers = (abs_z_scores < 3).all(axis=1)  # Z-score threshold of 3
    df = df[filter_outliers]
    
    return df


# Datamining technique: KMeans for SQL
def kmeans_sql():
    start_time = time()
    conn = sqlite3.connect('example.db')
    df = pd.read_sql('SELECT * FROM synthetic_data', conn)
    X = df.drop('target', axis=1)
    y = df['target']  # Ground truth labels
    kmeans = KMeans(n_clusters=2, init='random', random_state=None)
    kmeans.fit(X)
    labels = kmeans.labels_
    elapsed_time = time() - start_time
    conn.close()
    ari = adjusted_rand_score(y, labels)
    return elapsed_time, ari

# Datamining technique: KMeans for MongoDB

def kmeans_mongo():
    start_time = time()
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["example_db"]
    collection = db["synthetic_data"]
    df = pd.DataFrame(list(collection.find({}, {'_id': 0})))  # Exclude '_id' field
    X = df.drop('target', axis=1)
    y = df['target']  # Ground truth labels
    X = X.apply(pd.to_numeric, errors='coerce')  # Ensure all data is numeric
    kmeans = KMeans(n_clusters=2, init='random', random_state=None)
    kmeans.fit(X)
    labels = kmeans.labels_
    elapsed_time = time() - start_time
    ari = adjusted_rand_score(y, labels)
    return elapsed_time, ari

# Datamining technique: KMeans for CSV
def kmeans_csv():
    start_time = time()
    df = pd.read_csv('synthetic_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']  # Ground truth labels
    kmeans = KMeans(n_clusters=2, init='random', random_state=None)
    kmeans.fit(X)
    labels = kmeans.labels_
    elapsed_time = time() - start_time
    ari = adjusted_rand_score(y, labels)
    return elapsed_time, ari
# Datamining technique: Decision Tree for SQL
def decision_tree_sql():
    start_time = time()
    conn = sqlite3.connect('example.db')
    df = pd.read_sql('SELECT * FROM synthetic_data', conn)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = time() - start_time
    conn.close()
    return elapsed_time, accuracy

# Datamining technique: Decision Tree for MongoDB
def decision_tree_mongo():
    start_time = time()
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["example_db"]
    collection = db["synthetic_data"]
    df = pd.DataFrame(list(collection.find({}, {'_id': 0})))  # Exclude '_id' field
    X = df.drop('target', axis=1)
    y = df['target']
    X = X.apply(pd.to_numeric, errors='coerce')  # Ensure all data is numeric
    y = pd.to_numeric(y, errors='coerce')  # Ensure target is numeric
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = time() - start_time
    return elapsed_time, accuracy

# Datamining technique: Decision Tree for CSV
def decision_tree_csv():
    start_time = time()
    df = pd.read_csv('synthetic_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = time() - start_time
    return elapsed_time, accuracy

# Write results to CSV
def write_results(filename, rows, cleaning_time, dm_time, metric):
    with open(filename, 'w') as f:
        f.write(f"{rows},{cleaning_time},{dm_time},{metric}\n")
        
def append_results(filename, rows, cleaning_time, dm_time, metric):
    with open(filename, 'a') as f:
        f.write(f"{rows},{cleaning_time},{dm_time},{metric}\n")

# Main execution
initialize_sql()
initialize_mongo()

#add headers to the file
write_results('kmeans_sql.csv', "rows", "cleaning_time", "kmeans_time", "kmeans_inertia")
write_results('kmeans_mongo.csv', "rows", "cleaning_time", "kmeans_time", "kmeans_inertia")
write_results('kmeans_csv.csv', "rows", "cleaning_time", "kmeans_time", "kmeans_inertia")
write_results('decision_tree_sql.csv', "rows", "cleaning_time", "dt_time", "dt_accuracy")
write_results('decision_tree_mongo.csv', "rows", "cleaning_time", "dt_time", "dt_accuracy")
write_results('decision_tree_csv.csv', "rows", "cleaning_time", "dt_time", "dt_accuracy")


for rows in range(500, 25000 + 1, 500):
    df = generate_data(rows)
    # Load data
    # Load data
    load_sql(df)
    load_mongo(df)
    load_csv(df)
    
    # Preprocess data
    cleaning_time_sql = preprocess_sql()
    cleaning_time_mongo = preprocess_mongo()
    cleaning_time_csv = preprocess_csv()
    
    # Apply KMeans
    kmeans_time_sql, kmeans_inertia_sql = kmeans_sql()
    kmeans_time_mongo, kmeans_inertia_mongo = kmeans_mongo()
    kmeans_time_csv, kmeans_inertia_csv = kmeans_csv()
    
    # Apply Decision Tree
    dt_time_sql, dt_accuracy_sql = decision_tree_sql()
    dt_time_mongo, dt_accuracy_mongo = decision_tree_mongo()
    dt_time_csv, dt_accuracy_csv = decision_tree_csv()
    
    # Write results
    append_results('kmeans_sql.csv', rows, cleaning_time_sql, kmeans_time_sql, kmeans_inertia_sql)
    append_results('kmeans_mongo.csv', rows, cleaning_time_mongo, kmeans_time_mongo, kmeans_inertia_mongo)
    append_results('kmeans_csv.csv', rows, cleaning_time_csv, kmeans_time_csv, kmeans_inertia_csv)
    
    append_results('decision_tree_sql.csv', rows, cleaning_time_sql, dt_time_sql, dt_accuracy_sql)
    append_results('decision_tree_mongo.csv', rows, cleaning_time_mongo, dt_time_mongo, dt_accuracy_mongo)
    append_results('decision_tree_csv.csv', rows, cleaning_time_csv, dt_time_csv, dt_accuracy_csv)