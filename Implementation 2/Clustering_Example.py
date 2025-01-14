import mysql.connector
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,classification_report,silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from faker import Faker
import pandas as pd
import time
import random
import csv
import os 

'''
One time run to make the schema
Database - MYSQL: 
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10, 2)
);

CREATE TABLE transactions (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

'''

def connect_to_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='transaction_db'
    )
    

def generate_database(conn,cursor,rows): 
    # Clear existing data
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    cursor.execute("TRUNCATE TABLE transactions;")
    cursor.execute("TRUNCATE TABLE customers;")
    cursor.execute("TRUNCATE TABLE products;")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    conn.commit()
    
    # generate Fake data
    fake = Faker()
    
    # Generate Customers
    customers = []
    for _ in range(rows//10):
        name = fake.name()
        email = fake.email()
        customers.append((name,))
    insert_customers = "INSERT INTO customers (name) VALUES (%s)"
    cursor.executemany(insert_customers, customers)
    conn.commit()
    
    # Generate Products
    products = []
    for _ in range(rows//20):
        name = fake.word()
        price = round(random.uniform(10, 1000), 2)
        products.append((name, price))
    insert_products = "INSERT INTO products (name, price) VALUES (%s,  %s)"
    cursor.executemany(insert_products, products)
    conn.commit()
    
    #Generate Transactions
    transactions = []
    for _ in range(rows):
        customer_id = random.randint(1, rows//10)
        product_id = random.randint(1, rows//20)
        quantity = random.randint(1, 10)
        transactions.append((customer_id, product_id, quantity))
    insert_transactions = "INSERT INTO transactions (customer_id, product_id, quantity) VALUES (%s, %s, %s)"
    cursor.executemany(insert_transactions, transactions)
    conn.commit()


def generate_csv(conn, cursor, data_file):
    query = """
    SELECT *
    FROM customers c
    JOIN transactions t ON c.customer_id = t.customer_id
    JOIN products p ON p.product_id = t.product_id
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    # Get column names
    columns = [i[0] for i in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(data_file, index=False)



def clean_data_csv(data_file):
    start_time = time.time()
    data = pd.read_csv(data_file)
    cleaned_data = data[data['quantity'] > 0]
    end_time = time.time()
    return cleaned_data, end_time - start_time
    
    
## Data cleaning SQL operations
def clean_data_db(conn,cursor):
    start_time = time.time()
    clean_query = """
    DELETE FROM transactions WHERE quantity <= 0;
    """
    cursor.execute(clean_query)
    conn.commit()
    end_time = time.time()
    return  end_time - start_time


def apply_kmeans_csv(data):
    start_time = time.time()
    
    # Calculate total spent per customer
    customer_spending = data.groupby('customer_id').apply(
        lambda x: (x['quantity'] * x['price']).sum()
    ).reset_index()
    customer_spending.columns = ['customer_id', 'total_spent']
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_spending['cluster'] = kmeans.fit_predict(customer_spending[['total_spent']])
    
    end_time = time.time()
    
    # Calculate metrics
    silhouette_avg = silhouette_score(customer_spending[['total_spent']], customer_spending['cluster'])
    calinski_harabasz = calinski_harabasz_score(customer_spending[['total_spent']], customer_spending['cluster'])
    davies_bouldin = davies_bouldin_score(customer_spending[['total_spent']], customer_spending['cluster'])
    
    return silhouette_avg, calinski_harabasz, davies_bouldin, end_time - start_time
    

def apply_kmeans_db(conn, cursor):
    ## Applying K-means Clustering
    query = """
    SELECT t.customer_id, SUM(p.price * t.quantity) AS total_spent
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY t.customer_id
    """
    customer_spending = pd.read_sql(query, conn)
    conn.close()

    start_time = time.time()
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_spending['cluster'] = kmeans.fit_predict(customer_spending[['total_spent']])
    end_time = time.time()

    # Calculate metrics
    silhouette_avg = silhouette_score(customer_spending[['total_spent']], customer_spending['cluster'])
    calinski_harabasz = calinski_harabasz_score(customer_spending[['total_spent']], customer_spending['cluster'])
    davies_bouldin = davies_bouldin_score(customer_spending[['total_spent']], customer_spending['cluster'])
    return (silhouette_avg,calinski_harabasz,davies_bouldin,end_time - start_time)


def apply_classification_db(conn, cursor):
    # Retrieve data from database
    query = """
    SELECT t.customer_id, p.price * t.quantity AS total_spent, c.name AS customer_name
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    JOIN customers c ON t.customer_id = c.customer_id
    """
    data = pd.read_sql(query, conn)
    start_time = time.time()
    # Prepare data for classification
    X = data[['total_spent']]
    y = data['customer_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    end_time = time.time()
    return accuracy, report, end_time - start_time


def apply_classification_csv(data):
    start_time = time.time()
    
    # Calculate total spent per transection
    data['total_spent'] = data['quantity'] * data['price']
    X = data[['total_spent']]
    y = data['name']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    end_time = time.time()
    
    return accuracy, report, end_time - start_time


### MAIN ###
filedb_clustering = 'results_db_clustering.csv'
filecs_clustering = 'results_cs_clustering.csv'
filedb_classification = 'results_db_classification.csv'
filecs_classification = 'results_cs_classification.csv'
data_file = 'customer_transaction_data.csv'

# Initialize files for clustering results
with open(filedb_clustering, mode='w') as f:
    f.write("Data Size (rows),Cleaning Time (s),Clustering Time (s),Silhouette Score,Calinski-Harabasz Index,Davies-Bouldin Index\n")
with open(filecs_clustering, mode='w') as f:
    f.write("Data Size (rows),Cleaning Time (s),Clustering Time (s),Silhouette Score,Calinski-Harabasz Index,Davies-Bouldin Index\n")

# Initialize files for classification results
with open(filedb_classification, mode='w') as f:
    f.write("Data Size (rows),Cleaning Time (s),Classification Time (s),Accuracy\n")
with open(filecs_classification, mode='w') as f:
    f.write("Data Size (rows),Cleaning Time (s),Classification Time (s),Accuracy\n")


for rows in range(500, 15500, 500):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Generate Data
    generate_database(conn, cursor, rows)
    generate_csv(conn, cursor, data_file)
    
    # Cleaning
    cleaning_timedb = clean_data_db(conn, cursor)
    data, cleaning_timecsv = clean_data_csv(data_file)
    
    # Clustering
    silhouette_avg, calinski_harabasz, davies_bouldin, clustering_time = apply_kmeans_db(conn, cursor)
    with open(filedb_clustering, mode='a') as f:
        f.write(f"{rows},{cleaning_timedb:.4f},{clustering_time:.4f},{silhouette_avg:.4f},{calinski_harabasz:.4f},{davies_bouldin:.4f}\n")
    
    silhouette_avg, calinski_harabasz, davies_bouldin, clustering_time = apply_kmeans_csv(data)
    with open(filecs_clustering, mode='a') as f:
        f.write(f"{rows},{cleaning_timecsv:.4f},{clustering_time:.4f},{silhouette_avg:.4f},{calinski_harabasz:.4f},{davies_bouldin:.4f}\n")
    
    # reconnecting
    cursor.close()
    conn.close()
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Classification
    accuracy_db, report_db, classification_time_db = apply_classification_db(conn, cursor)
    with open(filedb_classification, mode='a') as f:
        f.write(f"{rows},{cleaning_timedb:.4f},{classification_time_db:.4f},{accuracy_db:.4f}\n")
   
    accuracy_csv, report_csv, classification_time_csv = apply_classification_csv(data)
    with open(filecs_classification, mode='a') as f:
        f.write(f"{rows},{cleaning_timecsv:.4f},{classification_time_csv:.4f},{accuracy_csv:.4f}\n")

    cursor.close()
    conn.close()




 