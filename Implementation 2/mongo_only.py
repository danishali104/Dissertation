import pymongo
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from faker import Faker
import pandas as pd
import time
import random

def connect_to_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["transaction_db"]
    return db

def generate_mongo_database(db, rows):
    db.customers.drop()
    db.products.drop()
    db.transactions.drop()
    
    fake = Faker()
    
    customers = []
    for _ in range(rows // 10):
        customers.append({"name": fake.name()})
    customer_ids = db.customers.insert_many(customers).inserted_ids
    
    products = []
    for _ in range(rows // 20):
        products.append({"name": fake.word(), "price": round(random.uniform(10, 1000), 2)})
    product_ids = db.products.insert_many(products).inserted_ids
    
    transactions = []
    for _ in range(rows):
        customer_id = random.choice(customer_ids)
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 10)
        transactions.append({"customer_id": customer_id, "product_id": product_id, "quantity": quantity})
    db.transactions.insert_many(transactions)

def clean_data_mongo(db):
    start_time = time.time()
    db.transactions.delete_many({"quantity": {"$lte": 0}})
    end_time = time.time()
    return end_time - start_time

def apply_kmeans_mongo(db):
    pipeline = [
        {
            "$lookup": {
                "from": "products",
                "localField": "product_id",
                "foreignField": "_id",
                "as": "product"
            }
        },
        {
            "$unwind": "$product"
        },
        {
            "$group": {
                "_id": "$customer_id",
                "total_spent": {"$sum": {"$multiply": ["$quantity", "$product.price"]}}
            }
        }
    ]
    customer_spending = list(db.transactions.aggregate(pipeline))
    df = pd.DataFrame(customer_spending)
    
    start_time = time.time()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['total_spent']])
    end_time = time.time()
    
    silhouette_avg = silhouette_score(df[['total_spent']], df['cluster'])
    calinski_harabasz = calinski_harabasz_score(df[['total_spent']], df['cluster'])
    davies_bouldin = davies_bouldin_score(df[['total_spent']], df['cluster'])
    
    return silhouette_avg, calinski_harabasz, davies_bouldin, end_time - start_time

def apply_classification_mongo(db):
    pipeline = [
        {
            "$lookup": {
                "from": "products",
                "localField": "product_id",
                "foreignField": "_id",
                "as": "product"
            }
        },
        {
            "$unwind": "$product"
        },
        {
            "$lookup": {
                "from": "customers",
                "localField": "customer_id",
                "foreignField": "_id",
                "as": "customer"
            }
        },
        {
            "$unwind": "$customer"
        },
        {
            "$project": {
                "customer_id": 1,
                "total_spent": {"$multiply": ["$quantity", "$product.price"]},
                "customer_name": "$customer.name"
            }
        }
    ]
    data = list(db.transactions.aggregate(pipeline))
    df = pd.DataFrame(data)
    
    start_time = time.time()
    X = df[['total_spent']]
    y = df['customer_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    end_time = time.time()
    
    return accuracy, report, end_time - start_time

### MAIN ###
filedb_clustering = 'results_mongo_clustering.csv'
filedb_classification = 'results_mongo_classification.csv'

# Initialize files for clustering results
#with open(filedb_clustering, mode='w') as f:
#    f.write("Data Size (rows),Cleaning Time (s),Clustering Time (s),Silhouette Score,Calinski-Harabasz Index,Davies-Bouldin Index\n")

# Initialize files for classification results
#with open(filedb_classification, mode='w') as f:
#    f.write("Data Size (rows),Cleaning Time (s),Classification Time (s),Accuracy\n")

for rows in range(13500, 15500, 500):
    db = connect_to_mongo()
    
    # Generate Data
    generate_mongo_database(db, rows)
    
    # Cleaning
    cleaning_time = clean_data_mongo(db)
    
    # Clustering
    silhouette_avg, calinski_harabasz, davies_bouldin, clustering_time = apply_kmeans_mongo(db)
    with open(filedb_clustering, mode='a') as f:
        f.write(f"{rows},{cleaning_time:.4f},{clustering_time:.4f},{silhouette_avg:.4f},{calinski_harabasz:.4f},{davies_bouldin:.4f}\n")
    
    # Classification
    accuracy, report, classification_time = apply_classification_mongo(db)
    with open(filedb_classification, mode='a') as f:
        f.write(f"{rows},{cleaning_time:.4f},{classification_time:.4f},{accuracy:.4f}\n")
