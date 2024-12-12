from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.sql import Row

# Initialize Spark Session
spark = SparkSession.builder.appName("ParallelKMeansClustering").getOrCreate()
sc = spark.sparkContext

# Load the provided dataset and broadcast it to all workers
initial_data = pd.read_csv("/opt/bitnami/spark/data/data/Clustering_dataset.txt", sep=' ', header=None)
initial_data.columns = ['x', 'y']
data = initial_data[['x', 'y']].values.tolist()  # Assuming dataset has 'x' and 'y' columns
data_broadcast = sc.broadcast(data)

# Define the range of k-values to test
k_values = [6, 7, 8, 9]

# Function to generate initial random centers for a given k
def generate_initial_centers(k):
    np_random_x = np.random.randint(100_000, 500_000, k)
    np_random_y = np.random.randint(100_000, 500_000, k)
    centers = [(float(x), float(y)) for x, y in zip(np_random_x, np_random_y)]
    return centers  

# Function to calculate silhouette score manually
def calculate_silhouette_score(df):
    scores = []
    for index, row in df.iterrows():
        # Intra-cluster distance (a)
        same_cluster = df[df['k'] == row['k']]
        a = np.mean(np.sqrt((same_cluster['x'] - row['x'])**2 + (same_cluster['y'] - row['y'])**2))

        # Nearest-cluster distance (b)
        other_clusters = df[df['k'] != row['k']]
        b = np.min([
            np.mean(np.sqrt((other_clusters[other_clusters['k'] == cluster]['x'] - row['x'])**2 +
                            (other_clusters[other_clusters['k'] == cluster]['y'] - row['y'])**2))
            for cluster in other_clusters['k'].unique()
        ])

        # Calculate silhouette score for the point
        score = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(score)

    # Return the average silhouette score for all points
    return np.mean(scores)

# K-Means Clustering function
def kmeans_clustering(k):
    data = data_broadcast.value
    df = pd.DataFrame(data, columns=['x', 'y'])
    centers = generate_initial_centers(k)
    centers = pd.DataFrame(centers, columns=['x', 'y'])
    check = False
    i = 1

    while not check and i < 10:
        for index, row in df.iterrows():
            distances = [
                np.sqrt((row['x'] - row2['x'])**2 + (row['y'] - row2['y'])**2)
                for _, row2 in centers.iterrows()
            ]
            df.loc[index, 'k'] = distances.index(min(distances)) + 1

        center_new = pd.DataFrame()
        for j in range(1, k + 1):
            center_new.loc[j, 'x'] = df[df['k'] == j]['x'].mean()
            center_new.loc[j, 'y'] = df[df['k'] == j]['y'].mean()

        center_new = center_new.round(2)

        if np.allclose(centers[['x', 'y']].values, center_new[['x', 'y']].values):
            check = True
        else:
            centers = center_new

        i += 1


    # Calculate silhouette score
    score = calculate_silhouette_score(df)

    # Save results for each k in separate files
    centers.to_csv(f"/opt/bitnami/spark/data/data/centers_k_{k}.csv", index=False)
    
    with open(f"/opt/bitnami/spark/data/data/silhouette_score_k_{k}.csv", "w") as f:
        f.write(f"k,{k}\nSilhouette Score,{score:.2f}")

    return k, centers, score  # Return for potential further processing if needed

# Parallel execution for each k-value
sc.parallelize(k_values, numSlices=4).repartition(4).map(kmeans_clustering).collect()

results = [
    (int(k), pd.read_csv(f"/opt/bitnami/spark/data/data/centers_k_{k}.csv").values.tolist(), 
     float(pd.read_csv(f"/opt/bitnami/spark/data/data/silhouette_score_k_{k}.csv").iloc[0, 1]))
    for k in k_values
]   

data = sc.parallelize(results)

# Convert the results to a DataFrame
results_df = data.map(lambda x: Row(k=x, centers=x[1], silhouette_score=x[2])).toDF()

# Show the DataFrame
results_df.show()

# Stop the Spark Context
spark.stop()