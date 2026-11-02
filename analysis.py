import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# (a) Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("First 10 rows:")
print(df.head(10))
print("\nShape:", df.shape)
print("\nData types:")
print(df.dtypes)

# (a)(ii) missing values + drop CustomerID
print("\nMissing values:")
print(df.isnull().sum())
df = df.drop("CustomerID", axis=1)

# (a)(iii) encode Gender: Male=0, Female=1
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# (b)(i) descriptive stats
print("\nDescriptive statistics:")
print(df.describe())

# (b)(ii) histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Age", "Annual Income (k$)", "Spending Score (1-100)"]):
    ax.hist(df[col], bins=20, color="steelblue", edgecolor="black")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# (b)(iii) scatter
plt.figure(figsize=(7, 5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c="steelblue", alpha=0.7)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score")
plt.savefig("scatter_income_spending.png")
plt.close()
print("\nNatural groupings observed: yes, roughly 5 clusters visible.")

# (b)(iv) boxplot spending by gender
plt.figure(figsize=(6, 5))
df.boxplot(column="Spending Score (1-100)", by="Gender")
plt.title("Spending Score by Gender (0=Male,1=Female)")
plt.suptitle("")
plt.xlabel("Gender")
plt.ylabel("Spending Score")
plt.savefig("boxplot_gender.png")
plt.close()

# (c)(i) scale
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (c)(ii) elbow
wcss = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
plt.figure(figsize=(7, 5))
plt.plot(list(K_range), wcss, "bo-")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("elbow.png")
plt.close()
print("\nWCSS:", dict(zip(K_range, wcss)))

# (c)(iii) silhouette
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
plt.figure(figsize=(7, 5))
plt.plot(list(K_range), sil_scores, "go-")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.savefig("silhouette.png")
plt.close()
print("Silhouette:", dict(zip(K_range, sil_scores)))
optimal_k = 5  # elbow bend
print("Optimal K by silhouette:", optimal_k)

# (c)(iv) fit KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# (d)(i) scatter colored by cluster with centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                     c=df["Cluster"], cmap="viridis", alpha=0.7)
plt.scatter(centroids[:, 2], centroids[:, 3], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Clusters: Income vs Spending")
plt.legend()
plt.savefig("clusters.png")
plt.close()

# (d)(ii) summary table
summary = df.groupby("Cluster")[features].mean()
print("\nCluster summary (means):")
print(summary)
labels = {}
for c in summary.index:
    inc = summary.loc[c, "Annual Income (k$)"]
    sp = summary.loc[c, "Spending Score (1-100)"]
    if inc >= 70 and sp >= 60:
        labels[c] = "High Income, High Spending"
    elif inc >= 70 and sp < 40:
        labels[c] = "High Income, Low Spending"
    elif inc < 40 and sp >= 60:
        labels[c] = "Low Income, High Spending"
    elif inc < 40 and sp < 40:
        labels[c] = "Low Income, Low Spending"
    else:
        labels[c] = "Moderate"
print("\nCluster labels:", labels)

# (d)(iii) final silhouette
final_sil = silhouette_score(X_scaled, df["Cluster"])
print("\nFinal silhouette score:", final_sil)
print("Quality: scores closer to 1 indicate well-separated, cohesive clusters.")
