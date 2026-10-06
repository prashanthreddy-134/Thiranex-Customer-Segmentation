from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# CUSTOMER SEGMENTATION - THIRANEX TASK 2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print(f"\nCustomers loaded: {len(df)}")


# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

df = df.drop_duplicates().copy()

numeric_columns = [
    "Age",
    "AnnualIncome",
    "PurchaseFrequency",
    "AverageOrderValue",
    "TotalSpend",
    "RecencyDays",
    "WebVisitsPerMonth",
    "DiscountUsageCount"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

print(f"Customers after cleaning: {len(df)}")


# ------------------------------------------------------------
# 3. SELECT BEHAVIORAL + DEMOGRAPHIC FEATURES
# ------------------------------------------------------------

features = [
    "AnnualIncome",
    "PurchaseFrequency",
    "AverageOrderValue",
    "TotalSpend",
    "RecencyDays",
    "WebVisitsPerMonth",
    "DiscountUsageCount"
]

X = df[features]


# ------------------------------------------------------------
# 4. STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 5. ELBOW METHOD
# ------------------------------------------------------------

print("\nCalculating Elbow Method...")

k_values = range(2, 9)

inertias = []
silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette = {score:.3f}"
    )


# ------------------------------------------------------------
# 6. ELBOW CHART
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_elbow_method.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 7. SILHOUETTE SCORE CHART
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_silhouette_scores.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 8. SELECT BEST K
# ------------------------------------------------------------

best_index = silhouette_scores.index(
    max(silhouette_scores)
)

best_k = list(k_values)[best_index]

best_silhouette = silhouette_scores[best_index]

print("\n" + "=" * 60)

print(
    f"Best number of clusters: {best_k}"
)

print(
    f"Best Silhouette Score: {best_silhouette:.3f}"
)

print("=" * 60)


# ------------------------------------------------------------
# 9. FINAL K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# ------------------------------------------------------------
# 10. PCA VISUALIZATION
# ------------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

pca_values = pca.fit_transform(
    X_scaled
)

df["PCA1"] = pca_values[:, 0]
df["PCA2"] = pca_values[:, 1]


plt.figure(figsize=(9, 6))

for cluster in sorted(
    df["Cluster"].unique()
):

    subset = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=35,
        alpha=0.65,
        label=f"Cluster {cluster}"
    )


plt.title(
    "Customer Segmentation using K-Means and PCA"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.legend()

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_customer_segments_pca.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 11. CUSTOMER SEGMENT PROFILE
# ------------------------------------------------------------

profile = df.groupby(
    "Cluster"
)[features].mean().round(2)

profile["CustomerCount"] = (
    df.groupby("Cluster").size()
)

profile["CustomerPercentage"] = (
    profile["CustomerCount"]
    / len(df)
    * 100
).round(2)


# ------------------------------------------------------------
# 12. CREATE BUSINESS SEGMENT NAMES
# ------------------------------------------------------------

segment_score = (

    profile["TotalSpend"].rank(
        pct=True
    ) * 0.45

    +

    profile["PurchaseFrequency"].rank(
        pct=True
    ) * 0.35

    +

    (
        1
        -
        profile["RecencyDays"].rank(
            pct=True
        )
    ) * 0.20
)


ranked_clusters = (
    segment_score
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


segment_names = [
    "High-Value Customers",
    "Loyal Regular Customers",
    "Potential Customers",
    "Low-Engagement Customers"
]


# Make sure number of names matches number of clusters
if best_k <= len(segment_names):

    labels = {}

    for i, cluster in enumerate(
        ranked_clusters
    ):

        labels[cluster] = segment_names[i]

else:

    labels = {
        cluster: f"Customer Segment {i + 1}"
        for i, cluster
        in enumerate(ranked_clusters)
    }


df["Segment"] = df["Cluster"].map(
    labels
)

profile["Segment"] = profile.index.map(
    labels
)


# ------------------------------------------------------------
# 13. SAVE SEGMENTED CUSTOMER DATA
# ------------------------------------------------------------

df.drop(
    columns=["PCA1", "PCA2"]
).to_csv(
    OUTPUT_DIR / "segmented_customers.csv",
    index=False
)


profile.sort_values(
    "TotalSpend",
    ascending=False
).to_csv(
    OUTPUT_DIR / "segment_profile.csv"
)


# ------------------------------------------------------------
# 14. SEGMENT DISTRIBUTION
# ------------------------------------------------------------

counts = (
    df["Segment"]
    .value_counts()
)


plt.figure(figsize=(10, 5))

counts.plot(
    kind="bar"
)

plt.title(
    "Customer Distribution by Segment"
)

plt.xlabel(
    "Customer Segment"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_segment_distribution.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 15. AVERAGE SPENDING BY SEGMENT
# ------------------------------------------------------------

spending = (
    df.groupby("Segment")
    ["TotalSpend"]
    .mean()
    .sort_values(
        ascending=False
    )
)


plt.figure(figsize=(10, 5))

spending.plot(
    kind="bar"
)

plt.title(
    "Average Total Spend by Customer Segment"
)

plt.xlabel(
    "Customer Segment"
)

plt.ylabel(
    "Average Total Spend"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_average_spend_by_segment.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 16. PURCHASE FREQUENCY VS SPENDING
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for segment in df["Segment"].unique():

    subset = df[
        df["Segment"] == segment
    ]

    plt.scatter(
        subset["PurchaseFrequency"],
        subset["TotalSpend"],
        s=30,
        alpha=0.55,
        label=segment
    )


plt.title(
    "Purchase Frequency vs Total Spend"
)

plt.xlabel(
    "Purchase Frequency"
)

plt.ylabel(
    "Total Spend"
)

plt.legend(
    fontsize=8
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_frequency_vs_spend.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 17. DEMOGRAPHIC ANALYSIS
# ------------------------------------------------------------

gender_summary = pd.crosstab(
    df["Segment"],
    df["Gender"]
)

gender_summary.to_csv(
    OUTPUT_DIR / "gender_by_segment.csv"
)


city_summary = pd.crosstab(
    df["Segment"],
    df["City"]
)

city_summary.to_csv(
    OUTPUT_DIR / "city_by_segment.csv"
)


# ------------------------------------------------------------
# 18. BUSINESS INSIGHTS
# ------------------------------------------------------------

highest_spending_cluster = (
    profile["TotalSpend"]
    .idxmax()
)

highest_frequency_cluster = (
    profile["PurchaseFrequency"]
    .idxmax()
)

least_recent_cluster = (
    profile["RecencyDays"]
    .idxmax()
)


insights = f"""
CUSTOMER SEGMENTATION ANALYSIS
==============================

Total Customers:
{len(df)}

Optimal Number of Clusters:
{best_k}

Silhouette Score:
{best_silhouette:.3f}


KEY FINDINGS
------------

Highest Spending Segment:
{labels[highest_spending_cluster]}

Highest Purchase Frequency:
{labels[highest_frequency_cluster]}

Least Recently Active Segment:
{labels[least_recent_cluster]}


BUSINESS RECOMMENDATIONS
------------------------

1. High-Value Customers
Provide premium offers, loyalty rewards,
exclusive products and early access.

2. Loyal Regular Customers
Use loyalty programs, membership benefits,
bundles and repeat-purchase incentives.

3. Potential Customers
Use personalized promotions,
cross-selling and product recommendations.

4. Low-Engagement Customers
Use re-engagement campaigns,
limited-time offers and targeted discounts.


METHODOLOGY
-----------

1. Data cleaning
2. Feature selection
3. Standardization
4. Elbow Method
5. Silhouette Score
6. K-Means clustering
7. PCA visualization
8. Segment profiling
9. Business recommendations
"""


(
    OUTPUT_DIR /
    "business_insights.txt"
).write_text(
    insights,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 19. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 60)

print(
    "CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    f"Customers analysed : {len(df)}"
)

print(
    f"Optimal clusters   : {best_k}"
)

print(
    f"Silhouette Score    : {best_silhouette:.3f}"
)

print(
    "\nSegment Distribution:"
)

print(
    df["Segment"]
    .value_counts()
)

print(
    "\nAll charts and CSV files are saved in:"
)

print(
    OUTPUT_DIR
)

print("=" * 60)