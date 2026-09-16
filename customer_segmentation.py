"""
Customer Segmentation using K-Means Clustering
==============================================

Segments customers based on RFM (Recency, Frequency, Monetary) behaviour
and demographic features (age, income), then visualizes the resulting
segments and their key characteristics.

Usage
-----
    python customer_segmentation.py                    # uses generated sample data
    python customer_segmentation.py --csv my_data.csv  # uses your own dataset

Expected CSV columns (header row required):
    customer_id, age, income, recency, frequency, monetary

Outputs
-------
    segmented_customers.csv   - original data with an assigned segment per customer
    segment_summary.csv       - average profile of each segment
    elbow_plot.png            - elbow curve used to choose K
    segment_plots.png         - scatter, size, revenue and profile visualisations

Requirements
------------
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # write plots to file instead of opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURES = ["recency", "frequency", "monetary", "age", "income"]

sns.set_style("whitegrid")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def generate_sample_data(n_per_group=80):
    """Create a realistic synthetic customer dataset with latent groups."""
    rng = np.random.default_rng(RANDOM_STATE)

    groups = [
        # (age, income, recency, frequency, monetary) as (mean, std)
        dict(n=70, age=(42, 9), income=(1_450_000, 250_000),
             recency=(12, 6), frequency=(22, 5), monetary=(185_000, 35_000)),
        dict(n=95, age=(31, 7), income=(720_000, 160_000),
             recency=(38, 14), frequency=(11, 3), monetary=(62_000, 15_000)),
        dict(n=85, age=(27, 6), income=(420_000, 110_000),
             recency=(95, 30), frequency=(4, 2), monetary=(18_000, 7_000)),
        dict(n=60, age=(52, 10), income=(980_000, 220_000),
             recency=(210, 55), frequency=(3, 2), monetary=(27_000, 11_000)),
    ]

    frames = []
    for g in groups:
        n = g["n"]
        frames.append(pd.DataFrame({
            "age": rng.normal(*g["age"], n).clip(18, 80).round().astype(int),
            "income": rng.normal(*g["income"], n).clip(120_000, None).round(-3).astype(int),
            "recency": rng.normal(*g["recency"], n).clip(1, None).round().astype(int),
            "frequency": rng.normal(*g["frequency"], n).clip(1, None).round().astype(int),
            "monetary": rng.normal(*g["monetary"], n).clip(1_500, None).round(-2).astype(int),
        }))

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    df.insert(0, "customer_id", ["C" + str(1001 + i) for i in range(len(df))])
    return df.reset_index(drop=True)


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in ["recency", "frequency", "monetary"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for col in ["age", "income"]:
            if col not in df.columns:
                df[col] = 0
        if "customer_id" not in df.columns:
            df.insert(0, "customer_id", ["C" + str(i) for i in range(1, len(df) + 1)])
        print(f"Loaded {len(df)} customers from {csv_path}")
        return df

    df = generate_sample_data()
    print(f"Generated sample dataset with {len(df)} customers")
    return df


# ----------------------------------------------------------------------
# 2. Explore
# ----------------------------------------------------------------------
def explore(df):
    print("\n--- Dataset preview ---")
    print(df.head())
    print("\n--- Summary statistics ---")
    print(df[FEATURES].describe().round(1))
    print("\n--- Missing values ---")
    print(df.isnull().sum())


# ----------------------------------------------------------------------
# 3. Choose K (elbow method + silhouette score)
# ----------------------------------------------------------------------
def find_optimal_k(X_scaled, k_range=range(2, 9), out="elbow_plot.png"):
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(silhouettes))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(k_range), inertias, "o-", color=PALETTE[0], linewidth=2)
    axes[0].set_title("Elbow method")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")

    axes[1].plot(list(k_range), silhouettes, "o-", color=PALETTE[2], linewidth=2)
    axes[1].axvline(best_k, color=PALETTE[1], linestyle="--", label=f"Best K = {best_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()

    print(f"\nSaved elbow/silhouette plot to {out}")
    print(f"Suggested K based on silhouette score: {best_k}")
    return best_k


# ----------------------------------------------------------------------
# 4. Cluster
# ----------------------------------------------------------------------
def run_kmeans(df, k):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df["cluster"])
    print(f"\nK-Means fitted with K={k} | silhouette score = {score:.3f}")
    return df, X_scaled


# ----------------------------------------------------------------------
# 5. Interpret and name segments
# ----------------------------------------------------------------------
def name_segments(df):
    overall = df[FEATURES].mean()
    profile = df.groupby("cluster")[FEATURES].mean()

    names, actions = {}, {}
    for cid, row in profile.iterrows():
        high_mon = row["monetary"] > overall["monetary"] * 1.3
        low_mon = row["monetary"] < overall["monetary"] * 0.7
        high_freq = row["frequency"] > overall["frequency"] * 1.3
        high_rec = row["recency"] > overall["recency"] * 1.4  # long gap since last purchase

        if high_mon and high_freq:
            names[cid] = "Champions"
            actions[cid] = ("Most valuable customers. Reward with loyalty perks, "
                            "early access and referral incentives.")
        elif high_rec and high_mon:
            names[cid] = "Can't Lose Them"
            actions[cid] = ("High spenders who have gone quiet. Prioritise personal "
                            "outreach and exclusive offers before they churn.")
        elif high_rec:
            names[cid] = "At-Risk / Lapsed"
            actions[cid] = ("Long inactive. Run win-back campaigns with targeted "
                            "discounts and re-engagement emails.")
        elif low_mon:
            names[cid] = "New / Low Spenders"
            actions[cid] = ("Recent but low-value buyers. Nurture with onboarding "
                            "content, bundles and first-repeat-purchase offers.")
        else:
            names[cid] = "Loyal Regulars"
            actions[cid] = ("Consistent, dependable buyers. Upsell complementary "
                            "products and grow basket size.")

    # keep names unique
    seen = {}
    for cid in names:
        base = names[cid]
        if base in seen:
            seen[base] += 1
            names[cid] = f"{base} {seen[base]}"
        else:
            seen[base] = 1

    df["segment"] = df["cluster"].map(names)

    summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        avg_age=("age", "mean"),
        avg_income=("income", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1)
    summary["revenue_share_%"] = (summary["total_revenue"] / summary["total_revenue"].sum() * 100).round(1)
    summary["recommended_action"] = [actions[c] for c in
                                     df.groupby("segment")["cluster"].first().loc[summary.index]]

    print("\n--- Segment summary ---")
    print(summary.drop(columns="recommended_action").to_string())
    print("\n--- Recommended actions ---")
    for seg, action in summary["recommended_action"].items():
        print(f"  {seg}: {action}")

    return df, summary


# ----------------------------------------------------------------------
# 6. Visualise
# ----------------------------------------------------------------------
def visualize(df, X_scaled, out="segment_plots.png"):
    segments = df["segment"].unique()
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(segments))}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Frequency vs Monetary
    for seg in sorted(segments):
        sub = df[df["segment"] == seg]
        axes[0, 0].scatter(sub["frequency"], sub["monetary"], s=28,
                           alpha=0.75, label=seg, color=colors[seg])
    axes[0, 0].set_xlabel("Frequency (orders)")
    axes[0, 0].set_ylabel("Monetary (total spend)")
    axes[0, 0].set_title("Segments: frequency vs monetary")
    axes[0, 0].legend(fontsize=8)

    # (b) PCA projection
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    for seg in sorted(segments):
        mask = (df["segment"] == seg).values
        axes[0, 1].scatter(coords[mask, 0], coords[mask, 1], s=28,
                           alpha=0.75, label=seg, color=colors[seg])
    var = pca.explained_variance_ratio_.sum() * 100
    axes[0, 1].set_xlabel("Principal component 1")
    axes[0, 1].set_ylabel("Principal component 2")
    axes[0, 1].set_title(f"PCA projection ({var:.0f}% variance explained)")
    axes[0, 1].legend(fontsize=8)

    # (c) Segment size
    sizes = df["segment"].value_counts().sort_index()
    axes[1, 0].bar(sizes.index, sizes.values, color=[colors[s] for s in sizes.index])
    axes[1, 0].set_title("Customers per segment")
    axes[1, 0].set_ylabel("Customers")
    axes[1, 0].tick_params(axis="x", rotation=20, labelsize=9)

    # (d) Revenue contribution
    rev = df.groupby("segment")["monetary"].sum().sort_index()
    axes[1, 1].bar(rev.index, rev.values, color=[colors[s] for s in rev.index])
    axes[1, 1].set_title("Total revenue by segment")
    axes[1, 1].set_ylabel("Revenue")
    axes[1, 1].tick_params(axis="x", rotation=20, labelsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\nSaved segment visualisations to {out}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Customer segmentation with K-Means")
    parser.add_argument("--csv", help="Path to your customer CSV file")
    parser.add_argument("--k", type=int, help="Number of clusters (skips auto-selection)")
    args = parser.parse_args()

    df = load_data(args.csv)
    explore(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    k = args.k or find_optimal_k(X_scaled)
    df, X_scaled = run_kmeans(df, k)
    df, summary = name_segments(df)
    visualize(df, X_scaled)

    df.to_csv("segmented_customers.csv", index=False)
    summary.to_csv("segment_summary.csv")
    print("\nSaved segmented_customers.csv and segment_summary.csv")
    print("Done.")


if __name__ == "__main__":
    main()
