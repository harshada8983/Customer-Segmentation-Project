<h1 align="center">Customer Segmentation Project</h1>

<p align="center">
  Segmenting customers by purchase behaviour and demographics using K-Means clustering
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=flat&logo=chart.js&logoColor=white" alt="Chart.js">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white" alt="HTML5">
</p>

---

## Overview

This project segments a customer base into distinct groups using **K-Means clustering** on RFM (Recency, Frequency, Monetary) behaviour combined with demographic features (age, income). Each segment is profiled, named, and paired with a recommended marketing action.

The project has two parts:

| Part | File | What it does |
|------|------|--------------|
| **Analysis** | `customer_segmentation.py` | Full clustering pipeline in Python with scikit-learn — scaling, optimal-K selection, K-Means, PCA, and matplotlib visualisations |
| **Dashboard** | `customer_segmentation_dashboard.html` | Interactive browser dashboard — clustering runs live, with adjustable K, switchable axes, and CSV import |

**Live dashboard:** https://yourusername.github.io/customer-segmentation/customer_segmentation_dashboard.html
*(replace `yourusername` with your GitHub username after enabling GitHub Pages)*

---

## Features

**Clustering & analysis**
- Feature standardisation with `StandardScaler` before clustering
- Optimal cluster count chosen automatically via the **elbow method** and **silhouette score**
- K-Means clustering with `k-means++` initialisation
- **PCA** projection to visualise high-dimensional clusters in 2D

**Segment insights**
- Automatic segment naming based on RFM profile — Champions, Loyal Regulars, At-Risk / Lapsed, Can't Lose Them, New / Low Spenders
- Per-segment averages: recency, order frequency, spend, age, income
- Revenue contribution and share per segment
- A recommended marketing action for each segment

**Interactive dashboard**
- KPI cards: total customers, segments found, average customer value, largest segment
- Scatter plot with switchable X and Y features
- Donut chart of segment sizes, radar chart comparing segment profiles, bar chart of revenue by segment
- Adjustable number of clusters (K = 3, 4, or 5)
- Import your own CSV, or use the built-in sample dataset

---

## Repository contents

```
├── customer_segmentation.py              # scikit-learn clustering pipeline
├── customer_segmentation_dashboard.html  # interactive browser dashboard
├── sample_customers.csv                  # sample dataset (310 customers)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Getting started

### Interactive dashboard (no installation)

Download `customer_segmentation_dashboard.html` and open it in any browser — or visit the live link above. Use the controls to change K, switch the scatter plot axes, or import your own CSV.

### Python analysis

```bash
pip install -r requirements.txt

# Run on the built-in sample dataset
python customer_segmentation.py

# Run on your own data
python customer_segmentation.py --csv sample_customers.csv

# Force a specific number of clusters
python customer_segmentation.py --k 4
```

**Generated outputs:**

| File | Contents |
|------|----------|
| `segmented_customers.csv` | Every customer with their assigned segment |
| `segment_summary.csv` | Average profile and revenue share per segment |
| `elbow_plot.png` | Elbow curve and silhouette scores used to pick K |
| `segment_plots.png` | Scatter, PCA projection, segment sizes, revenue by segment |

---

## Data format

Both the script and the dashboard accept a CSV with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `customer_id` | optional | Unique customer identifier |
| `age` | optional | Customer age in years |
| `income` | optional | Annual income |
| `recency` | **required** | Days since the customer's last purchase |
| `frequency` | **required** | Total number of orders placed |
| `monetary` | **required** | Total amount spent |

Example:

```csv
customer_id,age,income,recency,frequency,monetary
C1001,60,956000,238,1,33800
C1002,34,1373000,13,19,181400
C1003,55,1153000,10,24,195900
```

---

## Methodology

1. **Load and explore** — inspect distributions, check for missing values
2. **Scale features** — standardise all features so no single variable (like income) dominates the distance calculation
3. **Select K** — run K-Means across a range of K values, compare inertia (elbow) and silhouette scores
4. **Cluster** — fit K-Means with the chosen K and assign each customer to a segment
5. **Profile and name** — compute each segment's average behaviour and label it against overall averages
6. **Visualise and act** — plot the segments and attach a targeted marketing recommendation to each

---

## Sample results

On the included dataset, K-Means identifies clearly separated segments (silhouette score ≈ 0.54):

| Segment | Customers | Avg. orders | Avg. spend | Revenue share |
|---------|-----------|-------------|------------|---------------|
| Champions | 70 | 22.1 | ₹1,86,221 | 59.3% |
| New / Low Spenders | 181 | 7.4 | ₹41,201 | 33.9% |
| At-Risk / Lapsed | 59 | 3.2 | ₹25,134 | 6.7% |

**Key insight:** roughly 23% of customers (Champions) generate close to 60% of total revenue — a strong case for prioritising retention spend on this group.

---

## What this project demonstrates

- Unsupervised machine learning (K-Means clustering)
- RFM analysis for customer behaviour
- Feature scaling and dimensionality reduction (PCA)
- Cluster validation using elbow method and silhouette score
- Data visualisation with matplotlib, seaborn, and Chart.js
- Translating analytical output into targeted business recommendations

---

## Tech stack

**Python:** pandas, NumPy, scikit-learn, matplotlib, seaborn
**Dashboard:** HTML, CSS, JavaScript, Chart.js

---

Built as a virtual internship project.
