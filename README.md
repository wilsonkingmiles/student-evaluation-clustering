# Student Evaluation Clustering

Unsupervised analysis of 5,820 student evaluation records using K-means clustering, PCA, cluster profiling, and distance-based anomaly detection.

## What this project demonstrates
- Feature engineering from survey data
- Standardization
- K-means clustering
- Elbow and validation-metric comparison
- Silhouette, Calinski-Harabasz, and Davies-Bouldin evaluation
- PCA visualization
- Cluster profiling
- Distance-to-centroid anomaly analysis

## Modeling decision
Several values of `k` were compared. Although `k=2` produced the strongest silhouette score in the evaluated range, **k=3** was selected to provide more interpretable low-, moderate-, and high-evaluation profiles. That choice is documented as an analytical tradeoff rather than presented as the only valid answer.

## Repository structure
```text
src/student_evaluation_clustering.py
data/README.md
results/   # generated when the script runs
```

## Run locally
Install the requirements, place the dataset in the project root as `turkiye-student.csv`, and run:

`python src/student_evaluation_clustering.py`

## Portfolio note
This is a cleaned presentation of academic machine-learning work. Clusters are intended for exploratory quality improvement, not punitive evaluation of instructors or students.
