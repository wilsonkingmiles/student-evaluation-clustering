"""
K-Means Clustering Analysis: Turkiye Student Evaluation Dataset

This script performs exploratory data analysis, preprocessing, k-means clustering,
cluster evaluation, and anomaly detection for the UCI Turkiye Student Evaluation dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Reproducibility and output locations
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path('.')
DATA_CANDIDATES = [
    BASE_DIR / 'turkiye-student(1).csv',
    BASE_DIR / 'turkiye-student.csv',
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError('Could not locate the Turkiye student CSV file. Place it in the same folder as this script.')
OUTPUT_DIR = BASE_DIR.parent / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load and rename dataset
raw = pd.read_csv(DATA_PATH)
question_cols = [f'Q{i}' for i in range(1, 29)]
renamed_cols = ['instructor', 'repeat', 'attendance', 'difficulty'] + question_cols + ['course_code']
raw.columns = renamed_cols

# 2. Feature engineering for interpretation
raw['overall_evaluation_mean'] = raw[question_cols].mean(axis=1)
raw['course_structure_mean'] = raw[question_cols[:12]].mean(axis=1)
raw['instructor_engagement_mean'] = raw[question_cols[12:]].mean(axis=1)

# 3. Exploratory outputs
raw.head(10).to_csv(OUTPUT_DIR / 'dataset_head.csv', index=False)
raw.describe().T.to_csv(OUTPUT_DIR / 'numeric_summary.csv')
raw.isna().sum().to_csv(OUTPUT_DIR / 'missing_values.csv', header=['missing_count'])
raw.nunique().to_csv(OUTPUT_DIR / 'unique_values.csv', header=['unique_values'])
raw[['instructor', 'course_code']].value_counts().sort_index().to_csv(OUTPUT_DIR / 'instructor_course_counts.csv')

# 4. Select clustering features
features = ['repeat', 'attendance', 'difficulty'] + question_cols
X = raw[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Tune number of clusters using elbow and validation metrics
metrics = []
models = {}
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, algorithm='lloyd', max_iter=300)
    labels = model.fit_predict(X_scaled)
    models[k] = model
    metrics.append({
        'k': k,
        'inertia_within_cluster_sse': model.inertia_,
        'silhouette_score_sample_1000': silhouette_score(X_scaled, labels, sample_size=1000, random_state=RANDOM_STATE),
        'calinski_harabasz_score': calinski_harabasz_score(X_scaled, labels),
        'davies_bouldin_score': davies_bouldin_score(X_scaled, labels)
    })
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUTPUT_DIR / 'kmeans_evaluation_metrics.csv', index=False)

# Based on the elbow curve and interpretability, select k=3.
FINAL_K = 3
final_model = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=20, algorithm='lloyd', max_iter=300)
raw['cluster'] = final_model.fit_predict(X_scaled)

# Name clusters using their overall evaluation means.
cluster_order = raw.groupby('cluster')['overall_evaluation_mean'].mean().sort_values().index.tolist()
cluster_labels = {
    cluster_order[0]: 'Low evaluation / support-needed group',
    cluster_order[1]: 'Moderate evaluation / mixed-feedback group',
    cluster_order[2]: 'High evaluation / positive-feedback group'
}
raw['cluster_label'] = raw['cluster'].map(cluster_labels)

# 6. Cluster profiles
cluster_profile = raw.groupby(['cluster', 'cluster_label']).agg(
    size=('cluster', 'size'),
    repeat_mean=('repeat', 'mean'),
    attendance_mean=('attendance', 'mean'),
    difficulty_mean=('difficulty', 'mean'),
    overall_evaluation_mean=('overall_evaluation_mean', 'mean'),
    course_structure_mean=('course_structure_mean', 'mean'),
    instructor_engagement_mean=('instructor_engagement_mean', 'mean')
).reset_index()
cluster_profile['percentage'] = cluster_profile['size'] / len(raw) * 100
cluster_profile = cluster_profile[['cluster', 'cluster_label', 'size', 'percentage', 'repeat_mean', 'attendance_mean', 'difficulty_mean', 'overall_evaluation_mean', 'course_structure_mean', 'instructor_engagement_mean']]
cluster_profile.to_csv(OUTPUT_DIR / 'cluster_profile.csv', index=False)

question_profile = raw.groupby(['cluster', 'cluster_label'])[question_cols].mean().reset_index()
question_profile.to_csv(OUTPUT_DIR / 'cluster_question_means.csv', index=False)
raw.to_csv(OUTPUT_DIR / 'turkiye_student_with_clusters.csv', index=False)

# 7. PCA for visualization only
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    'PC1': pca_coords[:, 0],
    'PC2': pca_coords[:, 1],
    'cluster': raw['cluster'],
    'cluster_label': raw['cluster_label'],
    'overall_evaluation_mean': raw['overall_evaluation_mean']
})
pca_df.to_csv(OUTPUT_DIR / 'pca_coordinates.csv', index=False)
with open(OUTPUT_DIR / 'pca_variance.txt', 'w') as f:
    f.write(f"PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}\n")
    f.write(f"PC2 explained variance ratio: {pca.explained_variance_ratio_[1]:.4f}\n")
    f.write(f"Cumulative explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}\n")

# 8. Anomaly detection using distance to assigned centroid
centroid_distances = cdist(X_scaled, final_model.cluster_centers_)
raw['distance_to_centroid'] = centroid_distances[np.arange(len(raw)), raw['cluster']]
anomaly_threshold = np.percentile(raw['distance_to_centroid'], 95)
raw['anomaly_flag'] = raw['distance_to_centroid'] >= anomaly_threshold
anomaly_summary = raw.groupby(['cluster', 'cluster_label', 'anomaly_flag']).size().reset_index(name='count')
anomaly_summary.to_csv(OUTPUT_DIR / 'anomaly_summary.csv', index=False)
raw.nlargest(25, 'distance_to_centroid')[['instructor', 'course_code', 'repeat', 'attendance', 'difficulty', 'overall_evaluation_mean', 'cluster', 'cluster_label', 'distance_to_centroid']].to_csv(OUTPUT_DIR / 'top_25_anomalies.csv', index=False)
raw.to_csv(OUTPUT_DIR / 'turkiye_student_with_clusters_and_anomalies.csv', index=False)

# 9. Visualizations
plt.figure(figsize=(7, 4.5))
raw['course_code'].value_counts().sort_index().plot(kind='bar')
plt.title('Course code distribution')
plt.xlabel('Course code')
plt.ylabel('Number of evaluations')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'course_distribution.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.hist(raw['overall_evaluation_mean'], bins=25, edgecolor='black')
plt.title('Distribution of overall student evaluation mean')
plt.xlabel('Average rating across Q1-Q28')
plt.ylabel('Number of evaluations')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'hist_overall_evaluation_mean.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.boxplot([raw.loc[raw['difficulty'] == value, 'overall_evaluation_mean'] for value in sorted(raw['difficulty'].unique())], tick_labels=sorted(raw['difficulty'].unique()))
plt.title('Overall evaluation by perceived difficulty')
plt.xlabel('Difficulty code')
plt.ylabel('Overall evaluation mean')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'box_overall_by_difficulty.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.boxplot([raw.loc[raw['attendance'] == value, 'overall_evaluation_mean'] for value in sorted(raw['attendance'].unique())], tick_labels=sorted(raw['attendance'].unique()))
plt.title('Overall evaluation by attendance level')
plt.xlabel('Attendance code')
plt.ylabel('Overall evaluation mean')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'box_overall_by_attendance.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(metrics_df['k'], metrics_df['inertia_within_cluster_sse'], marker='o')
plt.title('Elbow method: within-cluster SSE by k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster SSE / inertia')
plt.xticks(metrics_df['k'])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'elbow_curve.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(metrics_df['k'], metrics_df['silhouette_score_sample_1000'], marker='o')
plt.title('Silhouette score by k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.xticks(metrics_df['k'])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'silhouette_curve.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
for cluster_value, group in pca_df.groupby('cluster_label'):
    plt.scatter(group['PC1'], group['PC2'], label=cluster_value, alpha=0.55, s=12)
plt.title('K-means clusters visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_cluster_scatter.png', dpi=200)
plt.close()

plt.figure(figsize=(8, 4.8))
profile_plot = cluster_profile.set_index('cluster_label')[['overall_evaluation_mean', 'course_structure_mean', 'instructor_engagement_mean']]
profile_plot.plot(kind='bar')
plt.title('Cluster profile: average evaluation measures')
plt.xlabel('Cluster label')
plt.ylabel('Mean rating')
plt.ylim(1, 5)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_profile_bars.png', dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
question_means = raw.groupby('cluster_label')[question_cols].mean().T
question_means.plot(marker='o', linewidth=1.5)
plt.title('Mean rating by question and cluster')
plt.xlabel('Survey question')
plt.ylabel('Mean rating')
plt.xticks(range(0, 28), question_cols, rotation=90)
plt.ylim(1, 5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'question_means_by_cluster.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.hist(raw['distance_to_centroid'], bins=30, edgecolor='black')
plt.axvline(anomaly_threshold, linestyle='--')
plt.title('Distance-to-centroid distribution for anomaly detection')
plt.xlabel('Distance to assigned centroid')
plt.ylabel('Number of evaluations')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anomaly_distance_distribution.png', dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
raw['cluster_label'].value_counts().sort_index().plot(kind='bar')
plt.title('Final cluster sizes')
plt.xlabel('Cluster label')
plt.ylabel('Number of evaluations')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_size_bar.png', dpi=200)
plt.close()

# 10. Plain-text summary for quick review
with open(OUTPUT_DIR / 'analysis_summary.txt', 'w') as f:
    f.write('DATA 645 Unit 8 - K-Means Clustering Summary\n')
    f.write(f'Dataset shape: {raw.shape[0]} rows, {len(raw.columns)} columns including engineered columns.\n')
    f.write(f'Missing values: {raw[renamed_cols].isna().sum().sum()}\n')
    f.write(f'Duplicated rows in original columns: {raw[renamed_cols].duplicated().sum()}\n')
    f.write(f'Final k: {FINAL_K}\n')
    f.write(f'Anomaly threshold, 95th percentile distance: {anomaly_threshold:.4f}\n')
    f.write(f'Number of anomaly flags: {raw["anomaly_flag"].sum()}\n\n')
    f.write('K-means evaluation metrics:\n')
    f.write(metrics_df.round(4).to_string(index=False))
    f.write('\n\nCluster profile:\n')
    f.write(cluster_profile.round(3).to_string(index=False))

print('Analysis complete. Outputs saved to:', OUTPUT_DIR)
print('Selected k:', FINAL_K)
print(cluster_profile.round(3).to_string(index=False))
