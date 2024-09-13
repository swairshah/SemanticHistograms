import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist
from sentences import data as sentences
import argparse
import json
from pathlib import Path
from diskcache import Cache
from report import generate_report, generate_sweep_report
import joblib

cache = Cache("embedding_cache")

def get_embedding(sentence, model_name):
    # Create a unique cache key
    cache_key = f"{sentence}:{model_name}"

    # Check if the embedding is already cached
    if cache_key in cache:
        return cache[cache_key]

    # If not cached, compute the embedding
    model = SentenceTransformer(model_name)
    embedding = model.encode(sentence)

    # Cache the result
    cache[cache_key] = embedding

    return embedding

def create_semantic_histogram(sentences, model_name, eps=0.5, min_samples=2):
    embeddings = np.array([get_embedding(sentence, model_name) for sentence in sentences])

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)

    cluster_sentences = {label: [] for label in unique_labels}
    for sentence, label in zip(sentences, labels):
        cluster_sentences[label].append(sentence)

    hover_text = []
    for label in unique_labels:
        text = f"Cluster {label}<br>"
        text += "<br>".join(cluster_sentences[label][:10])  # First 10 sentences
        if len(cluster_sentences[label]) > 10:
            text += "<br>..."
        hover_text.append(text)
    fig = go.Figure(data=[go.Bar(
        x=unique_labels, 
        y=counts,
        hovertext=hover_text,
        hoverinfo='text',
        marker_color='rgba(0, 255, 255, 0.8)'  # Cyan color with some transparency
    )])
    fig.update_layout(
        title='Semantic Frequency Histogram',
        xaxis_title='Cluster',
        yaxis_title='Number of Sentences',
        bargap=0.2,
        paper_bgcolor='rgba(0,0,0,0.9)',  # Dark background
        plot_bgcolor='rgba(0,0,0,0.9)',   # Dark plot area
        font=dict(color='white'),         # White text
        title_font_color='white',         # White title
        xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),  # Lighter grid lines
        yaxis=dict(gridcolor='rgba(255,255,255,0.2)')   # Lighter grid lines
    )

    sil_score = silhouette_score(embeddings, labels, metric='cosine')
    
    intra_distances = []
    for cluster_id in unique_labels:
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            intra_distances.append(np.mean(pdist(cluster_points, metric='cosine')))
    avg_intra_distance = np.mean(intra_distances)

    centroid_embeddings = [np.mean(embeddings[labels == cluster_id], axis=0) for cluster_id in unique_labels]
    inter_cluster_distances = pdist(centroid_embeddings, metric='cosine')
    avg_inter_distance = np.mean(inter_cluster_distances)

    evaluation_metrics = {
        'silhouette_score': sil_score,
        'avg_intra_distance': avg_intra_distance,
        'avg_inter_distance': avg_inter_distance
    }

    return fig, labels, embeddings, evaluation_metrics

def hyperparameter_sweep(sentences, model_name, n_clusters_range=(2, 20), eps_range=(0.1, 1.0, 0.1), min_samples_range=(2, 10)):
    embeddings = SentenceTransformer(model_name).encode(sentences)
    
    results = []
    
    # KMeans sweep
    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        sil_score = silhouette_score(embeddings, labels, metric='cosine')
        intra_cluster_distance = kmeans.inertia_
        
        results.append({
            'algorithm': 'KMeans',
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'intra_cluster_distance': intra_cluster_distance
        })
    
    # DBSCAN sweep
    for eps in np.arange(eps_range[0], eps_range[1], eps_range[2]):
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(embeddings)
            
            if len(set(labels)) > 1:  # Ensure we have at least 2 clusters
                sil_score = silhouette_score(embeddings, labels, metric='cosine')
                
                results.append({
                    'algorithm': 'DBSCAN',
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'silhouette_score': sil_score
                })
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("KMeans: Silhouette Score", "KMeans: Intra-cluster Distance",
                                                        "DBSCAN: Silhouette Score", "DBSCAN: Number of Clusters"))
    
    kmeans_results = [r for r in results if r['algorithm'] == 'KMeans']
    dbscan_results = [r for r in results if r['algorithm'] == 'DBSCAN']
    
    fig.add_trace(go.Scatter(x=[r['n_clusters'] for r in kmeans_results],
                             y=[r['silhouette_score'] for r in kmeans_results],
                             mode='lines+markers',
                             name='KMeans Silhouette'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[r['n_clusters'] for r in kmeans_results],
                             y=[r['intra_cluster_distance'] for r in kmeans_results],
                             mode='lines+markers',
                             name='KMeans Intra-cluster Distance'),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=[r['eps'] for r in dbscan_results],
                             y=[r['silhouette_score'] for r in dbscan_results],
                             mode='markers',
                             marker=dict(size=5, color=[r['min_samples'] for r in dbscan_results], colorscale='Viridis'),
                             name='DBSCAN Silhouette'),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=[r['eps'] for r in dbscan_results],
                             y=[r['n_clusters'] for r in dbscan_results],
                             mode='markers',
                             marker=dict(size=5, color=[r['min_samples'] for r in dbscan_results], colorscale='Viridis'),
                             name='DBSCAN Clusters'),
                  row=2, col=2)
    
    fig.update_layout(height=800, width=1000, title_text="Clustering Hyperparameter Sweep")
    fig.show()
    
    return results, fig


def main(input_file, model_name, run_sweep=False):
    with open(input_file, 'r') as f:
        sentences = f.read().splitlines()

    if run_sweep:
        results, sweep_fig = hyperparameter_sweep(sentences, model_name)
        sweep_fig.show()
        sweep_report_file = Path(f"./output/reports/{Path(input_file).stem}_sweep_report.html")
        generate_sweep_report(sentences, sweep_fig, results, sweep_report_file)
        print(f"Hyperparameter sweep report generated: {sweep_report_file}")

    fig, labels, embeddings, evaluation_metrics = create_semantic_histogram(sentences, model_name)
    fig.show()

    clusters = {}
    for sentence, label in zip(sentences, labels):
        label = int(label)  # Convert numpy.int64 to regular Python int
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentence)

    output_file = Path(f".output/clusters/{Path(input_file).stem}_clusters.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)

    print(f"Clusters saved to {output_file}")

    # Print clusters (optional, you can remove this if not needed)
    for cluster_id, cluster_sentences in clusters.items():
        print(f"\nCluster {cluster_id}:")
        for sentence in cluster_sentences:
            print(f" - {sentence}")

    # Generate the HTML report
    report_file = Path(f"./output/reports/{Path(input_file).stem}_report.html")
    generate_report(sentences, labels, embeddings, fig, clusters, evaluation_metrics, report_file)
    print(f"Report generated: {report_file}")

    # Save clustering results for Streamlit app
    joblib.dump(clusters, 'output/clusters.joblib')
    joblib.dump(embeddings, 'output/embeddings.joblib')
    joblib.dump(sentences, 'output/sentences.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create semantic clusters from input sentences.")
    parser.add_argument("input_file", help="Path to the input file containing sentences (one per line)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")

    args = parser.parse_args()

    main(args.input_file, args.model, run_sweep=args.sweep)