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
from typing import Union, List
import time
from functools import wraps
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_mode(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get('DEBUG_MODE') == '1':
            logger.info(f"Starting {func.__name__}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

cache = Cache("embedding_cache")

@debug_mode
def get_embedding(text: Union[str, List[str]], model_name: str) -> Union[List[float], List[List[float]]]:
    # Check if input is a single string or a list of strings
    if isinstance(text, str):
        text = [text]
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Use cache for embeddings
    cached_embeddings = []
    texts_to_encode = []
    for t in text:
        cache_key = f"{model_name}:{t}"
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            cached_embeddings.append(cached_embedding)
        else:
            texts_to_encode.append(t)
    
    if texts_to_encode:
        new_embeddings = model.encode(texts_to_encode)
        for t, embedding in zip(texts_to_encode, new_embeddings):
            cache_key = f"{model_name}:{t}"
            cache.set(cache_key, embedding.tolist())
        cached_embeddings.extend(new_embeddings.tolist())
    
    # If input was a single string, return a single embedding
    if len(text) == 1:
        return cached_embeddings[0]
    
    # Otherwise, return the list of embeddings
    return cached_embeddings

@debug_mode
def perform_clustering(sentences, model_name, algorithm='dbscan', n_clusters=None, eps=0.5, min_samples=2):
    embeddings = np.array(get_embedding(sentences, model_name))
    if algorithm.lower() == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    elif algorithm.lower() == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans clustering")
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    else:
        raise ValueError("Unsupported clustering algorithm. Choose 'dbscan' or 'kmeans'.")
    labels = clustering.labels_
    return embeddings, labels

@debug_mode
def create_semantic_histogram(sentences, embeddings, labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Sort the labels and counts by frequency (descending order)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sorted_indices]
    sorted_counts = counts[sorted_indices]

    cluster_sentences = {label: [] for label in unique_labels}
    for sentence, label in zip(sentences, labels):
        cluster_sentences[label].append(sentence)

    hover_text = []
    x_labels = []
    for i, label in enumerate(sorted_labels):
        text = f"Cluster {label}<br>"
        text += "<br>".join(cluster_sentences[label][:10])  # First 10 sentences
        if len(cluster_sentences[label]) > 10:
            text += "<br>..."
        hover_text.append(text)
        x_labels.append(f"Cluster {i+1}")

    fig = go.Figure(data=[go.Bar(
        x=x_labels, 
        y=sorted_counts,
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

    return fig, evaluation_metrics

@debug_mode
def hyperparameter_sweep(sentences, model_name, n_clusters_range=(5, 25), eps_range=(0.1, 1.0, 0.1), min_samples_range=(2, 10)):
    embeddings = np.array(get_embedding(sentences, model_name))
    
    results = []
    best_config = None
    best_score = -1
    
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
        
        if sil_score > best_score:
            best_score = sil_score
            best_config = {'algorithm': 'KMeans', 'n_clusters': n_clusters}
    
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
                
                if sil_score > best_score:
                    best_score = sil_score
                    best_config = {'algorithm': 'DBSCAN', 'eps': eps, 'min_samples': min_samples}
    
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
    #fig.show()
    
    return results, fig, best_config

@debug_mode
def main(input_file, model_name, run_sweep=False, generate_summary=False, algorithm='dbscan', n_clusters=None, eps=0.5, min_samples=2):
    start_time = time.time()

    with open(input_file, 'r') as f:
        sentences = f.read().splitlines()
    
    logger.info(f"Loaded {len(sentences)} sentences")

    if run_sweep:
        results, sweep_fig, best_config = hyperparameter_sweep(sentences, model_name)
        if generate_summary:
            sweep_report_file = Path(f"./output/reports/{Path(input_file).stem}_sweep_report.html")
            generate_sweep_report(sentences, sweep_fig, results, sweep_report_file)
            logger.info(f"Hyperparameter sweep report generated: {sweep_report_file}")
        
        # Use the best configuration for clustering
        if best_config['algorithm'] == 'KMeans':
            embeddings, labels = perform_clustering(sentences, model_name, algorithm='kmeans', n_clusters=best_config['n_clusters'])
            algorithm = 'kmeans'
            n_clusters = best_config['n_clusters']
        else:  # DBSCAN
            embeddings, labels = perform_clustering(sentences, model_name, algorithm='dbscan', eps=best_config['eps'], min_samples=best_config['min_samples'])
            algorithm = 'dbscan'
            eps = best_config['eps']
            min_samples = best_config['min_samples']
    else:
        embeddings, labels = perform_clustering(sentences, model_name, algorithm=algorithm, n_clusters=n_clusters, eps=eps, min_samples=min_samples)

    fig, evaluation_metrics = create_semantic_histogram(sentences, embeddings, labels)

    clusters = {}
    for sentence, label in zip(sentences, labels):
        label = int(label)  # Convert numpy.int64 to regular Python int
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentence)

    # Create a string with algorithm and hyperparameters for the filename
    if algorithm == 'kmeans':
        algo_params = f"kmeans_n{n_clusters}"
    else:  # dbscan
        algo_params = f"dbscan_eps{eps}_min{min_samples}"

    output_file = Path(f"./output/clusters/{Path(input_file).stem}_{model_name}_{algo_params}_clusters.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)

    logger.info(f"Clusters saved to {output_file}")

    for cluster_id, cluster_sentences in clusters.items():
        logger.info(f"\nCluster {cluster_id}:")
        for sentence in cluster_sentences[:5]:  # Print first 5 sentences of each cluster
            logger.info(f" - {sentence}")
        if len(cluster_sentences) > 5:
            logger.info(f" ... and {len(cluster_sentences) - 5} more")

    report_file = Path(f"./output/reports/{Path(input_file).stem}_{model_name}_{algo_params}_report.html")
    
    if generate_summary:
        from summarize import summarize_sentences
        cluster_summaries = {cluster_id: summarize_sentences(cluster_sentences) for cluster_id, cluster_sentences in clusters.items() if cluster_id != -1}
    else:
        cluster_summaries = {cluster_id: "" for cluster_id in clusters.keys() if cluster_id != -1}
    
    # Add algorithm and hyperparameters to the report
    algorithm_info = {
        'algorithm': algorithm,
        'n_clusters': n_clusters if algorithm == 'kmeans' else None,
        'eps': eps if algorithm == 'dbscan' else None,
        'min_samples': min_samples if algorithm == 'dbscan' else None
    }
    
    generate_report(sentences, labels, embeddings, fig, clusters, evaluation_metrics, report_file, 
                    cluster_summaries=cluster_summaries, sweep_fig=sweep_fig if run_sweep else None,
                    algorithm_info=algorithm_info)
    logger.info(f"Report generated: {report_file}")
    # Save clustering results for Streamlit app
    joblib.dump(clusters, 'output/clusters.joblib')
    joblib.dump(embeddings, 'output/embeddings.joblib')
    joblib.dump(sentences, 'output/sentences.joblib')

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create semantic clusters from input sentences.")
    parser.add_argument("input_file", help="Path to the input file containing sentences (one per line)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--summary", action="store_true", default=False, help="Generate summary reports")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--algorithm", choices=['dbscan', 'kmeans'], default='dbscan', help="Clustering algorithm to use")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters for KMeans")
    parser.add_argument("--eps", type=float, default=0.5, help="Epsilon value for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples for DBSCAN")
    args = parser.parse_args()

    if args.debug:
        os.environ['DEBUG_MODE'] = '1'
        logger.setLevel(logging.DEBUG)

    main(args.input_file, args.model, run_sweep=args.sweep, generate_summary=args.summary,
         algorithm=args.algorithm, n_clusters=args.n_clusters, eps=args.eps, min_samples=args.min_samples)