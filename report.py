import plotly
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import numpy as np
from jinja2 import Template
from summarize import summarize_sentences
import plotly.io as pio

def find_outliers(cluster_embeddings, n=3):
    if len(cluster_embeddings) <= n:
        return list(range(len(cluster_embeddings)))
    
    distances = cosine_distances(cluster_embeddings)
    avg_distances = np.mean(distances, axis=1)
    outlier_indices = np.argsort(avg_distances)[-n:]
    return outlier_indices.tolist()

def get_cluster_summary(cluster_sentences, cluster_embeddings):
    if len(cluster_sentences) <= 20:
        return summarize_sentences(cluster_sentences)
    
    center = np.mean(cluster_embeddings, axis=0)
    similarities = cosine_similarity([center], cluster_embeddings)[0]
    top_20_indices = np.argsort(similarities)[-20:]
    top_20_sentences = [cluster_sentences[i] for i in top_20_indices]
    return summarize_sentences(top_20_sentences)

def generate_report(sentences, labels, embeddings, fig, clusters, evaluation_metrics, output_file):
    if clusters is None:
        raise ValueError("The 'clusters' parameter cannot be None.")

    plot_html = plotly.io.to_html(fig, full_html=False)

    outliers = {}
    cluster_summaries = {}
    for cluster_id, cluster_sentences in clusters.items():
        if cluster_id == -1:  # Skip noise cluster
            continue
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        outlier_indices = find_outliers(cluster_embeddings)
        outliers[cluster_id] = [cluster_sentences[i] for i in outlier_indices]
        
        cluster_summaries[cluster_id] = get_cluster_summary(cluster_sentences, cluster_embeddings)

    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Semantic Clustering Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2 { color: #2c3e50; }
            .metric { margin-bottom: 10px; }
            .cluster-table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
            .cluster-table th, .cluster-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .cluster-table th { background-color: #f2f2f2; }
            .outlier { color: #e74c3c; }
            .cluster-table td { vertical-align: top; }
        </style>
    </head>
    <body>
        <h1>Semantic Clustering Report</h1>
        
        <h2>Clustering Plot</h2>
        {{ plot_html|safe }}
        
        <h2>Evaluation Metrics</h2>
        {% for metric, value in evaluation_metrics.items() %}
        <div class="metric"><strong>{{ metric|replace('_', ' ')|title }}:</strong> {{ "%.4f"|format(value) }}</div>
        {% endfor %}
        
        <h2>Clusters and Outliers</h2>
        <table class="cluster-table">
            <tr>
                <th>Cluster ID</th>
                <th>Size</th>
                <th>Sample Sentences</th>
                <th>Outliers</th>
                <th>Summary</th>
            </tr>
            {% for cluster_id, sentences in clusters.items() %}
            <tr>
                <td>{{ cluster_id }}</td>
                <td>{{ sentences|length }}</td>
                <td>
                    <ul>
                    {% for sentence in sentences[:5] %}
                        <li>{{ sentence }}</li>
                    {% endfor %}
                    {% if sentences|length > 5 %}
                        <li>...</li>
                    {% endif %}
                    </ul>
                </td>
                <td>
                    <ul>
                    {% for outlier in outliers.get(cluster_id, []) %}
                        <li class="outlier">{{ outlier }}</li>
                    {% endfor %}
                    </ul>
                </td>
                <td>{{ cluster_summaries[cluster_id] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    ''')

    html_content = template.render(
        plot_html=plot_html,
        evaluation_metrics=evaluation_metrics,
        clusters=clusters,
        outliers=outliers,
        cluster_summaries=cluster_summaries  # Add this line
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_sweep_report(sentences, sweep_fig, results, output_file):
    html_content = f"""
    <html>
    <head>
        <title>Clustering Hyperparameter Sweep Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Clustering Hyperparameter Sweep Report</h1>
        <p>Number of sentences analyzed: {len(sentences)}</p>
        
        <h2>Hyperparameter Sweep Results</h2>
        <div id="sweepPlot"></div>
        
        <h2>Best Results</h2>
        <h3>KMeans</h3>
        {get_best_kmeans_results(results)}
        
        <h3>DBSCAN</h3>
        {get_best_dbscan_results(results)}
        
        <script>
            var plotlyData = {pio.to_json(sweep_fig)};
            Plotly.newPlot('sweepPlot', plotlyData.data, plotlyData.layout);
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def get_best_kmeans_results(results):
    kmeans_results = [r for r in results if r['algorithm'] == 'KMeans']
    best_kmeans = max(kmeans_results, key=lambda x: x['silhouette_score'])
    return f"""
    <ul>
        <li>Best n_clusters: {best_kmeans['n_clusters']}</li>
        <li>Silhouette Score: {best_kmeans['silhouette_score']:.4f}</li>
        <li>Intra-cluster Distance: {best_kmeans['intra_cluster_distance']:.4f}</li>
    </ul>
    """

def get_best_dbscan_results(results):
    dbscan_results = [r for r in results if r['algorithm'] == 'DBSCAN']
    best_dbscan = max(dbscan_results, key=lambda x: x['silhouette_score'])
    return f"""
    <ul>
        <li>Best eps: {best_dbscan['eps']:.2f}</li>
        <li>Best min_samples: {best_dbscan['min_samples']}</li>
        <li>Number of clusters: {best_dbscan['n_clusters']}</li>
        <li>Silhouette Score: {best_dbscan['silhouette_score']:.4f}</li>
    </ul>
    """