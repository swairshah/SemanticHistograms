import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.manifold import TSNE

st.set_page_config(layout="wide", page_title="Semantic Clustering Explorer")

@st.cache_data
def load_data():
    clusters = joblib.load('output/clusters.joblib')
    embeddings = joblib.load('output/embeddings.joblib')
    sentences = joblib.load('output/sentences.joblib')
    # Sort clusters by size
    sorted_clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
    return sorted_clusters, embeddings, sentences

clusters, embeddings, sentences = load_data()

st.title('Semantic Clustering Explorer')

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Cluster Overview", "Sentence Explorer", "Visualization"])

def cluster_overview():
    st.header("Cluster Overview")
    
    # Display cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = pd.DataFrame({
        "Cluster": clusters.keys(),
        "Size": [len(cluster) for cluster in clusters.values()]
    }).sort_values("Size", ascending=False)
    
    # Use Plotly Express to create a sorted bar chart
    fig = px.bar(cluster_stats, x='Cluster', y='Size', 
                 title='Cluster Sizes',
                 labels={'Cluster': 'Cluster ID', 'Size': 'Number of Sentences'},
                 height=500)
    fig.update_xaxes(type='category', categoryorder='total descending')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top sentences from each cluster
    st.subheader("Top Sentences from Each Cluster")
    num_sentences = st.slider("Number of sentences to show", 1, 10, 3)
    
    for cluster, sentences in clusters.items():
        with st.expander(f"Cluster {cluster} (Size: {len(sentences)})"):
            for sentence in sentences[:num_sentences]:
                st.write(f"- {sentence}")

def sentence_explorer():
    st.header("Sentence Explorer")
    
    # Cluster selection
    selected_cluster = st.selectbox('Select Cluster', list(clusters.keys()))
    cluster_sentences = clusters[selected_cluster]
    cluster_embeddings = np.array([embeddings[sentences.index(s)] for s in cluster_sentences])
    
    st.write(f"Cluster {selected_cluster}: {len(cluster_sentences)} sentences")
    
    st.markdown("---")  # Separator
    
    # Sentence search
    search_query = st.text_input("Search for sentences (leave empty to show all)")
    filtered_sentences = [s for s in cluster_sentences if search_query.lower() in s.lower()] if search_query else cluster_sentences
    
    # Pagination
    sentences_per_page = st.slider("Sentences per page", 5, 50, 10)
    page_number = st.number_input('Page', min_value=1, max_value=(len(filtered_sentences) - 1) // sentences_per_page + 1, value=1)
    start = (page_number - 1) * sentences_per_page
    end = start + sentences_per_page
    
    st.subheader('Sentences in Cluster')
    
    # Display sentences as a table
    df = pd.DataFrame({
        "Index": range(start, min(end, len(filtered_sentences))),
        "Sentence": filtered_sentences[start:end]
    })
    st.table(df)
    
    st.markdown("---") 
    
    # Centroid analysis
    st.subheader('Centroid Analysis')
    centroid = np.mean(cluster_embeddings, axis=0)
    similarities = cosine_similarity(cluster_embeddings, [centroid]).flatten()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Sentences Closest to Centroid")
        k_nearest = st.slider('Number of nearest sentences', 1, 10, 3, key='nearest')
        nearest_indices = similarities.argsort()[-k_nearest:][::-1]
        nearest_df = pd.DataFrame({
            "Index": nearest_indices,
            "Sentence": [cluster_sentences[idx] for idx in nearest_indices]
        })
        st.table(nearest_df)
    
    with col2:
        st.write("Sentences Farthest from Centroid")
        k_farthest = st.slider('Number of farthest sentences', 1, 10, 3, key='farthest')
        farthest_indices = similarities.argsort()[:k_farthest]
        farthest_df = pd.DataFrame({
            "Index": farthest_indices,
            "Sentence": [cluster_sentences[idx] for idx in farthest_indices]
        })
        st.table(farthest_df)

def visualization():
    st.header("Cluster Visualization")
    
    # Perform dimensionality reduction
    @st.cache_data
    def get_tsne_embeddings():
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(embeddings)
    
    tsne_embeddings = get_tsne_embeddings()
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': tsne_embeddings[:, 0],
        'y': tsne_embeddings[:, 1],
        'cluster': [next(cluster for cluster, sents in clusters.items() if sentence in sents) for sentence in sentences],
        'sentence': sentences
    })
    
    # Plot
    fig = px.scatter(
        df, x='x', y='y', 
        color='cluster', 
        hover_data=['sentence'],
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        title='2D Visualization of Clusters'
    )
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Hover over points to see sentences. Use mouse to pan and zoom.")

    # cluster selection for highlighting
    selected_cluster = st.selectbox('Highlight Cluster', ['All'] + list(clusters.keys()))

# Main stuff
if page == "Cluster Overview":
    cluster_overview()
elif page == "Sentence Explorer":
    sentence_explorer()
elif page == "Visualization":
    visualization()

st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit")