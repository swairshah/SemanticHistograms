# Semantic Histogram

Instead of creating histograms of sentence frequencies, create histograms of semantically similarity sentences.
Generate some nice reports!

## Features

- Semantic embedding of sentences
- Clustering using KMeans and DBSCAN algorithms
- Hyperparameter tuning with grid search
- Generation of interactive visualization plots
- Outlier detection within clusters
- Cluster summarization
- Nice report gen!

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/semantic-clustering-analysis.git
   cd semantic-clustering-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input data as a text file with one sentence per line.

2. Run the main script:
   ```
   python cluster.py path/to/your/input.txt
   ```

3. The script will generate two reports:
   - `output/clustering_report.html`: Detailed clustering results
   - `output/sweep_report.html`: Hyperparameter sweep analysis

## Project Structure

- `cluster.py`: Main script for running the clustering analysis
- `sentences.py`: Handles sentence preprocessing and embedding
- `summarize.py`: Provides text summarization functionality
- `report.py`: Generates HTML reports with visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.