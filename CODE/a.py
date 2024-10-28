import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Load document embeddings from a pickle file
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return np.array(embeddings)


# Search for the most similar documents using cosine similarity
def search_similar_documents(query_embedding, document_embeddings, top_n=5):
    try:
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), document_embeddings
        )
        ranked_indices = np.argsort(similarities[0])[::-1]
        return ranked_indices[:top_n]
    except Exception as e:
        print(f"Error in searching similar documents: {e}")
        return []


# Cluster the documents into groups using KMeans
def cluster_documents(document_embeddings, num_clusters=5):
    if len(document_embeddings) < num_clusters:
        print(
            f"Number of samples ({len(document_embeddings)}) is less than the number of clusters ({num_clusters}). Reducing number of clusters."
        )
        num_clusters = len(document_embeddings)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(document_embeddings)
    return labels


# Visualize embeddings using t-SNE and cluster labels
def visualize_embeddings_tsne(document_embeddings, cluster_labels=None):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(document_embeddings)
    plt.figure(figsize=(10, 7))

    if cluster_labels is not None:
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap="viridis",
        )
        plt.legend(*scatter.legend_elements(), title="Clusters")
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    plt.title("t-SNE Visualization of Document Embeddings")
    plt.show()


# Compare two documents based on cosine similarity
def compare_documents(doc1, doc2):
    try:
        similarity = cosine_similarity(doc1.reshape(1, -1), doc2.reshape(1, -1))
        return similarity[0][0]
    except Exception as e:
        print(f"Error comparing documents: {e}")
        return None


# Normalize embeddings (optional, helps with cosine similarity)
def normalize_embeddings(document_embeddings):
    norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    return document_embeddings / norms


def main():
    # Load the document embeddings
    file_path = "document_embedding.pkl"
    document_embeddings = load_embeddings(file_path)
    print(f"Number of document embeddings: {len(document_embeddings)}")
    if len(document_embeddings) > 0:
        print(f"Shape of each embedding: {document_embeddings[0].shape}")
    if document_embeddings.shape[0] <= 1:
        print(
            "Insufficient document embeddings for clustering. Ensure more documents are loaded."
        )
        return

    # Normalize embeddings (optional)
    # document_embeddings = normalize_embeddings(document_embeddings)

    # Search for similar documents
    query_embedding = document_embeddings[
        0
    ]  # Using the first document as an example query
    top_indices = search_similar_documents(query_embedding, document_embeddings)
    print(f"Top similar documents: {top_indices}")

    # Cluster the documents
    num_clusters = 5
    cluster_labels = cluster_documents(document_embeddings, num_clusters)
    print(f"Cluster labels: {cluster_labels}")

    # Compare the first two documents
    similarity = compare_documents(document_embeddings[0], document_embeddings[1])
    print(f"Similarity between first two documents: {similarity}")

    # Visualize document embeddings using t-SNE, along with cluster labels
    visualize_embeddings_tsne(document_embeddings, cluster_labels)


if __name__ == "__main__":
    main()
