import numpy as np


def cosine_similarity(
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between a query vector and a corpus of vectors.

    Args:
        query_vector: Vectorized prompt query of shape (D,).
        corpus_vectors: Vectorized prompt corpus of shape (N, D).

    Returns:
        np.ndarray: The vector of shape (N,) with values in range [-1, 1] where 1
        is max similarity i.e., two vectors are the same.
    """
    dot_product = np.dot(corpus_vectors, query_vector)
    norm_query = np.linalg.norm(query_vector)
    norm_corpus = np.linalg.norm(corpus_vectors, axis=1)
    return dot_product / (norm_query * norm_corpus)