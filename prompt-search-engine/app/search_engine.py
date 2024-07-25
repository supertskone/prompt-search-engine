import numpy as np
from typing import List, Tuple

from .similarity import cosine_similarity
from .vectorizer import Vectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptSearchEngine:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.prompts = self.vectorizer.prompts
        self.corpus_vectors = self.vectorizer.transform(self.prompts)
        self.index_name = self.vectorizer.pinecone_index_name

    # def most_similar(self, query: str, n: int = 5) -> List[Tuple[float, str]]:
    #     logger.info(f"Encoding query: {query}")
    #     query_vector = self.vectorizer.model.encode([query])[0]
    #     logger.info(f"Encoded query vector: {query_vector}")
    # 
    #     search_result = self.vectorizer.index.query(
    #         vector=query_vector.tolist(),
    #         top_k=n,
    #         include_metadata=True
    #     )
    #     logger.info(f"Search result: {search_result}")
    # 
    #     # Retrieve and format the results
    #     results = [(match['score'], match['metadata']['text']) for match in search_result['matches'] if 'text' in match['metadata']]
    #     return results
    # 
    # def most_similar(self, query: str, n: int = 5) -> List[Tuple[float, str]]:
    #     query_vector = self.vectorizer.transform([query])[0]
    #     similarities = cosine_similarity(query_vector, self.corpus_vectors)
    #     top_n_indices = np.argsort(similarities)[-n:][::-1]
    #     return [(similarities[i], self.prompts[i]) for i in top_n_indices]

    def most_similar(self, query: str, n: int = 5) -> List[Tuple[float, str]]:
        logger.info(f"Encoding query: {query}")
        query_vector = self.vectorizer.transform([query])[0]
        logger.info(f"Encoded query vector: {query_vector}")
        try:
            # Convert numpy array to list of native Python floats
            query_vector_list = query_vector.tolist()
            search_result = self.vectorizer.index.query(
                vector=query_vector_list,
                top_k=n,
                include_metadata=True
            )
            logger.info(f"Search result: {search_result}")

            # Retrieve and format the results
            results = [(match['score'], match['metadata']['text']) for match in search_result['matches'] if
                       'text' in match['metadata']]
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            logger.info("Falling back to cosine similarity search.")

            # Fallback to cosine similarity search
            similarities = cosine_similarity(query_vector, self.corpus_vectors)
            top_n_indices = np.argsort(similarities)[-n:][::-1]
            results = [(float(similarities[i]), self.prompts[i]) for i in top_n_indices]

        return results
