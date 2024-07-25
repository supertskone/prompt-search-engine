import unittest
from unittest.mock import patch
import numpy as np
from app.search_engine import PromptSearchEngine


class TestPromptSearchEngine(unittest.TestCase):

    @patch('app.vectorizer.Vectorizer')
    def setUp(self, mock_vectorizer):
        self.mock_vectorizer = mock_vectorizer.return_value
        self.mock_vectorizer.transform.return_value = np.random.rand(10, 768)
        self.mock_vectorizer.prompts = ['prompt'] * 10
        self.search_engine = PromptSearchEngine()

    def test_most_similar_with_cosine_similarity(self):
        self.mock_vectorizer.index.query.side_effect = Exception('Pinecone error')
        results = self.search_engine.most_similar('query', use_pinecone=False)
        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0][0], float)
        self.assertIsInstance(results[0][1], str)

    def test_most_similar_with_pinecone(self):
        mock_search_result = {
            'matches': [
                {'score': np.float32(0.9), 'metadata': {'text': 'prompt1'}},
                {'score': np.float32(0.8), 'metadata': {'text': 'prompt2'}}
            ]
        }
        self.mock_vectorizer.index.query.return_value = mock_search_result

        results = self.search_engine.most_similar('query', use_pinecone=True)
        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0][0], float)
        self.assertIsInstance(results[0][1], str)


if __name__ == '__main__':
    unittest.main()