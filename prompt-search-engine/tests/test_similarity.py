import unittest
import numpy as np
from app.similarity import cosine_similarity


class TestSimilarity(unittest.TestCase):

    def test_cosine_similarity(self):
        query_vector = np.array([1, 2, 3])
        corpus_vectors = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        expected_result = np.array([1.0, 0.9746318461970762, 0.9594119455666703])
        result = cosine_similarity(query_vector, corpus_vectors)

        np.testing.assert_almost_equal(result, expected_result, decimal=6)


if __name__ == '__main__':
    unittest.main()
