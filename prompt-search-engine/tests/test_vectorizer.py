import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from app.vectorizer import Vectorizer


class TestVectorizer(unittest.TestCase):

    @patch('app.vectorizer.Pinecone')
    @patch('app.vectorizer.SentenceTransformer')
    def test_vectorizer_initialization(self, mock_sentence_transformer, mock_pinecone):
        mock_sentence_transformer.return_value.encode.return_value = np.random.rand(1, 768)
        vectorizer = Vectorizer(init_pinecone=False)
        self.assertEqual(vectorizer.batch_size, 64)
        self.assertEqual(vectorizer.pinecone_index_name, "prompts-index")

    @patch('app.vectorizer.load_dataset')
    @patch('app.vectorizer.Pinecone')
    def test_store_from_dataset(self, mock_pinecone, mock_load_dataset):
        mock_pinecone_instance = MagicMock()
        mock_pinecone.return_value = mock_pinecone_instance
        mock_load_dataset.return_value = [{'text': 'sample text'}]

        vectorizer = Vectorizer(init_pinecone=False)
        vectorizer.store_from_dataset(store_data=True)

        mock_load_dataset.assert_called_once_with('fantasyfish/laion-art', split='train')
        mock_pinecone_instance.Index.return_value.upsert.assert_called()

    def test_transform(self):
        with patch('app.vectorizer.SentenceTransformer') as mock_sentence_transformer:
            mock_sentence_transformer.return_value.encode.return_value = np.random.rand(1, 768)
            vectorizer = Vectorizer(init_pinecone=False)
            vectors = vectorizer.transform(['sample prompt'])
            self.assertEqual(vectors.shape, (1, 768))
