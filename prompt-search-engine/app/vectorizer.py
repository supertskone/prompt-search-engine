import os
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

# Disable parallelism for tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vectorizer:
    def __init__(self, model_name='all-mpnet-base-v2', batch_size=64, init_pinecone=True):
        logger.info(f"Initializing Vectorizer with model {model_name} and batch size {batch_size}")
        self.model = SentenceTransformer(model_name)
        self.prompts = []
        self.batch_size = batch_size
        self.pinecone_index_name = "prompts-index"
        self._init_pinecone = init_pinecone
        self._setup_pinecone()
        self._load_prompts()

    def _setup_pinecone(self):
        logger.info("Setting up Pinecone")
        # Initialize Pinecone
        pinecone = Pinecone(api_key='b514eb66-8626-4697-8a1c-4c411c06c090')
        # Check if the Pinecone index exists, if not create it
        existing_indexes = pinecone.list_indexes()

        logger.info(f"self.init_pineconeself.init_pineconeself"
                    f".init_pineconeself.init_pineconeself.init_pinecone: {self._init_pinecone}")
        if self.pinecone_index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.pinecone_index_name}")
            if self._init_pinecone:
                pinecone.create_index(
                    name=self.pinecone_index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
        else:
            logger.info(f"Pinecone index {self.pinecone_index_name} already exists")

        self.index = pinecone.Index(self.pinecone_index_name)

    def _load_prompts(self):
        logger.info("Loading prompts from Pinecone")
        self.prompts = []
        # Fetch vectors from the Pinecone index
        index_stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {index_stats}")

        namespaces = index_stats['namespaces']
        for namespace, stats in namespaces.items():
            vector_count = stats['vector_count']
            ids = [str(i) for i in range(vector_count)]
            for i in range(0, vector_count, self.batch_size):
                batch_ids = ids[i:i + self.batch_size]
                response = self.index.fetch(ids=batch_ids)
                for vector in response.vectors.values():
                    metadata = vector.get('metadata')
                    if metadata and 'text' in metadata:
                        self.prompts.append(metadata['text'])
        logger.info(f"Loaded {len(self.prompts)} prompts from Pinecone")

    def _store_prompts(self, dataset):
        logger.info("Storing prompts in Pinecone")
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            vectors = self.model.encode(batch)
            # Prepare data for Pinecone
            pinecone_data = [{'id': str(i + j), 'values': vector.tolist(), 'metadata': {'text': batch[j]}} for j, vector
                             in enumerate(vectors)]
            self.index.upsert(vectors=pinecone_data)
            logger.info(f"Upserted batch {i // self.batch_size + 1}/{len(dataset) // self.batch_size + 1} to Pinecone")

    def transform(self, prompts):
        return np.array(self.model.encode(prompts))

    def store_from_dataset(self, store_data=False):
        if store_data:
            logger.info("Loading dataset")
            dataset = load_dataset('fantasyfish/laion-art', split='train')
            logger.info(f"Loaded {len(dataset)} items from dataset")
            logger.info("Please wait for storing. This may take up to five minutes. ")
            self._store_prompts([item['text'] for item in dataset])
            logger.info("Items from dataset are stored.")
            # Ensure prompts are loaded after storing
            self._load_prompts()
            logger.info("Items from dataset are loaded.")
