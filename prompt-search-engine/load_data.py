from app.vectorizer import Vectorizer

if __name__ == "__main__":
    vectorizer = Vectorizer()
    vectorizer.store_from_dataset()  # Run this once to load the dataset into Pinecone
