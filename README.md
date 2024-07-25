# Prompt Search Engine

## Overview

The Prompt Search Engine is a Flask-based web application designed to search for the most similar prompts from a dataset using cosine similarity. It leverages Hugging Face's `sentence-transformers` to vectorize prompts and stores them in a Pinecone vector database for efficient querying. The frontend is built using Streamlit, providing an intuitive interface for users to input their queries and get results.

## Features

- **Efficient Vector Search**: Uses Pinecone for storing and querying vector embeddings.
- **Cosine Similarity Calculation**: Custom implementation for calculating cosine similarity between query vectors and stored vectors.
- **Streamlit Interface**: Simple and user-friendly interface for querying and displaying results.
- **Logging**: Comprehensive logging for easy debugging and monitoring.

## Prerequisites

- Python 3.9 or higher
- Pinecone API key

## Setup

### Clone the Repository

```
git clone https://github.com/your-username/prompt-search-engine.git
cd prompt-search-engine
```
## Create a Virtual Environment and Install Dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure Pinecone
```
# Replace YOUR_PINECONE_API_KEY with your actual Pinecone API key in the vectorizer.py file.
python
pinecone = Pinecone(api_key='YOUR_PINECONE_API_KEY')
```
### Initial Data Load
```
# Run the script to load the initial dataset into the Pinecone database:

python load_data.py
```

### Run the Flask Backend
```
python run.py
```

### Run the Streamlit Frontend
```
# In a new terminal (while the backend is running), start the Streamlit app:

streamlit run ui/app.py
```
### Running the Tests 
To run the tests, navigate to the root directory and execute:
```
python run_tests.py
```
Make sure that you executed python load_data.py before.
You should receive something like this:
<img width="576" alt="image" src="https://github.com/user-attachments/assets/a9cd8acb-9280-4b55-9bec-3009a0a61b87">

### Rebuild and run Docker container
```
docker build -t prompt-search-engine .
docker run -p 5000:5000 prompt-search-engine
```

### Usage
Open your web browser and go to http://localhost:8501.
Enter a query in the input box.
Adjust the number of results using the slider.
Click "Search" to get the most similar prompts from the dataset.

### File Descriptions
##### app/: Contains the core logic for vectorization and search functionality.
##### search_engine.py: Implements the PromptSearchEngine class for querying the Pinecone database.
##### vectorizer.py: Implements the Vectorizer class for loading and storing vectors in Pinecone.
##### ui/: Contains the Streamlit frontend application.
##### app.py: Streamlit app for user interface.
##### load_data.py: Script to load the initial dataset into Pinecone.
##### run.py: Flask application entry point.
##### requirements.txt: Lists the Python dependencies for the project.

### Logging
Logging is configured in the vectorizer.py and search_engine.py files.

### License
This project is licensed under the MIT License.

## Acknowledgements
##### - Hugging Face for sentence-transformers
##### - Pinecone for vector database services
##### - Streamlit for the web app interface
##### - Feel free to reach out if you have any questions or need further assistance. Enjoy using the Prompt Search Engine!
