import os
import logging
from flask import Flask, request, jsonify
from app.search_engine import PromptSearchEngine

app = Flask(__name__)

# Disable parallelism for tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

search_engine = PromptSearchEngine()

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    n = data.get('n', 5)
    logger.info(f"Received query: {query} with n: {n}")
    results = search_engine.most_similar(query, n)
    formatted_results = [{'score': score, 'prompt': prompt} for score, prompt in results]
    logger.info(f"Returning results: {formatted_results}")
    return jsonify(formatted_results)


if __name__ == '__main__':
    # logger.info("Starting Flask server")
    app.run(debug=True)
