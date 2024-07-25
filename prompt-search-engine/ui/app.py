import json

import streamlit as st
import requests

st.title("Prompt Search Engine")

query = st.text_input("Enter your query:")
use_pinecone = st.radio(
    "Choose search method:",
    ('Pinecone Vector Search', 'Cosine Similarity')
)
n = st.number_input("Number of results:", min_value=1, max_value=20, value=5)

if st.button("Search"):
    search_method = use_pinecone == 'Pinecone Vector Search'
    response = requests.post("http://localhost:5000/search", json={"query": query, "n": n, "use_pinecone": search_method})

    # Log the response for debugging
    st.write("Response Status Code:", response.status_code)

    try:
        results = response.json()
        # for score, prompt in results:
        #     st.write(f"{score:.2f} - {prompt}")
        for result in results:
            score = float(result['score'])
            prompt = result['prompt']
            st.write(f"{score:.2f} - {prompt}")
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON response: {e}")
        st.write(response.content)