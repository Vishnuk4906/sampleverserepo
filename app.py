import json
import numpy as np
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from IPython.display import display, Markdown
import google.generativeai as genai
genai.configure(api_key="AIzaSyAyflgig0oBMy8y3YNtusBS_KNKvau3apc")

model = genai.GenerativeModel("gemini-1.5-flash")

image_ = "sample.jpeg"
st.image(image_,use_column_width=False)
def fn_template(query,verses):
    template = f"Your designed to give good and motivated insights to the believer.\
                you have the believer's query and the verses from bible related to that.your role is to create \
                soulfull insights from both which will be an answer to the user.response should be maximum 2 sentances.\
                <query>{query}</query>\
                <verses>{verses}</verses>"
    return template
# response = model.generate_content("Hello")
# response.text
# Load the saved fitted vectorizer
with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Load the TF-IDF matrix
tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")

# Load the verse data (with verses and metadata)
with open("verse_data.json", 'r') as abc:
    data = json.load(abc)

# Search query function
def search_query(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    similarity_scores = similarities.flatten()
    sorted_indices = np.argsort(similarity_scores)[::-1][:3]
    return sorted_indices

# Display verse with metadata
def display_verse_with_metadata(verse, metadata):
    formatted_text = f"**{verse}**\n\n*{metadata}*"
    st.markdown(formatted_text)

# Streamlit input and display
st.subheader("Tell jesus what you feel!")
input_text = st.text_input(label="")
if st.button("Get verse"):
    if input_text:
        idxs = search_query(input_text)
        quote = ""
        for i in idxs:
            quote+=data[i]["quote"]
            display_verse_with_metadata(data[i]["quote"], data[i]["from"])
    res = model.generate_content(fn_template(input_text,quote))
    print(res)
    st.subheader(res.candidates[0].content.parts[0].text)

