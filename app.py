from flask import Flask, request, jsonify, render_template
import pickle
import os
import nltk
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)

print("Loading models and matrices...")
MODELS_DIR = "models"
try:
    with open(os.path.join(MODELS_DIR, "count_vectorizer.pkl"), "rb") as f:
        count_vectorizer = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, "svd.pkl"), "rb") as f:
        svd = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
        
    count_matrix = sparse.load_npz(os.path.join(MODELS_DIR, "count_matrix.npz"))
    tfidf_matrix = sparse.load_npz(os.path.join(MODELS_DIR, "tfidf_matrix.npz"))
    docs_dense = np.load(os.path.join(MODELS_DIR, "docs_dense.npy"))
    
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models.{e}")
    count_vectorizer, tfidf_vectorizer, svd, metadata = None, None, None, None
    count_matrix, tfidf_matrix, docs_dense = None, None, None

stemmer = SnowballStemmer("english")
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    processed_words = [stemmer.stem(w) for w in words if w.isalnum() and w not in stop_words]
    return " ".join(processed_words)

def compute_search(query, use_idf, use_svd, k_components):
    processed_query = preprocess_text(query)
    if not processed_query:
        return []
        
    vectorizer = tfidf_vectorizer if use_idf else count_vectorizer
    matrix = tfidf_matrix if use_idf else count_matrix
    
    # Task 6: Convert query to vector representation (bag-of-words)
    query_vector = vectorizer.transform([processed_query])
    
    if use_svd:
        # Task 8: Remove noise using SVD
        # Project query into the reduced SVD space
        query_dense = svd.transform(query_vector)
        
        k = int(k_components)
        q_k = query_dense[:, :k]
        d_k = docs_dense[:, :k]
        
        # Task 8: Compute new similarity
        similarities = cosine_similarity(q_k, d_k)[0]
    else:
        # Task 6 & 7: Compute cosine similarity
        similarities = cosine_similarity(query_vector, matrix)[0]
        
    # Top 10 results
    top_indices = similarities.argsort()[-10:][::-1]
    
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim > 0.0:
            doc_meta = metadata[idx]
            results.append({
                "id": int(doc_meta["id"]),
                "title": doc_meta["title"],
                "snippet": doc_meta["snippet"],
                "similarity": sim
            })
            
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search():
    if not metadata:
        return jsonify({"error": "Models were not loaded."}), 500
        
    query = request.args.get("q", "")
    if not query:
        return jsonify({"left": [], "right": []})
        
    # Left panel parameters
    l_idf = request.args.get("l_idf", "false").lower() == "true"
    l_svd = request.args.get("l_svd", "false").lower() == "true"
    l_k = int(request.args.get("l_k", "50"))
    
    # Right panel parameters
    r_idf = request.args.get("r_idf", "true").lower() == "true"
    r_svd = request.args.get("r_svd", "true").lower() == "true"
    r_k = int(request.args.get("r_k", "50"))
    
    left_results = compute_search(query, l_idf, l_svd, l_k)
    right_results = compute_search(query, r_idf, r_svd, r_k)
        
    return jsonify({
        "left": left_results,
        "right": right_results
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
