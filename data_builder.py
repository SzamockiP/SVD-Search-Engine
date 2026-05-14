import os
import pickle
import nltk
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    processed_words = [stemmer.stem(w) for w in words if w.isalnum() and w not in stop_words]
    return " ".join(processed_words)

def build_index():
    print("Loading dataset from HuggingFace...")
    # Task 1: Prepare a large dataset (> 1000 items) in English
    dataset = load_dataset("wikipedia", "20220301.simple", split="train[:10000]")
    
    docs = []
    metadata = []
    
    print("Preprocessing texts...")
    for i, item in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i} documents...")
        title = item['title']
        text = item['text']
        
        processed_text = preprocess_text(text)
        processed_title = preprocess_text(title)
        
        # Make title more important in search results
        boosted_content = (processed_title + " ") * 3 + processed_text
        
        docs.append(boosted_content)
        metadata.append({
            "id": i,
            "title": title,
            "snippet": text[:200] + "..."
        })
        
    print("Training CountVectorizer (TF - Bag of Words)...")
    # Task 2: Define dictionary of keywords (indexation)
    count_vectorizer = CountVectorizer(min_df=2, max_df=0.85, max_features=20000)
    # Task 3: Compute bag-of-words frequencies
    # Task 4: Build a sparse term-by-document matrix
    count_matrix = count_vectorizer.fit_transform(docs)
    
    print("Training TfidfVectorizer (TF-IDF)...")
    # Task 5: Multiply bag-of-words by inverse document frequency (IDF)
    tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.85, max_features=20000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    
    print("Applying TruncatedSVD (LSA) for noise reduction (k=300)...")
    # Task 8: Apply SVD and low rank approximation to remove noise
    n_components = 300
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    docs_dense = svd.fit_transform(tfidf_matrix)
    
    print("Saving models and matrices to disk...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/count_vectorizer.pkl", "wb") as f:
        pickle.dump(count_vectorizer, f)
        
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
        
    with open("models/svd.pkl", "wb") as f:
        pickle.dump(svd, f)
        
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    sparse.save_npz("models/count_matrix.npz", count_matrix)
    sparse.save_npz("models/tfidf_matrix.npz", tfidf_matrix)
    np.save("models/docs_dense.npy", docs_dense)
        
    print("Done! Index and matrices built successfully.")

if __name__ == "__main__":
    build_index()
