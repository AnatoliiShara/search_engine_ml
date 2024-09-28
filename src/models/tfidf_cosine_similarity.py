import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from tqdm import tqdm
import os
import joblib

class TfidfCosineSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.df = None
        
    def fit(self, texts: List[str]):
        print("Fitting TFIDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print("TFIDF vectorizer fitted.")
        
    def search(self, query: str, n: int = 5) -> List[Tuple[int, float, str]]:
        print(f"Searching for top {n} similar documents...")
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                cosine_similarities[idx],
                self.df.iloc[idx]['category']
            ))
        return results
    
    def save_model(self, path: str):
        # Ensure the directory exists before saving
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path)
        print(f"Saving model to {path}...")
        joblib.dump(self.vectorizer, os.path.join(path, "tfidf_vectorizer.pkl"))
        joblib.dump(self.tfidf_matrix, os.path.join(path, "tfidf_matrix.pkl"))
        print("Model saved successfully.")
        
    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.vectorizer = joblib.load(os.path.join(path, "tfidf_vectorizer.pkl"))
        self.tfidf_matrix = joblib.load(os.path.join(path, "tfidf_matrix.pkl"))
        print("Model loaded successfully.")
        
def main():
    print("Loading data...")
    df = pd.read_csv("/home/anatolii-shara/Documents/mlops_experiments/fake_disinformation_project/data/raw/political_disinformation_dataset_v2.csv")
    
    search_engine = TfidfCosineSearch()
    search_engine.fit(df['text'])
    search_engine.df = df
    
    # Ensure directory exists before saving the model
    search_engine.save_model("models/tfidf_cosine")
    
    query = "Donald Trump is a liar."
    results = search_engine.search(query)
    
    print(f"Top results for query '{query}':")
    for idx, similarity, category in results:
        print(f"Sentence: {df.iloc[idx]['text']}")
        print(f"Category: {category}")
        print(f"Similarity: {similarity:.2f}")
        print()
        
if __name__ == "__main__":
    main()
