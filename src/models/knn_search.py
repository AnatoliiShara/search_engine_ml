import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from tqdm import tqdm
import os
import joblib

class KNNSearch:
    def __init__(self, n_neighbors: int = 5):
        self.vectorizer = TfidfVectorizer()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.df = None
        
    def fit(self, texts: List[str]):
        print("Fitting TFIDF vectorizer and KNN model...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.knn.fit(self.tfidf_matrix)
        print("KNN model fitted.")
        
    def search(self, query: str, n: int = 5) -> List[Tuple[int, float, str]]:
        print(f"Searching for top {n} similar documents...")
        query_vector = self.vectorizer.transform([query])
        distances, top_indices = self.knn.kneighbors(query_vector, n_neighbors=n)
        results = []
        for idx, distance in zip(top_indices[0], distances[0]):
            results.append((
                idx,
                1 - distance,
                self.df.iloc[idx]['category']
            ))
        return results
    
    def save_model(self, path: str):
        print(f"Saving model to {path}...")
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path)
        joblib.dump(self.vectorizer, os.path.join(path, "knn_vectorizer.pkl"))
        joblib.dump(self.knn, os.path.join(path, "knn_model.pkl"))
        print("Model saved successfully.")
        
    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.vectorizer = joblib.load(os.path.join(path, "knn_vectorizer.pkl"))
        self.knn = joblib.load(os.path.join(path, "knn_model.pkl"))
        print("Model loaded successfully.")
        
    def get_param_grid(self):
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'metric': ['cosine', 'euclidean']
        }
        
def main():
    import pandas as pd
    print("Loading data...")
    df = pd.read_csv("/home/anatolii-shara/Documents/mlops_experiments/fake_disinformation_project/data/raw/political_disinformation_dataset_v2.csv")   
    search_engine = KNNSearch()
    search_engine.fit(df['text'])
    search_engine.df = df
    
    # Ensure directory exists before saving the model
    search_engine.save_model("models/knn_search")
    query = "Donald Trump is a liar."
    results = search_engine.search(query)
    
    print(f"\nTop 3 results for query '{query}':")  
    for idx, similarity, category in results:
        print(f"Sentence: {df.iloc[idx]['text']}")
        print(f"Similarity: {similarity:.2f}")
        print(f"Category: {category}")
        print()
        
if __name__ == "__main__":
    main()