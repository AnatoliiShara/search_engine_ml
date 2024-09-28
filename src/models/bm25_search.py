import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import pandas as pd

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer()
        self.idf = None
        self.doc_len = None
        self.avgdl = None
        self.df = None
        
    def fit(self, texts: List[str]):
        print("Fitting BM25...")
        tf = self.vectorizer.fit_transform(texts)
        self.doc_len = tf.sum(axis=1).A1
        self.avgdl = self.doc_len.mean()
        
        n_docs = len(texts)
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        print("BM25 fitted.")
        
    def search(self, query: str, n: int = 5) -> List[Tuple[int, float, str]]:
        print(f"Searching for top {n} similar documents...")
        query_tf = self.vectorizer.transform([query])
        scores = self._bm25_score(query_tf)
        top_indices = np.argsort(scores)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                scores[idx],
                self.df.iloc[idx]['category']
            ))  
        return results
    
    def _bm25_score(self, query_tf):
        # term frequency (for query terms in all documents)
        tf = query_tf.toarray()
        
        # Calculate BM25 scores for each document
        scores = np.zeros(self.doc_len.shape[0])  # Initialize a score array for all documents
        for term_idx in range(tf.shape[1]):
            if tf[0, term_idx] == 0:  # Skip terms that don't appear in the query
                continue
            
            # BM25 components
            term_idf = self.idf[term_idx]
            doc_tf = self.vectorizer.transform(self.df['text']).toarray()[:, term_idx]  # term freq in documents for this term
            
            # Compute BM25 for each document
            numerator = doc_tf * (self.k1 + 1)
            denominator = doc_tf + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
            scores += (numerator / denominator) * term_idf

        return scores
    
    def save_model(self, path: str):
        print(f"Saving model to {path}...")
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path)
        joblib.dump(self.vectorizer, os.path.join(path, "bm25_vectorizer.pkl"))
        joblib.dump(self.idf, os.path.join(path, "bm25_idf.pkl"))
        print("Model saved successfully.")
        
    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.vectorizer = joblib.load(os.path.join(path, "bm25_vectorizer.pkl"))
        self.idf = joblib.load(os.path.join(path, "bm25_idf.pkl"))
        print("Model loaded successfully.")
        
    def get_param_grid(self):
        return {
            'k1': [1.2, 1.5, 1.8],
            'b': [0.65, 0.75, 0.85]
        }
        
def main():
    print("Loading data...")
    df = pd.read_csv("/home/anatolii-shara/Documents/mlops_experiments/fake_disinformation_project/data/raw/political_disinformation_dataset_v2.csv")
    
    search_engine = BM25()
    search_engine.fit(df['text'].to_list())
    search_engine.df = df
    search_engine.save_model("./models/bm25")  # Changed path to local relative directory
    
    query = "Leaked documents show that the government is planning to cancel all upcoming elections"
    results = search_engine.search(query)
    print(f"\nTop 3 results for query '{query}':")
    for idx, score, category in results:
        print(f"Sentence: {df.iloc[idx]['text']}")
        print(f"Similarity score: {score}")
        print(f"Category: {category}")
        print()
        
if __name__ == "__main__":
    main()
