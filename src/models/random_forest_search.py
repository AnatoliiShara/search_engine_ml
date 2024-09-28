import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class RandomForestSearch:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
        ])
        self.df = None
        
    def fit(self, texts: List[str], categories: List[str]):
        print("Fitting TFIDF vectorizer and Random Forest model...")
        self.pipeline.fit(texts, categories)
        print("Random Forest model fitted.")
        
    def search(self, query: str, n: int = 5) -> List[Tuple[int, float, str]]:
        print(f"Searching for top {n} similar documents...")
        probabilities = self.pipeline.predict_proba([query])[0]
        top_indices = np.argsort(probabilities)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                probabilities[idx],
                self.df.iloc[idx]['category']
            ))
        return results
    
    def save_model(self, path: str):
        print(f"Saving model to {path}...")
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path)
        joblib.dump(self.pipeline, os.path.join(path, "random_forest_pipeline.pkl"))
        print("Model saved successfully.")
        
    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.pipeline = joblib.load(os.path.join(path, "random_forest_pipeline.pkl"))
        print("Model loaded successfully.")
        
    def get_param_grid(self):
        return {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 10, 20, 30],
            'tfidf_max_df': [0.5, 0.75, 1.0],
            'tfidf_min_df': [1, 2, 3],
        }
        
def main():
    print("Loading data...")
    df = pd.read_csv("/home/anatolii-shara/Documents/mlops_experiments/fake_disinformation_project/data/raw/political_disinformation_dataset_v2.csv")
    
    search_engine = RandomForestSearch()
    search_engine.fit(df['text'].to_list(), df['category'].to_list())
    search_engine.df = df
    
    # Ensure directory exists before saving the model
    save_path = "./models/random_forest"
    search_engine.save_model(save_path)
    
    query = "Leaked documents show that the government is planning to cancel all upcoming elections"
    results = search_engine.search(query, n=3)
    
    print(f"\nTop 3 results for query '{query}':")  
    for idx, score, category in results:
        print(f"Sentence: {df.iloc[idx]['text']}")
        print(f"Probability: {score}")
        print(f"Category: {category}")
        print("\n")
        
if __name__ == "__main__":
    main()