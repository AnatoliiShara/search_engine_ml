import os 
import joblib
import numpy as np
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

class NaiveBayesSearch:
    def __init__(self, alpha: float = 1.0):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB(alpha=alpha))
        ])
        self.df = None
        
    def fit(self, texts: List[str], categories: List[str]):
        print("Fitting TFIDF vectorizer and Naive Bayes model...")
        self.pipeline.fit(texts, categories)
        print("Naive Bayes model fitted.")
        
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
        joblib.dump(self.pipeline, os.path.join(path, "naive_bayes_pipeline.pkl"))
        print("Model saved successfully.")
        
    def load_model(self, path: str):
        print(f"Loading model from {path}...")
        self.pipeline = joblib.load(os.path.join(path, "naive_bayes_pipeline.pkl"))
        print("Model loaded successfully.")
        
    def get_param_grid(self):
        return {
            'nb__alpha': [0.1, 0.5, 1.0, 2.0],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__max_df': [0.5, 0.75, 1.0],
            'tfidf__min_df': [1, 2, 3],
        }
        
def main():
    import pandas as pd
    print("Loading data...")
    df = pd.read_csv("/home/anatolii-shara/Documents/mlops_experiments/fake_disinformation_project/data/raw/political_disinformation_dataset_v2.csv")
    
    search_engine = NaiveBayesSearch()
    search_engine.fit(df['text'].tolist(), df['category'].tolist())
    search_engine.df = df
    
    # Ensure directory exists before saving the model
    search_engine.save_model('models/naive_bayes')
    query = "Leaked documents show that the government is planning to cancel all upcoming elections."
    results = search_engine.search(query, n=3)
    
    print(f"\nTop {len(results)} search results:")
    for idx, prob, category in results:
        print(f"Sentence: {df.iloc[idx]['text']}")
        print(f"Similarity: {prob:.2f}")
        print(f"Category: {category}")
        print()
        
if __name__ == "__main__":
    main()
     
        
    