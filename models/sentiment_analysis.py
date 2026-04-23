import os
import json
import traceback
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob

def get_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='positive', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='positive', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='positive', zero_division=0)
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def main():
    print("Starting sentiment_analysis.py: NLP Models Module 2")
    try:
        data_path = os.path.join("outputs", "cleaned_tweets.csv")
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}. Please run Module 1 first.")
            return

        print("Loading cleaned tweets...")
        df = pd.read_csv(data_path)
        
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_csv(data_path, index=False)
        
        df = df.dropna(subset=['cleaned_text', 'sentiment'])
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        
        print("Sampling 100,000 tweets...")
        if len(df) > 100000:
            df = df.sample(100000, random_state=42)
            
        print("Splitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        test_df = pd.DataFrame({
            'text': X_test,
            'actual': y_test
        })

        metrics = {}
        
        # ---------------------------------------------------------
        # Model 1: VADER
        # ---------------------------------------------------------
        print("\n--- Model 1: VADER ---")
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        
        print("Running VADER inference on test set...")
        vader_preds = []
        for text in X_test:
            score = sia.polarity_scores(str(text))['compound']
            vader_preds.append('positive' if score > 0 else 'negative')
            
        test_df['vader_pred'] = vader_preds
        vader_metrics = get_metrics(y_test, vader_preds, "VADER")
        metrics['VADER'] = vader_metrics
        print(f"VADER Accuracy: {vader_metrics['accuracy']:.4f}")

        # ---------------------------------------------------------
        # Model 2: TF-IDF + Logistic Regression
        # ---------------------------------------------------------
        print("\n--- Model 2: TF-IDF + Logistic Regression ---")
        print("Vectorizing text with TF-IDF...")
        tfidf = TfidfVectorizer(max_features=10000)
        X_train_tfidf = tfidf.fit_transform(X_train.astype(str))
        X_test_tfidf = tfidf.transform(X_test.astype(str))
        
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_tfidf, y_train)
        
        print("Running TF-IDF + LR inference on test set...")
        tfidf_preds = lr_model.predict(X_test_tfidf)
        test_df['tfidf_pred'] = tfidf_preds
        
        tfidf_metrics = get_metrics(y_test, tfidf_preds, "TF-IDF + LR")
        metrics['TF-IDF + LR'] = tfidf_metrics
        print(f"TF-IDF + LR Accuracy: {tfidf_metrics['accuracy']:.4f}")
        
        print("Saving TF-IDF model...")
        with open(os.path.join("outputs", "tfidf_model.pkl"), "wb") as f:
            pickle.dump({'vectorizer': tfidf, 'model': lr_model}, f)

        # ---------------------------------------------------------
        # Model 3: TextBlob
        # ---------------------------------------------------------
        print("\n--- Model 3: TextBlob ---")
        
        print("Running TextBlob inference on test set...")
        textblob_preds = []
        for text in X_test.astype(str):
            score = TextBlob(text).sentiment.polarity
            textblob_preds.append('positive' if score > 0 else 'negative')
            
        test_df['textblob_pred'] = textblob_preds
        textblob_metrics = get_metrics(y_test, textblob_preds, "TextBlob")
        metrics['TextBlob'] = textblob_metrics
        print(f"TextBlob Accuracy: {textblob_metrics['accuracy']:.4f}")
        
        # ---------------------------------------------------------
        # Save Results
        # ---------------------------------------------------------
        print("\nSaving results...")
        results_path = os.path.join("outputs", "sentiment_results.csv")
        test_df.to_csv(results_path, index=False)
        print(f"Saved predictions to {results_path}")
        
        metrics_path = os.path.join("outputs", "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
        
        print("\n--- Final Model Comparison (Accuracy) ---")
        print(f"VADER: {vader_metrics['accuracy']:.4f}")
        print(f"TF-IDF + LR: {tfidf_metrics['accuracy']:.4f}")
        print(f"TextBlob: {textblob_metrics['accuracy']:.4f}")

        print("NLP Models pipeline completed successfully.")
        
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        
    print("Finished sentiment_analysis.py")

if __name__ == "__main__":
    main()
