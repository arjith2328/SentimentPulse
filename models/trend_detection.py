import os
import json
import traceback
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def get_top_keywords(text_series, n=50):
    if text_series.empty:
        return []
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_series.astype(str))
    sum_words = tfidf_matrix.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in tfidf.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def main():
    print("Starting trend_detection.py: Trend Detection Module 3")
    try:
        data_path = os.path.join("outputs", "cleaned_tweets.csv")
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}. Please run Module 1 first.")
            return

        print("Loading cleaned tweets...")
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['cleaned_text'])
        
        # Taking a sample if dataset is too large to speed up LDA
        if len(df) > 50000:
            df_sample = df.sample(50000, random_state=42)
        else:
            df_sample = df

        print("Applying LDA Topic Modeling...")
        cv = CountVectorizer(max_features=5000, stop_words='english')
        dtm = cv.fit_transform(df_sample['cleaned_text'].astype(str))
        
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda.fit(dtm)
        
        topics_dict = {}
        for index, topic in enumerate(lda.components_):
            top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topics_dict[f"Topic_{index+1}"] = top_words
            
        print("\n--- Top 5 Topics Found ---")
        for i in range(1, 6):
            print(f"Topic {i}: {', '.join(topics_dict[f'Topic_{i}'])}")

        print("\nSaving topics to topics.json...")
        with open(os.path.join("outputs", "topics.json"), "w") as f:
            json.dump(topics_dict, f, indent=4)

        print("Applying TF-IDF to find trending keywords...")
        overall_keywords = get_top_keywords(df['cleaned_text'], 50)
        
        pos_tweets = df[df['sentiment'] == 'positive']['cleaned_text']
        pos_keywords = get_top_keywords(pos_tweets, 20)
        
        neg_tweets = df[df['sentiment'] == 'negative']['cleaned_text']
        neg_keywords = get_top_keywords(neg_tweets, 20)
        
        print("Preparing trending keywords data...")
        keywords_data = []
        for word, freq in overall_keywords:
            keywords_data.append({'keyword': word, 'frequency': freq, 'sentiment_type': 'overall'})
        for word, freq in pos_keywords:
            keywords_data.append({'keyword': word, 'frequency': freq, 'sentiment_type': 'positive'})
        for word, freq in neg_keywords:
            keywords_data.append({'keyword': word, 'frequency': freq, 'sentiment_type': 'negative'})
            
        keywords_df = pd.DataFrame(keywords_data)
        
        keywords_path = os.path.join("outputs", "trending_keywords.csv")
        keywords_df.to_csv(keywords_path, index=False)
        print(f"Saved trending keywords to {keywords_path}")

        print("Trend detection pipeline completed successfully.")
        
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        
    print("Finished trend_detection.py")

if __name__ == "__main__":
    main()
