import os
import traceback
import pandas as pd
from sqlalchemy import create_engine
import collections

def main():
    print("Starting queries.py: Running 5 analytical queries")
    try:
        db_path = os.path.join("outputs", "sentiment_pulse.db")
        if not os.path.exists(db_path):
            print(f"Database not found at {db_path}. Please run setup_db.py first.")
            return
            
        engine = create_engine(f"sqlite:///{db_path}")
        
        print("\n--- Query 1: Sentiment Distribution Count ---")
        q1 = "SELECT sentiment, COUNT(*) as count FROM tweets GROUP BY sentiment"
        df1 = pd.read_sql(q1, engine)
        print(df1)
        
        print("\n--- Query 2: Top 10 Most Active Users ---")
        q2 = "SELECT user, COUNT(*) as tweet_count FROM tweets GROUP BY user ORDER BY tweet_count DESC LIMIT 10"
        df2 = pd.read_sql(q2, engine)
        print(df2)
        
        print("\n--- Query 3: Tweet Count by Hour of Day ---")
        # Extracts hour from string format like 'Mon Apr 06 22:19:45 PDT 2009'
        q3 = "SELECT substr(date, 12, 2) as hour, COUNT(*) as count FROM tweets GROUP BY hour ORDER BY hour"
        df3 = pd.read_sql(q3, engine)
        print(df3)
        
        print("\n--- Query 4: Average Text Length by Sentiment ---")
        q4 = "SELECT sentiment, AVG(LENGTH(text)) as avg_length FROM tweets GROUP BY sentiment"
        df4 = pd.read_sql(q4, engine)
        print(df4)
        
        print("\n--- Query 5: Most Common Words Per Sentiment ---")
        # Fetching a sample to compute most common words via Python 
        # since SQLite lacks a native string split function.
        q5 = "SELECT sentiment, cleaned_text FROM tweets WHERE cleaned_text IS NOT NULL LIMIT 100000"
        df5 = pd.read_sql(q5, engine)
        
        for sentiment in df5['sentiment'].unique():
            text_series = df5[df5['sentiment'] == sentiment]['cleaned_text'].astype(str)
            words = ' '.join(text_series).split()
            common = collections.Counter(words).most_common(10)
            print(f"\nTop 10 words for {sentiment}:")
            for word, count in common:
                print(f"  {word}: {count}")
            
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        
    print("\nFinished queries.py")

if __name__ == "__main__":
    main()
