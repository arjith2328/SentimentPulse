import os
import traceback
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text, stop_words, lemmatizer):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbols (keep the word)
    text = re.sub(r'#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords, apply lemmatization
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def main():
    print("Starting setup_db.py: Data Pipeline Module 1")
    try:
        # Download NLTK data if not present
        print("Downloading required NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        data_path = os.path.join("data", "sentiment140.csv")
        output_csv_path = os.path.join("outputs", "cleaned_tweets.csv")
        db_path = os.path.join("outputs", "sentiment_pulse.db")
        
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}. Please place the dataset in the data folder.")
            return
            
        print("Loading sentiment140.csv...")
        col_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(data_path, names=col_names, encoding='latin-1')
        print(f"Loaded {len(df)} rows.")
        
        print("Mapping sentiment targets...")
        df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
        df['sentiment'] = df['sentiment'].fillna('neutral')
        
        print("Cleaning text (this may take a while)...")
        # If the dataset is too large to process quickly, we might sample it.
        # However, the user asked to "Load sentiment140.csv into a SQLite database"
        # We process the entire dataframe
        df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x), stop_words, lemmatizer))
        
        print("Preparing data for database insertion...")
        db_df = df[['ids', 'text', 'sentiment', 'date', 'user', 'cleaned_text']].copy()
        db_df.rename(columns={'ids': 'id'}, inplace=True)
        
        print(f"Saving to SQLite database at {db_path}...")
        engine = create_engine(f"sqlite:///{db_path}")
        db_df.to_sql('tweets', con=engine, index=False, if_exists='replace', chunksize=50000)
        
        print(f"Saving cleaned data to {output_csv_path}...")
        db_df.to_csv(output_csv_path, index=False)
        
        print("Data pipeline completed successfully.")
        
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        
    print("Finished setup_db.py")

if __name__ == "__main__":
    main()
