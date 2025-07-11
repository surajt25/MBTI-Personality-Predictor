import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def extract_tfidf_features(df, text_column='clean_posts', max_features=5000):
    
    print(f"Extracting TF-IDF features from {len(df)} documents...")
    print(f"Using column: {text_column}")
    
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,            # Ignore terms that appear in less than 2 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        stop_words='english' # Additional stopword filtering
    )
    
    # Fit and transform the cleaned text
    print("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    print(f"✅ Extracted {len(feature_names)} TF-IDF features")
    print(f"📊 Feature matrix shape: {tfidf_matrix.shape}")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    feature_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    # Add your existing labels and metadata
    feature_df['type'] = df['type'].values
    feature_df['IE'] = df['IE'].values
    feature_df['NS'] = df['NS'].values
    feature_df['TF'] = df['TF'].values
    feature_df['JP'] = df['JP'].values
    
    # Optionally add original posts for reference
    feature_df['original_posts'] = df['posts'].values
    feature_df['clean_posts'] = df['clean_posts'].values
    
    return feature_df, vectorizer

def main():
    """Main function for TF-IDF feature extraction"""
    
    # Configuration
    CSV_PATH = r'mbti_preprocessed.csv'
    OUTPUT_FILE = 'mbti_tfidf_features.csv'
    VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
    MAX_FEATURES = 5000
    
    try:
        # Load your preprocessed data
        print("🔄 Loading preprocessed data...")
        df = pd.read_csv(CSV_PATH)
        print(f"📊 Data shape: {df.shape}")
        
        # Check if you have the required columns
        required_columns = ['posts', 'type', 'clean_posts', 'IE', 'NS', 'TF', 'JP']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # If clean_posts doesn't exist, run preprocessing first
        if 'clean_posts' not in df.columns:
            print("❌ 'clean_posts' column not found. Please run preprocessing first.")
            return
        
        # Extract TF-IDF features
        print("🔄 Starting TF-IDF feature extraction...")
        df = df.fillna('')
        feature_df, vectorizer = extract_tfidf_features(
            df, 
            text_column='clean_posts',
            max_features=MAX_FEATURES
        )
        
        # Save features
        print(f"💾 Saving features to {OUTPUT_FILE}...")
        feature_df.to_csv(OUTPUT_FILE, index=False)
        
        # Save vectorizer for future use
        print(f"💾 Saving vectorizer to {VECTORIZER_FILE}...")
        with open(VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Display summary
        print(f"\n✅ Feature extraction completed successfully!")
        print(f"📁 Features saved to: {OUTPUT_FILE}")
        print(f"📁 Vectorizer saved to: {VECTORIZER_FILE}")
        print(f"📊 Total features: {len(feature_df.columns)}")
        
        # Feature breakdown
        tfidf_features = [col for col in feature_df.columns if col not in ['type', 'IE', 'NS', 'TF', 'JP', 'original_posts', 'clean_posts']]
        print(f"📊 TF-IDF features: {len(tfidf_features)}")
        print(f"📊 Label columns: 5 (type, IE, NS, TF, JP)")
        print(f"📊 Text columns: 2 (original_posts, clean_posts)")
        
        # Show first few feature names
        print(f"\n🔍 Sample TF-IDF features:")
        for i, feature in enumerate(tfidf_features[:10]):
            print(f"  {i+1}. {feature}")
        
        # Show data preview
        print(f"\n👀 Data preview:")
        print(feature_df[['type', 'IE', 'NS', 'TF', 'JP'] + tfidf_features[:5]].head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()