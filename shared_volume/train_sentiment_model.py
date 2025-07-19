#!/usr/bin/env python3
"""
Train sentiment analysis model using news sentiment data
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle

# Paths
UPLOADS_DIR = '/app/shared_data/uploads'
MODELS_DIR = '/app/shared_data/models'

def load_sentiment_data():
    """Load and preprocess news sentiment data"""
    print("📰 Loading news sentiment data...")
    
    file_path = os.path.join(UPLOADS_DIR, 'news_sentiment_updated.json')
    
    if not os.path.exists(file_path):
        print(f"❌ Sentiment data file not found: {file_path}")
        return None
    
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    records = []
    for date, companies in data.items():
        for company, articles in companies.items():
            for article in articles:
                try:
                    records.append({
                        'date': article.get('date', ''),
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'publisher': article.get('publisher', ''),
                        'symbols': article.get('symbols', []),
                        'sentiment': article.get('sentiment', 0.0),
                        'company': company
                    })
                except Exception as e:
                    print(f"Warning: Skipping article due to error: {e}")
                    continue
    
    df = pd.DataFrame(records)
    print(f"📊 Loaded {len(df)} articles")
    
    # Create sentiment labels
    def categorize_sentiment(sentiment_score):
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)
    
    # Combine title and content for text features
    df['text'] = df['title'] + ' ' + df['content']
    
    # Clean text
    df['text'] = df['text'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    df['text'] = df['text'].str.strip()
    
    # Remove rows with empty text
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 10]
    
    print(f"📊 Final dataset: {len(df)} articles")
    print(f"📊 Sentiment distribution: {df['sentiment_label'].value_counts().to_dict()}")
    
    return df

def train_sentiment_model(df):
    """Train sentiment analysis model"""
    print("🤖 Training sentiment analysis model...")
    
    # Prepare features and target
    X = df['text']
    y = df['sentiment_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF features
    print("📝 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    print("🎯 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Model accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = model.feature_importances_
    
    # Get top features
    top_features_idx = np.argsort(feature_importance)[-20:]
    top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
    
    print("\n🔝 Top 20 important features:")
    for feature, importance in reversed(top_features):
        print(f"  {feature}: {importance:.4f}")
    
    return model, vectorizer, {
        'accuracy': float(accuracy),
        'feature_names': feature_names.tolist(),
        'feature_importance': feature_importance.tolist(),
        'top_features': [(str(feature), float(importance)) for feature, importance in top_features]
    }

def save_model(model, vectorizer, metadata, model_name='sentiment_model'):
    """Save the trained model and metadata"""
    print(f"💾 Saving model as {model_name}...")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    
    # Save vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, f'{model_name}_vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save metadata
    metadata_path = os.path.join(MODELS_DIR, f'{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Vectorizer saved to {vectorizer_path}")
    print(f"✅ Metadata saved to {metadata_path}")

def main():
    """Main training function"""
    print("🚀 Starting sentiment analysis model training...")
    
    # Load data
    df = load_sentiment_data()
    if df is None:
        return
    
    # Train model
    model, vectorizer, metadata = train_sentiment_model(df)
    
    # Save model
    save_model(model, vectorizer, metadata, 'sentiment_analysis_model')
    
    print("🎉 Sentiment analysis model training completed!")

if __name__ == "__main__":
    main() 