from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import io
import numpy as np
from datetime import datetime
import re
import openai
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
SHARED_DATA_DIR = '/app/shared_data'
RESULTS_FOLDER = os.path.join(SHARED_DATA_DIR, 'results')
IMAGES_FOLDER = os.path.join(SHARED_DATA_DIR, 'images')
VECTOR_DB_FOLDER = os.path.join(SHARED_DATA_DIR, 'vector_db')

# Ensure directories exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key':
    openai.api_key = OPENAI_API_KEY
    print("OpenAI API key configured successfully.")
else:
    print("Warning: OpenAI API key not set. Using fallback responses.")
    openai = None

# Global storage for vector database (in production, use a proper vector DB like Pinecone, Weaviate, etc.)
user_vector_db = {}

# Store conversation history for each user
conversation_history = {}

class VectorDatabase:
    """Simple in-memory vector database with user isolation"""
    
    def __init__(self):
        self.collections = {}
    
    def create_user_collection(self, user_id: str):
        """Create a new collection for a user"""
        if user_id not in self.collections:
            self.collections[user_id] = {
                'documents': [],
                'embeddings': [],
                'metadata': []
            }
    
    def add_document(self, user_id: str, text: str, metadata: Dict[str, Any], embedding: List[float] = None):
        """Add a document to user's collection"""
        if user_id not in self.collections:
            self.create_user_collection(user_id)
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._get_embedding(text)
        else:
            embedding = embedding
        
        self.collections[user_id]['documents'].append(text)
        self.collections[user_id]['embeddings'].append(embedding)
        self.collections[user_id]['metadata'].append(metadata)
    
    def search(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if user_id not in self.collections:
            return []
        
        query_embedding = self._get_embedding(query)
        embeddings = self.collections[user_id]['embeddings']
        documents = self.collections[user_id]['documents']
        metadata = self.collections[user_id]['metadata']
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top_k results
        similarities.sort(reverse=True)
        results = []
        for similarity, idx in similarities[:top_k]:
            results.append({
                'text': documents[idx],
                'metadata': metadata[idx],
                'similarity': similarity
            })
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        if openai is None:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to list of floats
            embedding = [float(b) / 255.0 for b in hash_bytes] * 60  # Repeat to get 1536 dimensions
            return embedding[:1536]
        
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 1536
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# Initialize vector database
vector_db = VectorDatabase()

def save_image_to_shared_volume(image_data: str, user_id: str, image_type: str) -> Optional[str]:
    """Save base64 image to shared volume and return file path"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Create user-specific folder
        user_images_folder = os.path.join(IMAGES_FOLDER, user_id)
        os.makedirs(user_images_folder, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_type}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(user_images_folder, filename)
        
        # Save image
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def create_analysis_context(results: Dict[str, Any], user_id: str) -> List[str]:
    """Create context documents from analysis results for vector database"""
    context_docs = []
    
    # Model information
    if 'model_info' in results:
        model_info = results['model_info']
        model_doc = f"Model Type: {model_info.get('model_type', 'Unknown')}. "
        if 'feature_names' in model_info:
            features = ', '.join(model_info['feature_names'])
            model_doc += f"Features: {features}. "
        if 'analyzed_at' in model_info:
            model_doc += f"Analyzed at: {model_info['analyzed_at']}."
        
        context_docs.append(model_doc)
    
    # Performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        metrics_doc = "Performance Metrics: "
        for key, value in metrics.items():
            metrics_doc += f"{key}: {value}, "
        context_docs.append(metrics_doc.rstrip(', '))
    
    # Data summary
    if 'data_summary' in results:
        data_summary = results['data_summary']
        data_doc = f"Dataset: Shape {data_summary.get('shape', 'Unknown')}. "
        if 'columns' in data_summary:
            columns = ', '.join(data_summary['columns'])
            data_doc += f"Columns: {columns}."
        context_docs.append(data_doc)
    
    # Feature importance
    if 'feature_importance' in results:
        importance = results['feature_importance']
        if isinstance(importance, dict):
            importance_doc = "Feature Importance: "
            for feature, score in importance.items():
                importance_doc += f"{feature}: {score}, "
            context_docs.append(importance_doc.rstrip(', '))
    
    # LIME Analysis Information
    lime_doc = "LIME Analysis: The analysis includes LIME (Local Interpretable Model-agnostic Explanations) visualizations that show local feature importance for individual predictions. LIME helps understand which words or features are most important for specific sentiment predictions."
    context_docs.append(lime_doc)
    
    # Visualization Summary
    if 'images' in results:
        viz_count = len(results['images'])
        viz_doc = f"Visualizations: The analysis generated {viz_count} comprehensive visualizations including LIME explanations, attention analysis plots, feature importance charts, correlation heatmaps, performance metrics, and detailed model explanations."
        context_docs.append(viz_doc)
        
        # Add specific visualization types
        viz_types = []
        if viz_count >= 6:  # Time series models
            viz_types.extend([
                "Enhanced Feature Importance Plot with color-coded bars and value labels",
                "Time Series Predictions with Confidence Intervals",
                "Feature Correlation Matrix with masked upper triangle",
                "Feature Distribution Analysis with 6-panel plots",
                "Model Performance Metrics visualization",
                "Time Series Decomposition (trend, seasonality, residuals)",
                "SHAP Summary Plot showing feature contributions",
                "SHAP Dependence Plots for top features",
                "SHAP Force Plots for sample predictions",
                "SHAP Waterfall Plot for individual predictions"
            ])
        elif viz_count >= 7:  # Text/sentiment models
            viz_types.extend([
                "Enhanced Word Importance Plot with top 25 words",
                "Multi-Class Sentiment Distributions",
                "Classification Performance Metrics",
                "Word Frequency Analysis",
                "Sentiment Trend Analysis",
                "Confusion Matrix visualization",
                "Word Cloud Analysis",
                "LIME Explanation for individual predictions",
                "Attention Analysis for transformer models",
                "Word Sentiment Association plots",
                "Feature Importance charts",
                "Model Performance visualizations"
            ])
        
        if viz_types:
            viz_details = "Detailed Visualizations: " + ". ".join(viz_types) + "."
            context_docs.append(viz_details)
    
    # LIME analysis summary
    if 'lime_analysis' in results and results['lime_analysis']:
        lime_analysis = results['lime_analysis']
        if 'top_features' in lime_analysis and lime_analysis['top_features']:
            top_feats = lime_analysis['top_features']
            lime_doc = "Top LIME Features: " + ", ".join([f"{f['feature']} (importance={f['importance']:.4f})" for f in top_feats])
            context_docs.append(lime_doc)
            
            # Add detailed LIME analysis
            lime_details = f"LIME Analysis Details: Analyzed {lime_analysis.get('total_features_analyzed', 0)} features for local explanations. "
            lime_details += f"Top feature '{top_feats[0]['feature']}' has the highest importance ({top_feats[0]['importance']:.4f}), indicating it has the strongest influence on this specific prediction."
            context_docs.append(lime_details)
        elif 'error' in lime_analysis:
            lime_doc = f"LIME Analysis Error: {lime_analysis['error']}"
            context_docs.append(lime_doc)
    
    # Visualization descriptions
    if 'images' in results:
        viz_descriptions = [
            "Feature Importance Analysis: Shows which variables have the most impact on model predictions",
            "SHAP Summary Plot: Displays how each feature contributes to individual predictions",
            "Feature Correlation Heatmap: Shows relationships between different features",
            "Feature Distributions: Displays the spread and shape of feature values",
            "Model Performance Analysis: Shows accuracy, precision, recall, and other metrics",
            "Model Explainability Summary: Provides overall insights into model behavior"
        ]
        
        for i, description in enumerate(viz_descriptions):
            if i < len(results['images']):
                context_docs.append(description)
    
    return context_docs

def save_results_to_shared_volume(results: Dict[str, Any], user_id: str) -> Optional[str]:
    """Save analysis results as JSON file to shared volume"""
    try:
        # Create user-specific results folder
        user_results_folder = os.path.join(RESULTS_FOLDER, user_id)
        os.makedirs(user_results_folder, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        file_path = os.path.join(user_results_folder, filename)
        
        # Save results as JSON
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Saved results to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving results to shared volume: {e}")
        return None

def store_results_in_vector_db(results: Dict[str, Any], user_id: str):
    """Store analysis results in vector database"""
    try:
        # Save results to shared volume first
        save_results_to_shared_volume(results, user_id)
        
        # Handle different types of results
        result_type = results.get('type', 'unknown')
        
        # Check for data_statistics in the new format
        if 'data_statistics' in results:
            data_stats = results['data_statistics']
            data_type = data_stats.get('data_type', 'unknown')
            visualizations = data_stats.get('visualizations', [])
            
            data_doc = f"Data Statistics Analysis: Data type: {data_type}. "
            data_doc += "Analysis includes: comprehensive data overview, sentiment distribution histogram, per-asset sentiment boxplots, keyword frequency analysis, word sentiment associations, asset distribution charts, and text length distribution. "
            data_doc += "Data Overview: Shows dataset statistics including total articles, average title length, date range, distinct assets, and column information. "
            data_doc += "Sentiment Distribution: Histogram showing the distribution of sentiment scores across all articles with neutral (0) as reference point. "
            data_doc += "Per-Asset Sentiment: Boxplot analysis showing sentiment patterns for the top 25 assets by article count. "
            data_doc += "Keyword Insights: Horizontal bar chart showing the top 15 most frequent words in article titles. "
            data_doc += "Word Sentiment Associations: Two-panel chart showing top 10 words driving positive sentiment (green bars) and negative sentiment (red bars) with exact scores. "
            data_doc += "Asset Distribution: Horizontal bar chart showing the top 20 assets by article count. "
            data_doc += "Text Length Distribution: Histogram showing the distribution of article title lengths with mean, median, and standard deviation markers. "
            data_doc += f" Visualizations generated: {', '.join(visualizations)}."
            
            # Add actual plot data to the document
            plot_data = data_stats.get('plot_data', {})
            if 'word_sentiment' in plot_data:
                word_sentiment = plot_data['word_sentiment']
                pos_words = word_sentiment.get('positive_words', [])
                neg_words = word_sentiment.get('negative_words', [])
                
                if pos_words:
                    data_doc += f" Top 10 positive words: {', '.join([f'{word}({score:.2f})' for word, score in pos_words[:5]])}. "
                if neg_words:
                    data_doc += f" Top 10 negative words: {', '.join([f'{word}({score:.2f})' for word, score in neg_words[:5]])}. "
            
            if 'keywords' in plot_data:
                keywords = plot_data['keywords'].get('top_keywords', [])
                if keywords:
                    data_doc += f" Top 15 keywords: {', '.join([f'{word}({count})' for word, count in keywords[:5]])}. "
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'data_statistics',
                'data_type': data_type,
                'visualizations': visualizations,
                'timestamp': data_stats.get('timestamp', datetime.now().isoformat())
            }
            vector_db.add_document(user_id, data_doc, metadata)
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"data_stats_{i+1}"
                    
                    # Handle both base64 and file path formats
                    if isinstance(image_data, dict) and 'image' in image_data:
                        # Base64 image data
                        base64_data = image_data['image']
                        image_path = save_image_to_shared_volume(base64_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
                    elif isinstance(image_data, str):
                        # Direct base64 string
                        image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
        
        elif result_type == 'data_statistics':
            # Store data statistics results
            data_doc = f"Data Statistics Analysis: Data type: {results.get('data_type', 'unknown')}. "
            data_doc += "Analysis includes: word sentiment associations, keyword insights, sentiment distribution, per-asset sentiment analysis, and comprehensive data overview. "
            data_doc += "Word sentiment analysis shows the top 10 words driving positive and negative sentiment in the dataset. "
            data_doc += "Keyword insights show the most frequent words in article titles. "
            data_doc += "Asset-specific analysis shows sentiment patterns by financial asset/ticker."
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'data_statistics',
                'data_type': results.get('data_type', 'unknown'),
                'insights': results.get('insights', {}),
                'timestamp': results.get('timestamp', datetime.now().isoformat())
            }
            vector_db.add_document(user_id, data_doc, metadata)
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"data_stats_{i+1}"
                    
                    # Handle both base64 and file path formats
                    if isinstance(image_data, dict) and 'image' in image_data:
                        # Base64 image data
                        base64_data = image_data['image']
                        image_path = save_image_to_shared_volume(base64_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
                    elif isinstance(image_data, str):
                        # Direct base64 string
                        image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
        
        elif result_type == 'xai_analysis':
            # Clear previous XAI results for this user
            if user_id in vector_db.collections:
                # Remove old XAI-related documents
                old_docs = []
                old_embeddings = []
                old_metadata = []
                
                for i, metadata_doc in enumerate(vector_db.collections[user_id]['metadata']):
                    if metadata_doc.get('doc_type') in ['xai_analysis', 'xai_visualization']:
                        # Skip this document (don't add to new lists)
                        continue
                    else:
                        # Keep this document
                        old_docs.append(vector_db.collections[user_id]['documents'][i])
                        old_embeddings.append(vector_db.collections[user_id]['embeddings'][i])
                        old_metadata.append(metadata_doc)
                
                # Replace the collection with only non-XAI documents
                vector_db.collections[user_id]['documents'] = old_docs
                vector_db.collections[user_id]['embeddings'] = old_embeddings
                vector_db.collections[user_id]['metadata'] = old_metadata
            
            # Store new XAI analysis results
            xai_data = results.get('visualizations', {})
            example_index = results.get('example_index', 'N/A')
            model_type = results.get('model_type', 'N/A')
            
            xai_doc = f"XAI Analysis Results: Example index: {example_index}. "
            xai_doc += f"Model type: {model_type}. "
            xai_doc += f"Visualizations generated: {', '.join(xai_data.keys()) if isinstance(xai_data, dict) else 'multiple visualizations'}. "
            
            # Add specific descriptions for each visualization type
            if 'lime' in xai_data and xai_data['lime']:
                xai_doc += "LIME Analysis: Local Interpretable Model-agnostic Explanations showing which words are most important for the specific sentiment prediction. "
                xai_doc += "LIME highlights the top contributing words that drive the model's decision for this particular text example. "
            
            if 'attention' in xai_data and xai_data['attention']:
                xai_doc += "Attention Analysis: Shows attention weights from the transformer model, highlighting which words the model focuses on when making predictions. "
            
            if 'confidence' in xai_data and xai_data['confidence']:
                xai_doc += "Prediction Confidence: Visualizes the confidence levels of the model's predictions across different sentiment categories. "
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'xai_analysis',
                'example_index': example_index,
                'model_type': model_type,
                'visualizations': list(xai_data.keys()) if isinstance(xai_data, dict) else [],
                'timestamp': results.get('timestamp', datetime.now().isoformat())
            }
            vector_db.add_document(user_id, xai_doc, metadata)
            
            # Store XAI visualizations
            if isinstance(xai_data, dict):
                for viz_type, viz_data in xai_data.items():
                    if isinstance(viz_data, str):  # Base64 image
                        image_path = save_image_to_shared_volume(viz_data, user_id, f"xai_{viz_type}")
                        if image_path:
                            viz_doc = f"XAI Visualization: {viz_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'xai_visualization',
                                'visualization_type': viz_type,
                                'image_path': image_path,
                                'example_index': example_index
                            }
                            vector_db.add_document(user_id, viz_doc, metadata)
        
        else:
            # Handle legacy results format
            # Create context documents
            context_docs = create_analysis_context(results, user_id)
            
            # Store each document in vector database
            for i, doc in enumerate(context_docs):
                metadata = {
                    'user_id': user_id,
                    'doc_type': 'analysis_context',
                    'index': i,
                    'timestamp': datetime.now().isoformat()
                }
                vector_db.add_document(user_id, doc, metadata)
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"viz_{i+1}"
                    image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                    
                    if image_path:
                        # Create image metadata document
                        image_doc = f"Visualization {i+1}: {image_type} stored at {image_path}"
                        metadata = {
                            'user_id': user_id,
                            'doc_type': 'visualization',
                            'image_path': image_path,
                            'image_type': image_type,
                            'index': i
                        }
                        vector_db.add_document(user_id, image_doc, metadata)
        
        print(f"Stored results in vector database for user {user_id}, type: {result_type}")
        
    except Exception as e:
        print(f"Error storing results in vector database: {e}")

def add_to_conversation_history(user_id: str, question: str, answer: str):
    """Add a question-answer pair to user's conversation history"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        'question': question,
        'answer': answer,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 10 exchanges to prevent context from getting too long
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]

def get_conversation_context(user_id: str) -> str:
    """Get recent conversation context for a user"""
    if user_id not in conversation_history or not conversation_history[user_id]:
        return ""
    
    context_parts = []
    for exchange in conversation_history[user_id][-5:]:  # Last 5 exchanges
        context_parts.append(f"Previous Q: {exchange['question']}")
        context_parts.append(f"Previous A: {exchange['answer']}")
    
    return "\n".join(context_parts)

def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    if user_id in conversation_history:
        del conversation_history[user_id]

def generate_rag_response(question: str, user_id: str) -> str:
    """Generate response using RAG approach"""
    try:
        # Search for relevant context
        relevant_docs = vector_db.search(user_id, question, top_k=5)  # Increased from 3 to 5
        
        print(f"DEBUG: Found {len(relevant_docs)} relevant documents for question: '{question}'")
        for i, doc in enumerate(relevant_docs):
            print(f"DEBUG: Doc {i+1}: {doc['text'][:100]}... (similarity: {doc['similarity']:.3f})")
        
        if not relevant_docs:
            return "I don't have access to your analysis results yet. Please upload and analyze a model first."
        
        # Build context from relevant documents
        context = "\n".join([doc['text'] for doc in relevant_docs])
        
        # Get conversation history
        conversation_context = get_conversation_context(user_id)
        
        print(f"DEBUG: Context length: {len(context)} characters")
        print(f"DEBUG: Context preview: {context[:200]}...")
        print(f"DEBUG: Conversation context length: {len(conversation_context)} characters")
        
        # Create system prompt
        system_prompt = """You are an AI assistant specialized in Explainable AI (XAI) analysis. 
        You help users understand their machine learning models, feature importance, predictions, and performance metrics.
        Always provide clear, helpful explanations based on the context provided.
        If you don't have enough information to answer a question, say so politely.
        When discussing sentiment analysis, be specific about word importance and their impact on sentiment predictions.
        You have access to previous conversation context, so you can build on previous questions and answers."""
        
        # Create user prompt with context and conversation history
        user_prompt = f"""Based on the following analysis context and conversation history, please answer this question: {question}

Analysis Context:
{context}

{conversation_context if conversation_context else ""}

Please provide a clear, helpful response based on the context above. If the context contains specific data (like word sentiment analysis, attention scores, performance metrics), use that information in your response. You can reference previous questions and build on the conversation. Do NOT mention SHAP analysis as it is not available in this system."""
        
        print(f"DEBUG: Using OpenAI: {openai is not None}")
        
        # Generate response using OpenAI or fallback
        if openai is None:
            # Enhanced fallback response based on actual context
            question_lower = question.lower()
            
            # Extract specific information from context
            model_info = ""
            features = ""
            performance = ""
            data_info = ""
            word_sentiment = ""
            attention_analysis = ""
            
            for doc in relevant_docs:
                doc_text = doc['text'].lower()
                if "model type:" in doc_text:
                    model_info = doc['text']
                elif "features:" in doc_text:
                    features = doc['text']
                elif "performance metrics:" in doc_text:
                    performance = doc['text']
                elif "dataset:" in doc_text:
                    data_info = doc['text']
                elif "word sentiment" in doc_text or "negative sentiment" in doc_text or "positive sentiment" in doc_text:
                    word_sentiment = doc['text']
                elif "attention analysis" in doc_text or "top important tokens" in doc_text:
                    attention_analysis = doc['text']
            
            # Generate specific responses based on question and available context
            if "model" in question_lower and "type" in question_lower:
                if model_info:
                    return f"Based on your analysis: {model_info}"
                else:
                    return "I can see you have a machine learning model analyzed, but the specific model type information isn't available in the current context."
            
            elif "feature" in question_lower and ("important" in question_lower or "key" in question_lower):
                if features:
                    return f"Based on your analysis: {features}"
                else:
                    return "Feature importance analysis shows which variables have the most impact on your model's predictions. The specific features and their importance scores are detailed in your analysis results."
            
            elif "performance" in question_lower or "well" in question_lower or "accuracy" in question_lower:
                if performance:
                    return f"Based on your analysis: {performance}"
                else:
                    return "Model performance metrics are available in your analysis results. These metrics help evaluate how well your model is performing on the given task."
            
            elif "data" in question_lower or "dataset" in question_lower:
                if data_info:
                    return f"Based on your analysis: {data_info}"
                else:
                    return "Your dataset information is available in the analysis results, including the data shape and features used for training."
            
            elif "visualization" in question_lower or "plot" in question_lower or "chart" in question_lower:
                viz_count = len([doc for doc in relevant_docs if "visualization" in doc['text'].lower()])
                if viz_count > 0:
                    return f"Your analysis includes {viz_count} visualizations that help explain your model's behavior. These include:\n\n1. **Feature Importance Plots** - Shows which features are most important for predictions\n2. **Word Frequency Analysis** - Displays most common words in titles/text\n3. **Sentiment Word Analysis** - Shows positive vs negative words\n4. **Asset-Specific Analysis** - Sentiment distribution by asset\n5. **Prediction Distribution** - Histogram of model predictions\n\nYou can view these visualizations in the dashboard. The AI assistant can describe what each visualization shows and help you interpret the results."
                else:
                    return "Your analysis includes several visualizations to help understand your model's behavior. These visualizations show feature importance, word frequency analysis, sentiment patterns, and prediction distributions. You can view them in the dashboard and ask me to explain what they mean."
            
            elif "negative words" in question_lower or "positive words" in question_lower or "sentiment words" in question_lower or "most negative" in question_lower:
                # Check for data statistics documents specifically
                data_stats_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'data_statistics']
                data_viz_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'data_statistics_visualization']
                
                if data_stats_docs:
                    return f"Based on your data statistics analysis, I can see word sentiment associations that show the top words driving positive and negative sentiment in your dataset. The analysis includes visualizations showing the most frequent words and their sentiment scores. However, I need to see the specific word sentiment plots to tell you the exact top negative words. Please check the data statistics visualizations in your dashboard for the detailed word sentiment analysis."
                else:
                    return "I can see you have data statistics available that include word sentiment analysis. This analysis shows which words are most strongly associated with positive and negative sentiment in your dataset. To see the specific top negative words, please generate data statistics and check the word sentiment associations plots in your dashboard."
            
            elif "attention" in question_lower or "tokens" in question_lower:
                if attention_analysis:
                    return f"Based on your attention analysis: {attention_analysis}"
                else:
                    return "Attention analysis shows which words the model focuses on when making sentiment predictions. This helps understand how the model processes and weighs different parts of the text."
            
            elif "word" in question_lower or "text" in question_lower or "title" in question_lower:
                return "Your analysis includes word-based visualizations that show:\n\n1. **Word Frequency Analysis** - Most common words in news titles\n2. **Sentiment Word Analysis** - Positive vs negative words and their frequencies\n3. **Asset-Specific Word Analysis** - Top words for each asset/ticker\n\nThese visualizations help identify which words are most associated with positive or negative sentiment, and how different assets are discussed in the news."
            
            elif "asset" in question_lower or "ticker" in question_lower:
                return "Your analysis includes asset-specific visualizations showing:\n\n1. **Asset Sentiment Distribution** - Average sentiment scores for each asset\n2. **Article Count by Asset** - Number of news articles per asset\n3. **Asset-Specific Word Analysis** - Most common words in news about each asset\n\nThis helps identify which assets have the most positive/negative news coverage and what topics are most discussed for each asset."
            
            elif "xai" in question_lower or "lime" in question_lower or "attention" in question_lower or "explain" in question_lower:
                # Check for XAI analysis results
                xai_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'xai_analysis']
                attention_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'attention_analysis']
                
                print(f"DEBUG: Found {len(xai_docs)} XAI docs and {len(attention_docs)} attention docs")
                print(f"DEBUG: All doc types: {[doc['metadata'].get('doc_type', 'unknown') for doc in relevant_docs]}")
                
                if xai_docs:
                    xai_doc = xai_docs[0]
                    example_index = xai_doc['metadata'].get('example_index', '')
                    model_type = xai_doc['metadata'].get('model_type', '')
                    visualizations = xai_doc['metadata'].get('visualizations', [])
                    
                    response = f"Based on your XAI analysis:\n\n"
                    response += f"**Example Index:** {example_index}\n\n"
                    response += f"**Model Type:** {model_type}\n\n"
                    response += f"**Visualizations Generated:** {', '.join(visualizations)}\n\n"
                    
                    if 'lime' in visualizations:
                        response += "**LIME Analysis:** Shows which words in the text are most important for the model's prediction. It highlights the key features that influenced the sentiment classification.\n\n"
                    
                    if 'attention' in visualizations:
                        response += "**Enhanced Attention Analysis:** Shows which words the model focuses on when making its prediction. The attention weights indicate which parts of the text are most relevant for the sentiment classification.\n\n"
                        
                        # Add detailed attention insights if available
                        if attention_docs:
                            attention_doc = attention_docs[0]
                            insights = attention_doc['metadata'].get('insights', {})
                            
                            # Top tokens information
                            top_tokens = insights.get('top_tokens', [])
                            if top_tokens:
                                response += "**Top Important Tokens:**\n"
                                for i, (token, score) in enumerate(top_tokens[:5], 1):
                                    response += f"{i}. '{token}' (attention score: {score:.3f})\n"
                                response += "\n"
                            
                            # Attention metrics
                            max_score = insights.get('max_attention_score', 0)
                            concentration = insights.get('attention_concentration', 'unknown')
                            sentiment_corr = insights.get('sentiment_correlation', 'neutral')
                            
                            response += f"**Attention Insights:**\n"
                            response += f"- The model focuses most on '{top_tokens[0][0] if top_tokens else 'unknown'}' with attention score {max_score:.3f}\n"
                            response += f"- Attention pattern is {concentration}\n"
                            response += f"- Sentiment correlation: {sentiment_corr}\n\n"
                            
                            # Pattern analysis
                            if concentration == 'concentrated':
                                response += "**Pattern Analysis:** The model shows concentrated attention, focusing heavily on specific key terms. This suggests the model is making decisions based on a few important words rather than considering the entire text equally.\n\n"
                            else:
                                response += "**Pattern Analysis:** The model shows distributed attention across multiple tokens. This suggests the model is considering the broader context and multiple words contribute to the sentiment prediction.\n\n"
                    
                    return response
                else:
                    # Check if we have any XAI-related documents
                    xai_related_docs = [doc for doc in relevant_docs if any(term in doc['text'].lower() for term in ['xai', 'lime', 'attention', 'explain'])]
                    if xai_related_docs:
                        return f"I can see XAI-related information in your analysis. You have {len(xai_related_docs)} documents that mention XAI, LIME, or attention analysis. However, I need more specific XAI analysis results to provide detailed insights. Please run XAI analysis on specific examples to get detailed explanations."
                    else:
                        return "I can see you have XAI analysis capabilities available. XAI (Explainable AI) helps understand how the model makes predictions. You can run XAI analysis on specific examples to see LIME explanations and enhanced attention visualizations that show which words are most important for the model's decisions."
            
            elif "data statistics" in question_lower or "statistics" in question_lower or "data analysis" in question_lower:
                # Check for data statistics results
                data_stats_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'data_statistics']
                data_viz_docs = [doc for doc in relevant_docs if doc['metadata'].get('doc_type') == 'data_statistics_visualization']
                
                print(f"DEBUG: Found {len(data_stats_docs)} data stats docs and {len(data_viz_docs)} data viz docs")
                
                if data_stats_docs:
                    data_doc = data_stats_docs[0]
                    data_type = data_doc['metadata'].get('data_type', 'unknown')
                    insights = data_doc['metadata'].get('insights', {})
                    
                    response = f"Based on your Data Statistics Analysis:\n\n"
                    response += f"**Data Type:** {data_type}\n\n"
                    
                    if insights:
                        response += "**Analysis Includes:**\n"
                        for insight_type, insight_data in insights.items():
                            response += f"- {insight_type.replace('_', ' ').title()}\n"
                        response += "\n"
                    
                    response += f"**Visualizations Generated:** {len(data_viz_docs)} data statistics plots\n\n"
                    
                    response += "**Available Insights:**\n"
                    response += "1. **Title Insights** - Analysis of article titles and their patterns\n"
                    response += "2. **Word Frequency Analysis** - Most common words in the dataset\n"
                    response += "3. **Sentiment Distribution** - Distribution of sentiment scores\n"
                    response += "4. **Asset-Specific Analysis** - Sentiment analysis by asset/ticker\n"
                    response += "5. **Word Cloud** - Visual representation of most frequent words\n"
                    response += "6. **Financial Terms Analysis** - Key financial terms and their frequencies\n\n"
                    
                    response += "These visualizations help understand the structure and patterns in your sentiment analysis dataset, including word frequencies, sentiment distributions, and asset-specific insights."
                    
                    return response
                else:
                    return "I can see you have data statistics available. The analysis includes comprehensive insights about your dataset, including title analysis, word frequency patterns, sentiment distributions, and asset-specific analysis. These visualizations help understand the structure and patterns in your sentiment analysis data."
            
            else:
                # Provide a summary of available information
                available_info = []
                if model_info:
                    available_info.append("model type and features")
                if performance:
                    available_info.append("performance metrics")
                if data_info:
                    available_info.append("dataset information")
                if any("visualization" in doc['text'].lower() for doc in relevant_docs):
                    available_info.append("visualizations")
                
                if available_info:
                    info_list = ", ".join(available_info)
                    return f"Based on your analysis, I can help you understand: {info_list}. Please ask specific questions about these aspects of your model."
                else:
                    return "I can see you have analysis results available. Please ask specific questions about your model type, features, performance, or visualizations."
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return "I'm having trouble connecting to the AI service. Please try again later."
        
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'ai_outputs_rag'})

@app.route('/store-data', methods=['POST'])
def store_data():
    """Store user data information"""
    try:
        data = request.json
        user_id = data.get('user_id')
        data_info = data.get('data_info', {})
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Store data info in vector database
        data_doc = f"Dataset: Shape {data_info.get('shape', 'Unknown')}. "
        if 'columns' in data_info:
            columns = ', '.join(data_info['columns'])
            data_doc += f"Columns: {columns}. "
        if 'data_type' in data_info:
            data_doc += f"Data type: {data_info['data_type']}."
        
        vector_db.add_document(
            user_id=user_id,
            text=data_doc,
            metadata={
                'doc_type': 'data_info',
                'timestamp': datetime.now().isoformat(),
                'data_info': data_info
            }
        )
        
        return jsonify({'message': 'Data information stored successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/store-results', methods=['POST'])
def store_results():
    """Store XAI results in vector database"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Store results in vector database
        store_results_in_vector_db(data, user_id)
        
        return jsonify({'message': 'Results stored successfully in vector database'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/store-attention-insights', methods=['POST'])
def store_attention_insights():
    """Store attention analysis insights for AI assistant access"""
    try:
        data = request.json
        user_id = data.get('user_id')
        attention_analysis = data.get('attention_analysis', {})
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Create attention insights document for vector database
        example_text = attention_analysis.get('example_text', '')
        insights = attention_analysis.get('insights', {})
        
        # Generate detailed attention insights document
        attention_doc = f"Attention Analysis for: '{example_text[:100]}...'\n\n"
        
        # Top tokens information
        top_tokens = insights.get('top_tokens', [])
        if top_tokens:
            attention_doc += "Top Important Tokens:\n"
            for i, (token, score) in enumerate(top_tokens[:5], 1):
                attention_doc += f"{i}. '{token}' (score: {score:.3f})\n"
            attention_doc += "\n"
        
        # Attention metrics
        max_score = insights.get('max_attention_score', 0)
        variance = insights.get('attention_variance', 0)
        concentration = insights.get('attention_concentration', 'unknown')
        sentiment_corr = insights.get('sentiment_correlation', 'neutral')
        
        attention_doc += f"Attention Metrics:\n"
        attention_doc += f"- Maximum attention score: {max_score:.3f}\n"
        attention_doc += f"- Attention variance: {variance:.3f}\n"
        attention_doc += f"- Attention pattern: {concentration}\n"
        attention_doc += f"- Sentiment correlation: {sentiment_corr}\n\n"
        
        # Store in vector database
        metadata = {
            'doc_type': 'attention_analysis',
            'example_text': example_text,
            'insights': insights,
            'timestamp': attention_analysis.get('timestamp', datetime.now().isoformat())
        }
        
        vector_db.add_document(user_id, attention_doc, metadata)
        
        return jsonify({
            'message': 'Attention insights stored successfully',
            'insights_count': len(top_tokens) if top_tokens else 0
        })
        
    except Exception as e:
        print(f"Error storing attention insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<user_id>', methods=['GET'])
def get_results(user_id):
    """Get stored results for a user"""
    try:
        if user_id not in vector_db.collections:
            return jsonify({'error': 'No results found for user'}), 404
        
        # Return summary of stored documents
        collection = vector_db.collections[user_id]
        
        # Safely extract document types and timestamps
        document_types = []
        timestamps = []
        
        for doc in collection['metadata']:
            if 'doc_type' in doc:
                document_types.append(doc['doc_type'])
            if 'timestamp' in doc:
                timestamps.append(doc['timestamp'])
        
        summary = {
            'user_id': user_id,
            'total_documents': len(collection['documents']),
            'document_types': list(set(document_types)),
            'last_updated': max(timestamps) if timestamps else None
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat questions using RAG approach"""
    try:
        data = request.json
        question = data.get('question')
        user_id = data.get('user_id')
        
        if not question or not user_id:
            return jsonify({'error': 'Missing question or user_id'}), 400
        
        # Check if user has any stored results
        if user_id not in vector_db.collections:
            return jsonify({'error': 'No results found for user. Please upload and analyze a model first.'}), 404
        
        # Generate RAG response
        answer = generate_rag_response(question, user_id)
        
        # Store conversation history
        add_to_conversation_history(user_id, question, answer)
        
        return jsonify({
            'answer': answer,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-user-data/<user_id>', methods=['DELETE'])
def clear_user_data(user_id):
    """Clear all data for a specific user"""
    try:
        if user_id in vector_db.collections:
            del vector_db.collections[user_id]
        
        # Clear conversation history
        clear_conversation_history(user_id)
        
        # Also clear user's image folder
        user_images_folder = os.path.join(IMAGES_FOLDER, user_id)
        if os.path.exists(user_images_folder):
            import shutil
            shutil.rmtree(user_images_folder)
        
        return jsonify({'message': f'All data cleared for user {user_id}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True) 