from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from misinformation_detection import test_ai_response, preprocess_tweet, setup_langchain
import pandas as pd
from datetime import datetime
from scipy.sparse import hstack
from misinformation_detection import extract_tweet_features

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'models/twitter_misinfo_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/tfidf_vectorizer.pkl')
count_vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/count_vectorizer.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(count_vectorizer_path, 'rb') as f:
        count_vectorizer = pickle.load(f)
    
    # Set up LangChain
    langchain = setup_langchain()
    
except FileNotFoundError:
    print("Model or vectorizer file not found. Please run save_model.py first.")
    model = None
    vectorizer = None
    count_vectorizer = None
    langchain = None
except Exception as e:
    print(f"Error loading model or setting up LangChain: {e}")
    model = None
    vectorizer = None
    count_vectorizer = None
    langchain = None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/check_misinformation', methods=['POST'])
def check_misinformation():
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded',
            'prediction': 'unknown',
            'confidence': 0.5
        }), 500
        
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Preprocess the tweet text
    processed_text = preprocess_tweet(text)
    
    # Check if the text is an opinion
    is_opinion = detect_opinion(text)
    if is_opinion:
        return jsonify({
            'prediction': 'opinion',
            'confidence': 0.8,
            'explanation': 'This appears to be an opinion rather than a factual claim.'
        })
    
    # Check for scientific context
    is_scientific, scientific_explanation = detect_scientific_context(text)
    
    # Check source credibility
    has_credible_source, source_explanation = verify_source_credibility(text)
    
    # If scientific context and credible sources are detected, adjust the prediction
    if is_scientific or has_credible_source:
        # Still run the model for comparison but use processed text
        # Extract features
        tfidf_features = vectorizer.transform([processed_text])
        count_features = count_vectorizer.transform([text]) if count_vectorizer else None
        tweet_features = extract_tweet_features(text)
        
        # Combine features
        if count_features is not None:
            combined_features = hstack([tfidf_features, count_features, tweet_features])
        else:
            combined_features = hstack([tfidf_features, tweet_features])
        
        # Predict
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        
        # Get confidence
        confidence = float(max(probabilities))
        
        # If model predicts fake but we have scientific context/credible sources
        if prediction == 'fake':
            explanation = "This content contains scientific terminology or references credible sources, " + \
                         "which suggests it may be discussing legitimate research findings. " + \
                         "Consider verifying with the original research paper."
            
            if scientific_explanation:
                explanation += f" {scientific_explanation}."
            
            if source_explanation:
                explanation += f" {source_explanation}."
            
            return jsonify({
                'prediction': 'needs_verification',
                'confidence': 0.6,
                'original_prediction': prediction,
                'original_confidence': confidence,
                'explanation': explanation
            })
    
    # Continue with regular classification using processed text
    # Extract features
    tfidf_features = vectorizer.transform([processed_text])
    count_features = count_vectorizer.transform([text]) if count_vectorizer else None
    tweet_features = extract_tweet_features(text)
    
    # Combine features
    if count_features is not None:
        combined_features = hstack([tfidf_features, count_features, tweet_features])
    else:
        combined_features = hstack([tfidf_features, tweet_features])
    
    # Predict
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    
    # Get confidence
    confidence = float(max(probabilities))
    
    # Sort classes by probability
    sorted_indices = probabilities.argsort()[::-1]
    top_classes = [(model.classes_[idx], probabilities[idx]) for idx in sorted_indices[:2]]
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'top_classes': [{'class': cls, 'probability': prob} for cls, prob in top_classes]
    })

def detect_opinion(text):
    text_lower = text.lower()
    
    for phrase in opinion_phrases:
        if phrase in text_lower:
            return True
    
    if "?" in text and not text.endswith("?"):
        return True
    
    subjective_count = sum(1 for word in subjective_words if f" {word} " in f" {text_lower} ")
    if subjective_count >= 2:
        return True
    
    return False

@app.route('/langchain_verify', methods=['POST'])
def langchain_verify():
    if langchain is None:
        return jsonify({
            'error': 'LangChain not initialized',
            'prediction': 'unknown',
            'confidence': 0.5
        }), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    try:
        # Use LangChain to analyze the text
        llm_result = langchain.run(statement=text)
        
        # Parse the LLM response
        llm_classification = None
        if "FAKE" in llm_result:
            llm_classification = "fake"
        elif "REAL" in llm_result:
            llm_classification = "real"
        else:
            llm_classification = "unknown"
        
        # Extract confidence level
        llm_confidence = 0.7  # Default medium confidence
        if "HIGH" in llm_result:
            llm_confidence = 0.9
        elif "LOW" in llm_result:
            llm_confidence = 0.6
        
        # Extract explanation (everything after the classification and confidence)
        explanation = None
        if llm_result:
            # Try to extract just the explanation part
            explanation_parts = llm_result.split("3.")
            if len(explanation_parts) > 1:
                explanation = explanation_parts[1].strip()
            else:
                explanation = llm_result
        
        return jsonify({
            'prediction': llm_classification,
            'confidence': llm_confidence,
            'explanation': explanation
        })
    
    except Exception as e:
        print(f"Error using LangChain: {e}")
        return jsonify({
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.5
        }), 500

def verify_source_credibility(text):
    """
    Check if the tweet references credible scientific or medical sources.
    """
    credible_sources = [
        # Medical journals
        "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com", "nature.com", 
        "science.org", "cell.com", "pnas.org", "plos.org", "frontiersin.org",
        
        # Health organizations
        "who.int", "cdc.gov", "nih.gov", "fda.gov", "ema.europa.eu", "ecdc.europa.eu",
        
        # Medical institutions
        "mayoclinic.org", "hopkinsmedicine.org", "clevelandclinic.org", "health.harvard.edu",
        "medlineplus.gov", "pubmed.ncbi.nlm.nih.gov"
    ]
    
    # Extract URLs from text
    import re
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text)
    
    for url in urls:
        for source in credible_sources:
            if source in url:
                return True, f"References credible source: {source}"
    
    # Check for mentions of credible sources even without URLs
    text_lower = text.lower()
    for source in ["nejm", "lancet", "jama", "bmj", "nature", "science", "who", "cdc", "nih", "fda"]:
        if source in text_lower:
            return True, f"Mentions credible source: {source}"
    
    return False, None

def match_against_known_facts(text):
    """
    Match the tweet against a database of known COVID-19 facts and myths.
    """
    # Sample database of known COVID-19 facts and myths
    known_facts = {
        "vaccines cause autism": {
            "status": "myth",
            "explanation": "Scientific consensus has firmly established that vaccines do not cause autism."
        },
        "5g causes covid": {
            "status": "myth",
            "explanation": "There is no scientific evidence linking 5G technology to COVID-19."
        },
        "masks prevent spread": {
            "status": "fact",
            "explanation": "Scientific studies have shown that masks help reduce the transmission of respiratory viruses."
        },
        # Add more known facts and myths
    }
    
    text_lower = text.lower()
    for claim, info in known_facts.items():
        if claim in text_lower:
            return True, info
    
    return False, None

def analyze_sentiment(text):
    """
    Analyze the sentiment of the tweet to detect emotional manipulation.
    """
    # This would ideally use a sentiment analysis library like NLTK or TextBlob
    # For demonstration, using a simple approach
    
    fear_words = ["deadly", "danger", "threat", "scary", "terrifying", "alarming", 
                 "catastrophic", "fatal", "lethal", "kill", "death", "die"]
    
    anger_words = ["outrage", "scandal", "corrupt", "criminal", "evil", "wicked",
                  "sinister", "plot", "scheme", "conspiracy"]
    
    fear_count = sum(1 for word in fear_words if word in text.lower())
    anger_count = sum(1 for word in anger_words if word in text.lower())
    
    if fear_count >= 2 or anger_count >= 2:
        return True, "High emotional content detected"
    
    return False, None

def analyze_context(text):
    """
    Analyze the context of the tweet to detect out-of-context information.
    """
    # Check for date references that might indicate outdated information
    date_pattern = r'\b(202[0-3]|january|february|march|april|may|june|july|august|september|october|november|december)\b'
    dates = re.findall(date_pattern, text.lower())
    
    if dates:
        return True, "Contains date references that may indicate time-sensitive information"
    
    # Check for location-specific references
    location_pattern = r'\b(in [a-z]+|at [a-z]+)\b'
    locations = re.findall(location_pattern, text.lower())
    
    if locations:
        return True, "Contains location-specific information that may not apply globally"
    
    return False, None

def analyze_thread_context(text, thread_context):
    """
    Analyze the thread context to see if it supports or contradicts the claim.
    """
    # Split the thread context into individual tweets
    tweets = thread_context.split(" [TWEET_SEPARATOR] ")
    
    # Initialize analysis result
    result = {
        'supports_claim': False,
        'contradicts_claim': False,
        'explanation': None
    }
    
    # Check for contradictions or supporting evidence
    # This is a simplified version - you would want to use NLP techniques for better analysis
    contradiction_indicators = ["that's not true", "false", "incorrect", "wrong", "misleading"]
    support_indicators = ["exactly", "that's right", "correct", "true", "agree"]
    
    contradictions = 0
    supports = 0
    
    for tweet in tweets:
        tweet_lower = tweet.lower()
        
        # Skip the original tweet
        if tweet_lower == text.lower():
            continue
            
        # Check for contradictions
        for indicator in contradiction_indicators:
            if indicator in tweet_lower:
                contradictions += 1
                
        # Check for supporting evidence
        for indicator in support_indicators:
            if indicator in tweet_lower:
                supports += 1
    
    # Determine the overall context
    if contradictions > supports:
        result['contradicts_claim'] = True
        result['explanation'] = f"Thread contains {contradictions} contradicting responses"
    elif supports > contradictions:
        result['supports_claim'] = True
        result['explanation'] = f"Thread contains {supports} supporting responses"
    else:
        result['explanation'] = "Thread context is neutral or mixed"
    
    return result

# Initialize feedback storage
feedback_file = os.path.join(os.path.dirname(__file__), 'user_feedback.csv')
if not os.path.exists(feedback_file):
    # Create empty feedback file with headers
    pd.DataFrame(columns=[
        'text', 'prediction', 'is_correct', 'timestamp', 'processed'
    ]).to_csv(feedback_file, index=False)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    if not data or 'text' not in data or 'prediction' not in data or 'is_correct' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Load existing feedback
        feedback_df = pd.read_csv(feedback_file)
        
        # Add new feedback
        new_feedback = pd.DataFrame([{
            'text': data['text'],
            'prediction': data['prediction'],
            'is_correct': data['is_correct'],
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'processed': False
        }])
        
        # Append to existing feedback
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        
        # Save updated feedback
        feedback_df.to_csv(feedback_file, index=False)
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_unprocessed_feedback', methods=['GET'])
def get_unprocessed_feedback():
    try:
        # Load feedback
        feedback_df = pd.read_csv(feedback_file)
        
        # Filter unprocessed feedback
        unprocessed = feedback_df[feedback_df['processed'] == False]
        
        return jsonify({
            'count': len(unprocessed),
            'items': unprocessed.to_dict('records')
        })
    
    except Exception as e:
        print(f"Error retrieving feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/mark_feedback_processed', methods=['POST'])
def mark_feedback_processed():
    data = request.json
    if not data or 'indices' not in data:
        return jsonify({'error': 'No indices provided'}), 400
    
    try:
        # Load feedback
        feedback_df = pd.read_csv(feedback_file)
        
        # Mark specified indices as processed
        indices = data['indices']
        for idx in indices:
            if 0 <= idx < len(feedback_df):
                feedback_df.at[idx, 'processed'] = True
        
        # Save updated feedback
        feedback_df.to_csv(feedback_file, index=False)
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error updating feedback: {e}")
        return jsonify({'error': str(e)}), 500

# Add this function to your server.py
def detect_scientific_context(text):
    """
    Detect if the text contains references to scientific studies or publications.
    """
    scientific_indicators = [
        "study", "research", "paper", "journal", "published", "authors",
        "findings", "results", "data", "evidence", "clinical", "trial",
        "peer-reviewed", "doi", "publication", "researchers", "scientists",
        "investigation", "analysis", "experiment"
    ]
    
    scientific_journals = [
        "lancet", "nejm", "jama", "bmj", "nature", "science", "cell", 
        "pnas", "plos", "frontiers", "npj", "medrxiv", "biorxiv"
    ]
    
    text_lower = text.lower()
    
    # Check for scientific terminology
    scientific_terms_count = sum(1 for term in scientific_indicators if term in text_lower)
    
    # Check for journal references
    journal_references = any(journal in text_lower for journal in scientific_journals)
    
    # If multiple scientific terms or a journal reference is found, likely scientific context
    if scientific_terms_count >= 2 or journal_references:
        return True, "Contains scientific terminology or journal references"
    
    return False, None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)