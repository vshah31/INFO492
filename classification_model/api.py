from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from covid_misinfo_classifier import FastMisinfoClassifier
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

app = Flask(__name__)
CORS(app)  

MODEL_PATH = 'models/misinfo_classifier_model.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'
COUNT_PATH = 'models/count_vectorizer.pkl'

classifier = None


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('subjectivity')
nltk.download('opinion_lexicon')


sentimentanalyzer = SentimentIntensityAnalyzer()

def load_model():
    global classifier
    
    try:
        print("Loading model and vectorizers...")
        classifier = FastMisinfoClassifier()
        
        classifier.model = joblib.load(MODEL_PATH)
        classifier.tfidf_vectorizer = joblib.load(TFIDF_PATH)
        classifier.count_vectorizer = joblib.load(COUNT_PATH)
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'ok',
        'message': 'Misinformation Detection API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model not loaded',
                'prediction': 'unknown',
                'confidence': 0.0
            }), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided',
            'prediction': 'unknown',
            'confidence': 0.0
        }), 400
    
    text = data['text']
    
    try:
        prediction = classifier.predict([text])[0]
        
        return jsonify({
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'explanation': prediction['explanation']
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.0
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if classifier is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model not loaded',
                'predictions': []
            }), 500
    
    data = request.json
    if not data or 'texts' not in data:
        return jsonify({
            'error': 'No texts provided',
            'predictions': []
        }), 400
    
    texts = data['texts']
    
    try:
        predictions = classifier.predict(texts)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'explanation': pred['explanation']
            })
        
        return jsonify({
            'predictions': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'predictions': []
        }), 500

@app.route('/analyze_opinion', methods=['POST']) 
def analyze_opinion(): 
    data = request.json 
    if not data or 'text' not in data: 
        return jsonify({'error': 'No text provided'}), 400 
    
    text = data['text'] 
    
     
    sentiment_scores = sentimentanalyzer.polarity_scores(text) 
    
    
    
    is_opinion = abs(sentiment_scores['compound']) > 0.2 
    
   
    if sentiment_scores['compound'] > 0.05: 
        sentiment = 'positive' 
    elif sentiment_scores['compound'] < -0.05: 
        sentiment = 'negative' 
    else: 
        sentiment = 'neutral' 
    
    
    opinion_markers = ['I think', 'I believe', 'in my opinion', 'I feel', 
                      'should', 'must', 'ought to', 'seems like', 'appears to be'] 
    
    contains_markers = any(marker in text.lower() for marker in opinion_markers) 
    
    
    opinion_indicators = [] 
    
    
    try: 
        from nltk.corpus import opinion_lexicon 
        positive_words = set(opinion_lexicon.positive()) 
        negative_words = set(opinion_lexicon.negative()) 
        
        
        tokens = word_tokenize(text.lower()) 
        
        
        opinion_words = [word for word in tokens if word in positive_words or word in negative_words] 
        if opinion_words: 
            opinion_indicators.append("Contains subjective words: " + ", ".join(opinion_words[:5])) 
            is_opinion = True 
    except: 
        pass 
    
    
    try: 
        
        tagged_tokens = pos_tag(word_tokenize(text)) 
        
       
        personal_pronouns = [word for word, tag in tagged_tokens if tag in ('PRP', 'PRP$') 
                            and word.lower() in ('i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours')] 
        
        
        modal_verbs = [word for word, tag in tagged_tokens if tag == 'MD' 
                      and word.lower() in ('should', 'could', 'would', 'might', 'must')] 
        
        if personal_pronouns and modal_verbs: 
            opinion_indicators.append("Contains personal pronouns with modal verbs") 
            is_opinion = True 
        elif len(personal_pronouns) > 1: 
            opinion_indicators.append("Contains multiple personal pronouns") 
            is_opinion = is_opinion or len(personal_pronouns) > 2 
    except: 
        pass 
    
    
    try: 
        adjectives = [word for word, tag in tagged_tokens if tag.startswith('JJ')] 
        adverbs = [word for word, tag in tagged_tokens if tag.startswith('RB')] 
        
        if len(adjectives) > 2: 
            opinion_indicators.append("Contains multiple adjectives") 
            is_opinion = is_opinion or len(adjectives) > 3 
        
        if len(adverbs) > 2 and any(adv in ['really', 'very', 'extremely', 'absolutely'] for adv in adverbs): 
            opinion_indicators.append("Contains intensifying adverbs") 
            is_opinion = True 
    except: 
        pass 
    
    
    is_opinion = is_opinion or contains_markers 
    
    
    if data.get('check_misinfo', False): 
        
        misinfo_result = classifier.predict([text])[0]
        combined_result = { 
            'is_opinion': is_opinion, 
            'sentiment': sentiment, 
            'sentiment_scores': sentiment_scores, 
            'contains_opinion_markers': contains_markers, 
            'opinion_indicators': opinion_indicators, 
            'misinfo_prediction': misinfo_result['prediction'], 
            'misinfo_confidence': misinfo_result['confidence'], 
            'misinfo_explanation': misinfo_result['explanation'] 
        } 
        return jsonify(combined_result) 
    
    return jsonify({ 
        'is_opinion': is_opinion, 
        'sentiment': sentiment, 
        'sentiment_scores': sentiment_scores, 
        'contains_opinion_markers': contains_markers, 
        'opinion_indicators': opinion_indicators 
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)