from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from covid_misinfo_classifier import FastMisinfoClassifier
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from train_comprehensive_model import ChatGroqClassifier, EnsembleClassifier


nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)  
nltk.download('omw-1.4', quiet=True)  

app = Flask(__name__)
CORS(app)  

MODEL_PATH = 'models/comprehensive_misinfo_model.pkl'
TFIDF_PATH = 'models/comprehensive_tfidf_vectorizer.pkl'
COUNT_PATH = 'models/comprehensive_count_vectorizer.pkl'


classifier = None
fast_classifier = None
chatgroq_classifier = None
ensemble_classifier = None

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


sentimentanalyzer = SentimentIntensityAnalyzer()

def load_model():
    global classifier, fast_classifier, chatgroq_classifier, ensemble_classifier
    
    try:
        print("Loading models and vectorizers...")
        
        
        fast_classifier = FastMisinfoClassifier()
        fast_classifier.model = joblib.load(MODEL_PATH)
        fast_classifier.tfidf_vectorizer = joblib.load(TFIDF_PATH)
        fast_classifier.count_vectorizer = joblib.load(COUNT_PATH)
        
        
        chatgroq_api_key = "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" or os.environ.get("CHATGROQ_API_KEY")
        
        if chatgroq_api_key:
            chatgroq_classifier = ChatGroqClassifier(api_key=chatgroq_api_key)
            
            
            ensemble_classifier = EnsembleClassifier(
                [fast_classifier, chatgroq_classifier], 
                weights=[0.35, 0.65]
            )
            classifier = ensemble_classifier
            print("Ensemble classifier created with balanced weights between models")
        else:
            
            classifier = fast_classifier
            print("ChatGroq API key not found, using only FastMisinfoClassifier")
        
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
    if not data:
        return jsonify({
            'error': 'No data provided',
            'prediction': 'unknown',
            'confidence': 0.0
        }), 400
    
    
    if 'text' in data:
        text = data['text']
        include_sentiment = data.get('include_sentiment', False)
        
        try:
            # Check if we're using the ensemble with ChatGroq
            using_chatgroq = isinstance(classifier, EnsembleClassifier) and len(classifier.classifiers) > 1
            
            # Get prediction
            prediction = classifier.predict([text])[0]
            
            # Add information about which classifier made the decision
            result = {
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'explanation': prediction['explanation'],
                'classifier_used': 'ChatGroq' if using_chatgroq else 'FastMisinfoClassifier'
            }
            
            
            if include_sentiment:
                sentiment_scores = sentimentanalyzer.polarity_scores(text)
                
                if sentiment_scores['compound'] > 0.05:
                    sentiment = 'positive'
                elif sentiment_scores['compound'] < -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                result['sentiment'] = sentiment
                result['sentiment_scores'] = sentiment_scores
            
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0
            }), 500
    
    
    elif 'texts' in data:
        texts = data['texts']
        
        try:
            
            using_chatgroq = isinstance(classifier, EnsembleClassifier) and len(classifier.classifiers) > 1
            
            predictions = classifier.predict(texts)
            
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence'],
                    'explanation': pred['explanation'],
                    'classifier_used': 'ChatGroq' if using_chatgroq else 'FastMisinfoClassifier'
                })
            
            return jsonify({
                'predictions': results
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'predictions': []
            }), 500
    
    else:
        return jsonify({
            'error': 'Invalid request format. Please provide either "text" or "texts" field.',
            'prediction': 'unknown',
            'confidence': 0.0
        }), 400

if __name__ == '__main__':
    
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)