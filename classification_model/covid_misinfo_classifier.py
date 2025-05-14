import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class FastMisinfoClassifier:
    
    def __init__(self, model_path=None):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.twitter_slang = self._load_twitter_slang()
        self.model = None
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_twitter_slang(self):
        return {
            'imo': 'in my opinion',
            'imho': 'in my humble opinion',
            'tbh': 'to be honest',
            'irl': 'in real life',
            'idk': 'i do not know',
            'rn': 'right now',
            'u': 'you',
            'ur': 'your',
            'b/c': 'because',
            'w/': 'with',
            'w/o': 'without',
            'b4': 'before',
            'ppl': 'people',
            'dm': 'direct message',
            'fwiw': 'for what it is worth',
            'lol': 'laughing',
            'smh': 'shaking my head',
            'tfw': 'that feeling when',
            'yolo': 'you only live once',
            'rt': 'retweet',
            'fb': 'facebook',
            'ig': 'instagram',
            'yt': 'youtube',
            'btw': 'by the way',
            'brb': 'be right back',
            'omg': 'oh my god',
            'wtf': 'what the fuck',
            'af': 'as fuck',
            'rip': 'rest in peace',
            'fomo': 'fear of missing out',
            'fyi': 'for your information',
            'tbt': 'throwback thursday',
            'tl;dr': 'too long did not read',
            'nvm': 'never mind',
            'iirc': 'if i recall correctly',
            'ftw': 'for the win',
            'ama': 'ask me anything',
            'eli5': 'explain like i am five',
            'nsfw': 'not safe for work',
            'tldr': 'too long did not read',
            'thx': 'thanks',
            'ty': 'thank you',
            'bc': 'because',
            'cuz': 'because',
            'cause': 'because',
            'gonna': 'going to',
            'wanna': 'want to',
            'gotta': 'got to',
            'ya': 'you',
            'yall': 'you all',
            'y\'all': 'you all',
            'srsly': 'seriously',
            'def': 'definitely',
            'prob': 'probably',
            'w/e': 'whatever',
            'tho': 'though',
            'thru': 'through',
            'k': 'okay',
            'kk': 'okay',
            'tmrw': 'tomorrow',
            'tmw': 'tomorrow',
            'yday': 'yesterday',
            'rly': 'really',
            'plz': 'please',
            'pls': 'please',
            'govt': 'government',
            'vax': 'vaccine',
            'vaxx': 'vaccine',
            'covid': 'covid-19',
            'corona': 'coronavirus',
            'rona': 'coronavirus',
            'fake news': 'misinformation',
            'msm': 'mainstream media',
            'hoax': 'false information',
            'sheeple': 'gullible people',
            'woke': 'aware',
            'conspiracy': 'conspiracy theory'
        }
    
    def preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ''
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text)
        text = re.sub(r'@\w+', ' [USER] ', text)
        text = re.sub(r'#(\w+)', r' \1 [HASHTAG] ', text)
        
        for slang, replacement in self.twitter_slang.items():
            text = re.sub(r'\b' + slang + r'\b', replacement, text)
        
        text = re.sub(r'[^\x00-\x7F]+', ' [EMOJI] ', text)
        text = re.sub(r'\b\d+%\b', ' [PERCENTAGE] ', text)
        text = re.sub(r'\b\d+\b', ' [NUMBER] ', text)
        text = re.sub(r'[^\w\s\?\!\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = nltk.word_tokenize(text)
        
        filtered_words = []
        for word in words:
            if word in ['[URL]', '[USER]', '[HASHTAG]', '[EMOJI]', '[PERCENTAGE]', '[NUMBER]']:
                filtered_words.append(word)
            elif word not in self.stop_words and len(word) > 2:
                filtered_words.append(self.lemmatizer.lemmatize(word))
        
        processed_text = ' '.join(filtered_words)
        
        return processed_text
    
    def extract_features(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return np.zeros(9)
        
        word_count = len(text.split())
        
        is_retweet = 1 if re.match(r'^RT @\w+:', text) else 0
        hashtag_count = len(re.findall(r'#\w+', text))
        mention_count = len(re.findall(r'@\w+', text))
        url_count = len(re.findall(r'http\S+|www\S+|https\S+', text))
        
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        misinfo_keywords = ['hoax', 'conspiracy', 'truth', 'fake', 'scam', 'lie', 'microchip', '5g', 'bill gates', 
                           'coverup', 'cover-up', 'exposed', 'secret', 'hidden', 'they dont want you to know',
                           'wake up', 'sheeple', 'plandemic', 'deep state', 'new world order', 'illuminati']
        misinfo_count = sum(1 for word in misinfo_keywords if word in text.lower())
        
        science_keywords = ['study', 'research', 'evidence', 'scientist', 'doctor', 'expert', 'data', 'clinical',
                            'peer-reviewed', 'journal', 'published', 'analysis', 'experiment', 'laboratory',
                            'controlled', 'trial', 'sample', 'statistical', 'significant']
        science_count = sum(1 for word in science_keywords if word in text.lower())
        
        return np.array([
            word_count, 
            hashtag_count, 
            mention_count, 
            url_count, 
            exclamation_count + question_count,
            uppercase_ratio,
            is_retweet,
            misinfo_count,
            science_count
        ])
    
    def _standardize_label(self, label):
        if pd.isna(label):
            return 'unknown'
            
        label = str(label).lower().strip()
        
        if label in ['real', 'true', 'reliable', 'credible', '1', 'support', 'favor']:
            return 'real'
        elif label in ['fake', 'false', 'unreliable', 'misinformation', '0', 'against', 'against_fake']:
            return 'fake'
        else:
            return 'unknown'
    
    def load_and_prepare_data(self, file_paths, text_col='text', label_col='label', test_size=0.2):
        print("Loading and preparing data...")
        
        column_mappings = {
            'data/covmis_stance.csv': {'text_col': 'mis', 'label_col': 'label'},
            'data/misinformation.csv': {'text_col': 'mis', 'label_col': 'mid'}
        }
        
        dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    custom_mapping = column_mappings.get(file_path, {})
                    file_text_col = custom_mapping.get('text_col', text_col)
                    file_label_col = custom_mapping.get('label_col', label_col)
                    
                    if file_text_col not in df.columns:
                        text_candidates = ['text', 'tweet', 'content', 'message', 'title', 'claim', 'mis']
                        for candidate in text_candidates:
                            if candidate in df.columns:
                                file_text_col = candidate
                                break
                        else:
                            print(f"Warning: No text column found in {file_path}")
                            continue
                    
                    if file_label_col not in df.columns:
                        label_candidates = ['label', 'class', 'category', 'is_fake', 'is_misinformation', 
                                           'stance', 'veracity', 'fake', 'real', 'true', 'false', 'mid']
                        for candidate in label_candidates:
                            if candidate in df.columns:
                                file_label_col = candidate
                                break
                        else:
                            print(f"Warning: No label column found in {file_path}")
                            continue
                    
                    df[file_label_col] = df[file_label_col].apply(self._standardize_label)
                    
                    df = df[[file_text_col, file_label_col]]
                    
                    df.columns = ['text', 'label']
                    
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        if not dfs:
            raise ValueError("No valid datasets found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        combined_df.drop_duplicates(subset=['text'], inplace=True)
        
        combined_df = combined_df[combined_df['text'].notna() & (combined_df['text'].str.strip() != '')]
        
        combined_df = combined_df[combined_df['label'] != 'unknown']
        
        print("Class distribution:")
        print(combined_df['label'].value_counts())
        
        X_train, X_test, y_train, y_test = train_test_split(
            combined_df['text'], 
            combined_df['label'], 
            test_size=test_size, 
            random_state=42,
            stratify=combined_df['label']
        )
        
        print("Extracting Twitter-specific features...")
        feature_array_train = np.array([self.extract_features(text) for text in X_train])
        feature_array_test = np.array([self.extract_features(text) for text in X_test])
        
        return X_train, X_test, y_train, y_test, feature_array_train, feature_array_test
    
    def train(self, file_paths, text_col='text', label_col='label', test_size=0.2):
        start_time = time.time()
        
        X_train, X_test, y_train, y_test, feature_array_train, feature_array_test = self.load_and_prepare_data(
            file_paths, text_col, label_col, test_size
        )
        
        print("Preprocessing text...")
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        
        print("Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            max_features=20000,
            sublinear_tf=True,
            use_idf=True,
            norm='l2',
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_processed)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test_processed)
        
        has_twitter_elements = any('@' in text or '#' in text for text in X_train)
        
        if has_twitter_elements:
            print("Creating count features for hashtags and mentions...")
            self.count_vectorizer = CountVectorizer(
                analyzer='word',
                token_pattern=r'(?:@|#)\w+',
                max_features=1000
            )
            X_train_count = self.count_vectorizer.fit_transform(X_train)
            X_test_count = self.count_vectorizer.transform(X_test)
            
            print("Combining features...")
            X_train_combined = hstack([X_train_tfidf, X_train_count, feature_array_train])
            X_test_combined = hstack([X_test_tfidf, X_test_count, feature_array_test])
        else:
            print("No Twitter-specific elements (hashtags/mentions) found, skipping those features...")
            self.count_vectorizer = None
            
            print("Combining features...")
            X_train_combined = hstack([X_train_tfidf, feature_array_train])
            X_test_combined = hstack([X_test_tfidf, feature_array_test])
        
        print("Training ensemble model...")
        
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='liblinear')
        svc = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', dual=False, max_iter=1000, loss='squared_hinge')
        )
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5,
            subsample=0.8,
            min_samples_split=10
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('svc', svc),
                ('rf', rf),
                ('gb', gb)
            ],
            voting='soft',
            weights=[1, 1, 2, 2]
        )
        
        self.model.fit(X_train_combined, y_train)
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print("Saving model...")
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer
        }
        joblib.dump(model_data, 'models/fast_misinfo_model.pkl')
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, 
                    yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report,
            'training_time': training_time
        }
    
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        if self.count_vectorizer is not None:
            count_features = self.count_vectorizer.transform(texts)
            feature_arrays = np.array([self.extract_features(text) for text in texts])
            
            combined_features = hstack([tfidf_features, count_features, feature_arrays])
        else:
            feature_arrays = np.array([self.extract_features(text) for text in texts])
            
            combined_features = hstack([tfidf_features, feature_arrays])
        
        predictions = self.model.predict(combined_features)
        probabilities = self.model.predict_proba(combined_features)
        
        results = []
        for i, text in enumerate(texts):
            prediction = predictions[i]
            confidence = max(probabilities[i])
            
            features = self.extract_features(text)
            explanation = self._generate_explanation(text, prediction, probabilities[i], features)
            
            results.append({
                'text': text,
                'prediction': prediction,
                'confidence': confidence,
                'explanation': explanation
            })
        
        return results
    
    def _generate_explanation(self, text, prediction, probabilities, features):
        explanations = []
        
        confidence = max(probabilities)
        
        if confidence > 0.9:
            explanations.append(f"High confidence prediction ({confidence:.2f})")
        elif confidence > 0.7:
            explanations.append(f"Moderate confidence prediction ({confidence:.2f})")
        else:
            explanations.append(f"Low confidence prediction ({confidence:.2f})")
        
        word_count, hashtag_count, mention_count, url_count, punctuation_count, uppercase_ratio, is_retweet, misinfo_count, science_count = features
        
        if misinfo_count > 2:
            explanations.append(f"Contains {misinfo_count} terms commonly associated with misinformation")
        
        if science_count > 2:
            explanations.append(f"Contains {science_count} scientific or research-related terms")
        
        if is_retweet:
            explanations.append("Content is a retweet, which may amplify existing information")
        
        if hashtag_count > 3:
            explanations.append(f"Contains {hashtag_count} hashtags, which may indicate hashtag hijacking")
        
        if url_count == 0 and prediction == 'fake':
            explanations.append("No sources (URLs) provided to support claims")
        
        if uppercase_ratio > 0.3:
            explanations.append("Contains excessive use of uppercase letters, which may indicate sensationalism")
        
        if punctuation_count > 5:
            explanations.append("Contains excessive punctuation, which may indicate emotional language")
        
        if re.search(r'wake up|open your eyes|do your research', text.lower()):
            explanations.append("Contains phrases commonly used in misinformation to create urgency")
        
        if re.search(r'they don\'t want you to know|what they aren\'t telling you|the truth about', text.lower()):
            explanations.append("Contains conspiracy-framing language suggesting hidden knowledge")
        
        if len(explanations) <= 1:
            if prediction == 'fake':
                explanations.append("Content shows patterns consistent with misinformation")
            else:
                explanations.append("Content shows patterns consistent with reliable information")
        
        return explanations
    
    def _load_model(self, model_path):
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.count_vectorizer = model_data.get('count_vectorizer')
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def plot_feature_importance(model, feature_names, top_n=20, output_dir="results"):
    """
    Plot top N feature importances for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): List of all feature names
        top_n (int): Number of top features to display
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="crest")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()


def plot_timing_bar(training_time, inference_time, output_dir="results"):
    """
    Plot training vs. inference time bar chart.
    
    Args:
        training_time (float): Time taken to train the model
        inference_time (float): Time taken to run inference on a batch of examples
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = ['Training Time', 'Inference Time']
    values = [training_time, inference_time]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=values, palette="Set2")
    plt.ylabel("Time (seconds)")
    plt.title("Training vs. Inference Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_bar_chart.png"))
    plt.close()

def main():
    classifier = FastMisinfoClassifier()
    
    datasets = [
        'data/covmis_stance.csv',
        'data/misinformation.csv'
    ]
    
    print("Analyzing dataset structure...")
    for file_path in datasets:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\nColumns in {file_path}:")
                for col in df.columns:
                    print(f"  - {col}")
                
                print(f"\nSample data from {file_path}:")
                print(df.head(2))
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
    
    
    results = classifier.train(datasets)

    
    test_texts = [
        "COVID-19 vaccines contain microchips to track people. Wake up sheeple!",
        "According to CDC data, wearing masks helps reduce the spread of COVID-19 in indoor settings.",
        "BREAKING: Scientists discover that 5G towers are spreading the coronavirus! Share this before they delete it!",
        "A new peer-reviewed study in Nature shows promising results for the treatment of severe COVID-19 cases."
    ]
    
    predictions = classifier.predict(test_texts)
    
    print("\nTest Predictions:")
    for pred in predictions:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.2f})")
        print(f"Explanation: {', '.join(pred['explanation'])}")
        print()

    
    print("Generating evaluation visualizations...")

    
    _, _, y_train, y_test, _, _ = classifier.load_and_prepare_data(datasets)

    
    

    
    

    #Feature Importance
    tfidf_names = classifier.tfidf_vectorizer.get_feature_names_out().tolist()
    count_names = classifier.count_vectorizer.get_feature_names_out().tolist() if classifier.count_vectorizer else []
    manual_feature_names = [
        "word_count", "hashtag_count", "mention_count", "url_count",
        "punctuation_count", "uppercase_ratio", "is_retweet",
        "misinfo_keyword_count", "science_keyword_count"
    ]
    your_feature_names = tfidf_names + count_names + manual_feature_names
    plot_feature_importance(classifier.model.estimators_[2], your_feature_names)

    #Inference Timing
    import time
    start_time = time.time()
    classifier.predict(test_texts)
    end_time = time.time()
    total_inference_time = end_time - start_time
    plot_timing_bar(results['training_time'], total_inference_time)

    print("All visualizations saved to the 'results/' directory.")
    

if __name__ == "__main__":
    main()