import os
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import requests
import json
import re  
from typing import List, Dict, Any, Tuple, Optional


from covid_misinfo_classifier import FastMisinfoClassifier

class ChatGroqClassifier:
    """Classifier that uses ChatGroq API for misinformation detection"""
    
    def __init__(self, api_key=None):
        self.api_key = "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" or os.environ.get("CHATGROQ_API_KEY")
        if not self.api_key:
            raise ValueError("ChatGroq API key is required. Set it as CHATGROQ_API_KEY environment variable or pass it to the constructor.")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"  
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Classify a single text using ChatGroq
        Returns a dictionary with probabilities for each class
        """
        prompt = f"""
        Analyze the following text and determine if it contains COVID-19 misinformation.
        Respond ONLY with a valid JSON object containing:
        1. "misinformation": probability between 0 and 1 that the text contains misinformation
        2. "factual": probability between 0 and 1 that the text is factual
        3. "explanation": a list of 2-3 specific reasons for your classification, based on:
           - Presence/absence of verifiable claims
           - Consistency with scientific consensus
           - Presence of misleading statements or logical fallacies
           - Citation of credible sources or lack thereof
        
        Do not include any text before or after the JSON object.
        
        Text to analyze: \"{text}\"
        """
        
        try:
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 300
                },
                timeout=10  
            )
            response.raise_for_status()
            
            
            if not response.text or response.text.isspace():
                print("Empty response from ChatGroq API")
                return {"misinformation": 0.5, "factual": 0.5, "explanation": ["API returned empty response"]}
            
            
            result = response.json()
            
            
            if "choices" not in result or not result["choices"] or "message" not in result["choices"][0]:
                print(f"Invalid response structure: {result}")
                return {"misinformation": 0.5, "factual": 0.5, "explanation": ["Invalid API response structure"]}
                
            content = result["choices"][0]["message"]["content"]
            
            
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as json_err:
                print(f"Failed to parse JSON from API response: {content[:100]}...")
                
                json_match = re.search(r'```\s*({[\s\S]*?})\s*```', content)
                if json_match:
                    try:
                        
                        response_data = json.loads(json_match.group(1))
                        
                        probabilities = {
                            "misinformation": response_data.get("misinformation", 0.5),
                            "factual": response_data.get("factual", 0.5)
                        }
                        
                        # Extract explanation if available
                        if "explanation" in response_data and isinstance(response_data["explanation"], list):
                            probabilities["explanation"] = response_data["explanation"]
                        else:
                            probabilities["explanation"] = ["Based on AI language model analysis"]
                            
                        return probabilities
                    except json.JSONDecodeError:
                        
                        pass
                # Fall back to extracting probabilities from text
                return self._extract_probabilities_from_text(content)
            
            
            probabilities = {
                "misinformation": response_data.get("misinformation", 0.5),
                "factual": response_data.get("factual", 0.5)
            }
            
            
            if "explanation" in response_data and isinstance(response_data["explanation"], list):
                probabilities["explanation"] = response_data["explanation"]
            else:
                probabilities["explanation"] = ["Based on AI language model analysis"]
                
            return probabilities
            
        except requests.exceptions.Timeout:
            print("Request to ChatGroq API timed out")
            return {"misinformation": 0.5, "factual": 0.5, "explanation": ["API request timed out"]}
        except requests.exceptions.RequestException as req_err:
            print(f"Request error with ChatGroq API: {str(req_err)}")
            return {"misinformation": 0.5, "factual": 0.5, "explanation": ["API connection error"]}
        except Exception as e:
            print(f"Error classifying text with ChatGroq: {str(e)}")
            # Return default probabilities if there's an error
            return {"misinformation": 0.5, "factual": 0.5, "explanation": ["Error in classification process"]}
    
    
    def _extract_probabilities_from_text(self, text: str) -> Dict[str, Any]:
        """Extract probability values from text when JSON parsing fails"""
        
        result = {"misinformation": 0.5, "factual": 0.5, "explanation": ["Extracted from text response"]}
        
        
        misinfo_match = re.search(r'misinformation[":\s]+(0\.\d+|1\.0)', text)
        factual_match = re.search(r'factual[":\s]+(0\.\d+|1\.0)', text)
        
        if misinfo_match:
            try:
                result["misinformation"] = float(misinfo_match.group(1))
            except (ValueError, IndexError):
                pass
                
        if factual_match:
            try:
                result["factual"] = float(factual_match.group(1))
            except (ValueError, IndexError):
                pass
        
        
        explanation_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('{') and not line.endswith('}') and ':' not in line[:15]:
                explanation_lines.append(line)
        
        if explanation_lines:
            result["explanation"] = explanation_lines[:3]  # Limit to 3 lines
        
        return result
    
    def batch_classify(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, float]]:
        """Classify a batch of texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.classify_text(text) for text in batch]
            results.extend(batch_results)
            
            
            if i + batch_size < len(texts):
                time.sleep(1)
        
        return results
    
    def train(self, dataset_files: List[str]) -> Dict[str, Any]:
        """
        'Train' the ChatGroq classifier
        
        Note: LLMs like ChatGroq don't need traditional training, but we'll
        use this method to evaluate its performance on the datasets
        """
        results = {
            "accuracy": [],
            "classification_reports": [],
            "confusion_matrices": []
        }
        
        for file_path in dataset_files:
            try:
                
                df = pd.read_csv(file_path)
                
                
                text_col = None
                label_col = None
                
                
                for col in ['text', 'content', 'tweet', 'mis', 'claim']:
                    if col in df.columns:
                        text_col = col
                        break
                
                
                for col in ['label', 'class', 'fake', 'real', 'misinformation']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if text_col is None or label_col is None:
                    print(f"Skipping {file_path}: Could not identify text or label columns")
                    continue
                
                
                sample_size = min(100, len(df))
                df_sample = df.sample(sample_size, random_state=42)
                
                
                texts = df_sample[text_col].tolist()
                true_labels = df_sample[label_col].tolist()
                
                
                binary_labels = []
                for label in true_labels:
                    if isinstance(label, str):
                        binary_labels.append(1 if label.lower() in ['fake', 'false', 'misinformation', '1', 'yes'] else 0)
                    else:
                        binary_labels.append(int(bool(label)))
                
                
                predictions = self.batch_classify(texts)
                
                
                binary_predictions = [1 if pred.get('misinformation', 0) > pred.get('factual', 0) else 0 
                                     for pred in predictions]
                
                
                accuracy = accuracy_score(binary_labels, binary_predictions)
                report = classification_report(binary_labels, binary_predictions, target_names=['factual', 'misinformation'])
                conf_matrix = confusion_matrix(binary_labels, binary_predictions)
                
                
                results["accuracy"].append(accuracy)
                results["classification_reports"].append(report)
                results["confusion_matrices"].append(conf_matrix)
                
                print(f"ChatGroq evaluation on {file_path}: Accuracy = {accuracy:.4f}")
                print(report)
                
            except Exception as e:
                print(f"Error evaluating ChatGroq on {file_path}: {str(e)}")
        
        
        if results["accuracy"]:
            results["avg_accuracy"] = sum(results["accuracy"]) / len(results["accuracy"])
            print(f"ChatGroq average accuracy: {results['avg_accuracy']:.4f}")
        
        return results

class EnsembleClassifier:
    """Ensemble classifier that combines predictions from multiple classifiers"""
    
    def __init__(self, classifiers, weights=None):
        """
        Initialize the ensemble classifier
        
        Args:
            classifiers: List of classifier objects
            weights: Optional list of weights for each classifier (must sum to 1)
        """
        self.classifiers = classifiers
        
        if weights is None:
            
            self.weights = [1/len(classifiers)] * len(classifiers)
        else:
            if len(weights) != len(classifiers):
                raise ValueError("Number of weights must match number of classifiers")
            if abs(sum(weights) - 1.0) > 1e-10:
                raise ValueError("Weights must sum to 1")
            self.weights = weights
    
    def train(self, dataset_files):
        """Train all classifiers in the ensemble"""
        results = {}
        
        for i, classifier in enumerate(self.classifiers):
            print(f"\nTraining classifier {i+1}/{len(self.classifiers)}...")
            classifier_results = classifier.train(dataset_files)
            results[f"classifier_{i+1}"] = classifier_results
        
        return results
    
    def save_models(self, base_dir='models'):
        """Save all models and their components"""
        os.makedirs(base_dir, exist_ok=True)
        
        
        for i, classifier in enumerate(self.classifiers):
            if hasattr(classifier, 'model') and hasattr(classifier.__class__, '__module__'):
                try:
                    
                    if hasattr(classifier.model, 'predict'):
                        model_path = f"{base_dir}/classifier_{i+1}_model.pkl"
                        joblib.dump(classifier.model, model_path)
                        print(f"Saved model {i+1} to {model_path}")
                    
                    
                    if hasattr(classifier, 'tfidf_vectorizer'):
                        tfidf_path = f"{base_dir}/classifier_{i+1}_tfidf_vectorizer.pkl"
                        joblib.dump(classifier.tfidf_vectorizer, tfidf_path)
                        print(f"Saved TF-IDF vectorizer {i+1} to {tfidf_path}")
                    
                    if hasattr(classifier, 'count_vectorizer'):
                        count_path = f"{base_dir}/classifier_{i+1}_count_vectorizer.pkl"
                        joblib.dump(classifier.count_vectorizer, count_path)
                        print(f"Saved Count vectorizer {i+1} to {count_path}")
                        
                except Exception as e:
                    print(f"Error saving classifier {i+1}: {str(e)}")
        
        
        weights_path = f"{base_dir}/ensemble_weights.pkl"
        joblib.dump(self.weights, weights_path)
        print(f"Saved ensemble weights to {weights_path}")

    def predict(self, texts):
        """Predict class for a list of texts"""
        all_predictions = []
        
        for text in texts:
            
            classifier_predictions = []
            
            for i, classifier in enumerate(self.classifiers):
                try:
                    if hasattr(classifier, 'classify_text') and i == 1:  
                        
                        probs = classifier.classify_text(text)
                        prediction = "fake" if probs.get("misinformation", 0) > probs.get("factual", 0) else "real"
                        confidence = max(probs.get("misinformation", 0), probs.get("factual", 0))
                        explanation = probs.get("explanation", ["Based on AI language model analysis"])
                        
                        classifier_predictions.append({
                            "prediction": prediction,
                            "confidence": confidence,
                            "explanation": explanation,
                            "weight": self.weights[i]
                        })
                    elif hasattr(classifier, 'predict'):  
                        pred = classifier.predict([text])[0]
                        pred["weight"] = self.weights[i]
                        classifier_predictions.append(pred)
                except Exception as e:
                    print(f"Error with classifier {i}: {str(e)}")
                    
                    classifier_predictions.append({
                        "prediction": "unknown",
                        "confidence": 0.5,
                        "explanation": [f"Classifier error: {str(e)}"],
                        "weight": self.weights[i]
                    })
            
            
            chatgroq_pred = None
            fast_pred = None
            
            for pred in classifier_predictions:
                if pred["weight"] > 0.5:  
                    chatgroq_pred = pred
                else:
                    fast_pred = pred
            
            
            
            if chatgroq_pred and chatgroq_pred["confidence"] > 0.8:
                final_pred = {
                    "prediction": chatgroq_pred["prediction"],
                    "confidence": chatgroq_pred["confidence"],
                    "explanation": chatgroq_pred["explanation"],
                    "classifier_used": "ChatGroq"
                }
            
            elif fast_pred and fast_pred["confidence"] > 0.75:
                final_pred = {
                    "prediction": fast_pred["prediction"],
                    "confidence": fast_pred["confidence"],
                    "explanation": fast_pred["explanation"],
                    "classifier_used": "FastMisinfoClassifier"
                }
            
            else:
                
                weighted_fake = sum(pred["confidence"] * pred["weight"] for pred in classifier_predictions 
                                if pred["prediction"] == "fake")
                weighted_real = sum(pred["confidence"] * pred["weight"] for pred in classifier_predictions 
                                if pred["prediction"] == "real")
                
                
                if weighted_fake > weighted_real:
                    prediction = "fake"
                    confidence = weighted_fake / (weighted_fake + weighted_real) if (weighted_fake + weighted_real) > 0 else 0.5
                else:
                    prediction = "real"
                    confidence = weighted_real / (weighted_fake + weighted_real) if (weighted_fake + weighted_real) > 0 else 0.5
                
                
                all_explanations = []
                for pred in classifier_predictions:
                    if isinstance(pred["explanation"], list):
                        all_explanations.extend(pred["explanation"])
                    else:
                        all_explanations.append(str(pred["explanation"]))
                
                final_pred = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "explanation": all_explanations[:3],  
                    "classifier_used": "Ensemble"
                }
            
            all_predictions.append(final_pred)
        
        return all_predictions

def find_all_datasets(base_dir='data'):
    """Find all available datasets in the data directory"""
    dataset_files = []
    
    
    for file_path in glob.glob(f"{base_dir}/**/*.csv", recursive=True):
        
        if 'metadata' in file_path.lower() or 'summary' in file_path.lower():
            continue
        dataset_files.append(file_path)
    
    print(f"Found {len(dataset_files)} dataset files")
    return dataset_files

def analyze_datasets(dataset_files):
    """Analyze the structure of each dataset"""
    dataset_info = {}
    
    for file_path in dataset_files:
        try:
            df = pd.read_csv(file_path)
            
            
            text_col = None
            label_col = None
            
            
            for col in ['text', 'content', 'tweet', 'mis', 'claim']:
                if col in df.columns:
                    text_col = col
                    break
            
            
            for col in ['label', 'class', 'fake', 'real', 'misinformation']:
                if col in df.columns:
                    label_col = col
                    break
            
            
            dataset_info[file_path] = {
                'rows': len(df),
                'columns': list(df.columns),
                'text_col': text_col,
                'label_col': label_col
            }
            
            print(f"Analyzed {file_path}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    return dataset_info

def train_comprehensive_model():
    """Train a comprehensive model using all available datasets"""
    start_time = time.time()
    
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    
    dataset_files = find_all_datasets()
    
    
    dataset_info = analyze_datasets(dataset_files)
    
    
    valid_datasets = [
        file_path for file_path, info in dataset_info.items() 
        if info['text_col'] is not None and info['label_col'] is not None
    ]
    
    print(f"\nFound {len(valid_datasets)} valid datasets for training")
    
    
    fast_classifier = FastMisinfoClassifier()
    
    
    chatgroq_api_key = os.environ.get("CHATGROQ_API_KEY")
    if not chatgroq_api_key:
        print("\nWARNING: ChatGroq API key not found in environment variables.")
        print("Please set the CHATGROQ_API_KEY environment variable to use ChatGroq classifier.")
        print("Continuing with only the FastMisinfoClassifier...")
        ensemble = EnsembleClassifier([fast_classifier], weights=[1.0])
    else:
        chatgroq_classifier = ChatGroqClassifier(api_key=chatgroq_api_key)
        
        ensemble = EnsembleClassifier(
            [fast_classifier, chatgroq_classifier], 
            weights=[0.4, 0.6]
        )
    
    
    print("\nTraining comprehensive ensemble model...")
    results = ensemble.train(valid_datasets)
    
    
    print("\nSaving models and vectorizers...")
    ensemble.save_models()
    
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    train_comprehensive_model()