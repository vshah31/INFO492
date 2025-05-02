import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.calibration import CalibratedClassifierCV

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. Enhanced Text Preprocessing Function
def preprocess_text(text):
    if pd.isna(text):
        return ''
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text)
    text = re.sub(r'@\w+', ' mention ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# 2. Load and Preprocess Dataset
print("Loading datasets...")
train_df = pd.read_csv('data/nanyy1025_covid_fake_news_train.csv')
validation_df = pd.read_csv('data/nanyy1025_covid_fake_news_validation.csv')
test_df = pd.read_csv('data/nanyy1025_covid_fake_news_test.csv')

text_column = 'tweet'
combined_train_df = pd.concat([train_df, validation_df])

print("Preprocessing text...")
combined_train_df[text_column] = combined_train_df[text_column].apply(preprocess_text)
test_df[text_column] = test_df[text_column].apply(preprocess_text)

X_train = combined_train_df[text_column]
y_train = combined_train_df['label']
X_test = test_df[text_column]
y_test = test_df['label']

# 3. Vectorization
print("Converting text data to enhanced TF-IDF features...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=15000,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Model Training - Ensemble Approach
print("Training an ensemble of classifiers...")
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

nb_model = MultinomialNB(alpha=0.5)
lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
base_svc = LinearSVC(C=1.0, class_weight='balanced', dual=False, max_iter=1000)
svc_model = CalibratedClassifierCV(base_svc)

ensemble = VotingClassifier(
    estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svc', svc_model)
    ],
    voting='soft'
)

ensemble.fit(X_train_tfidf, y_train)
model = ensemble

# 5. Model Evaluation
print("Making predictions and evaluating...")
y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 6. AI Response Testing Function
def test_ai_response(response_text, model, vectorizer):
    response_text_processed = preprocess_text(response_text)
    response_tfidf = vectorizer.transform([response_text_processed])
    
    prediction = model.predict(response_tfidf)[0]
    proba = model.predict_proba(response_tfidf)[0]
    
    sorted_indices = proba.argsort()[::-1]
    top2_classes = [(model.classes_[idx], proba[idx]) for idx in sorted_indices[:2]]
    
    return {
        "Prediction": prediction,
        "Top 2 Classes": top2_classes
    }

# 7. LangChain Integration with Groq
def setup_langchain():
    os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
    
    template = """
    Analyze the following statement about COVID-19 and determine if it contains misinformation.
    Consider scientific consensus, credible sources, and logical reasoning.
    
    Statement: {statement}
    
    Is this statement likely to contain misinformation? Respond with:
    1. Classification (REAL or FAKE)
    2. Confidence (LOW, MEDIUM, HIGH)
    3. Brief explanation (1-2 sentences)
    """
    
    prompt = PromptTemplate(
        input_variables=["statement"],
        template=template,
    )
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain

def hybrid_classification(text, base_model, vectorizer, llm_chain=None, threshold=0.75, uncertainty_threshold=0.10, base_confidence_threshold=0.65):
    base_result = test_ai_response(text, base_model, vectorizer)
    prediction = base_result["Prediction"]
    top_class_prob = base_result["Top 2 Classes"][0][1]
    
    if len(base_result["Top 2 Classes"]) > 1:
        prob_difference = base_result["Top 2 Classes"][0][1] - base_result["Top 2 Classes"][1][1]
    else:
        prob_difference = 1.0
    
    if top_class_prob >= threshold and prob_difference >= uncertainty_threshold:
        return {
            "prediction": prediction,
            "confidence": "HIGH",
            "method": "base_model",
            "base_model_result": base_result,
            "llm_result": None
        }
    
    if llm_chain and (top_class_prob < base_confidence_threshold or prob_difference < uncertainty_threshold):
        try:
            llm_result = llm_chain.run(statement=text)
            
            llm_classification = None
            if "FAKE" in llm_result:
                llm_classification = "fake"
            elif "REAL" in llm_result:
                llm_classification = "real"
            
            llm_confidence = "MEDIUM"
            if "HIGH" in llm_result:
                llm_confidence = "HIGH"
            elif "LOW" in llm_result:
                llm_confidence = "LOW"
            
            if llm_classification and top_class_prob < base_confidence_threshold:
                return {
                    "prediction": llm_classification,
                    "confidence": llm_confidence,
                    "method": "llm_override",
                    "base_model_result": base_result,
                    "llm_result": llm_result,
                    "override_reason": f"Base model confidence too low: {top_class_prob:.2f} < {base_confidence_threshold}"
                }
            elif llm_classification and prob_difference < uncertainty_threshold:
                return {
                    "prediction": llm_classification,
                    "confidence": llm_confidence,
                    "method": "llm_override",
                    "base_model_result": base_result,
                    "llm_result": llm_result,
                    "override_reason": f"Base model uncertainty: {prob_difference:.2f} < {uncertainty_threshold}"
                }
            else:
                return {
                    "prediction": prediction,
                    "confidence": "MEDIUM",
                    "method": "hybrid",
                    "base_model_result": base_result,
                    "llm_result": llm_result
                }
        except Exception as e:
            print(f"LLM error: {e}")
    
    return {
        "prediction": prediction,
        "confidence": "LOW" if top_class_prob < threshold else "MEDIUM",
        "method": "base_model",
        "base_model_result": base_result,
        "llm_result": None
    }

try:
    print("\nSetting up LangChain with Groq...")
    llm_chain = setup_langchain()
    print("LangChain setup successful!")
except Exception as e:
    print(f"Error setting up LangChain: {e}")
    llm_chain = None

# 8. Test Example AI Responses
print("\nTesting Integrated AI Responses:")
response_texts = [
    "COVID-19 vaccines are a government conspiracy.",
    "Wearing masks helps prevent the spread of COVID-19.",
    "5G networks spread COVID-19.",
    "Social distancing reduces the risk of infection.",
    "COVID-19 is no more dangerous than the seasonal flu.",
    "The COVID-19 pandemic is a hoax.",
    "The COVID-19 vaccine can alter your DNA.",
    "COVID-19 vaccines were developed using standard clinical trial protocols.",
    "COVID-19 primarily spreads through respiratory droplets.",
    "Hydroxychloroquine is a proven treatment for COVID-19.",
    "COVID-19 vaccines include microchips for tracking people."
]

for response in response_texts:
    result = hybrid_classification(response, model, vectorizer, llm_chain)
    print(f"\nStatement: '{response}'")
    print(f"Final Classification: {result['prediction'].upper()} (Confidence: {result['confidence']})")
    print(f"Method: {result['method']}")
    
    base_pred = result['base_model_result']['Prediction']
    base_conf = result['base_model_result']['Top 2 Classes'][0][1]
    print(f"Base Model: {base_pred.upper()} (Confidence: {base_conf:.2f})")
    
    if result['llm_result']:
        print(f"LLM Analysis: {result['llm_result']}")
    
    if 'override_reason' in result:
        print(f"Note: {result['override_reason']}")
    elif 'note' in result:
        print(f"Note: {result['note']}")

# 9. Print some real test tweets
print("\nSample tweets from real test dataset:")
for i in range(5):
    print(f"Tweet: '{test_df[text_column].iloc[i]}'")
    print(f"Label: {test_df['label'].iloc[i]}\n")

# 10. Model Export for Browser Extension
import joblib

print("Exporting model and vectorizer for browser extension...")
joblib.dump(model, 'covid_misinfo_model.joblib')
joblib.dump(vectorizer, 'covid_misinfo_vectorizer.joblib')

def predict_for_extension(text):
    text_processed = preprocess_text(text)
    text_tfidf = vectorizer.transform([text_processed])
    prediction = model.predict(text_tfidf)[0]
    
    try:
        proba = model.predict_proba(text_tfidf)[0]
        confidence = max(proba)
    except:
        confidence = 0.5
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "text": text
    }

print("\nExample of lightweight prediction for browser extension:")
example_text = "New research shows COVID-19 vaccines are safe and effective."
result = predict_for_extension(example_text)
print(f"Text: '{example_text}'")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")

