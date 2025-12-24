"""
Resume Classifier Utilities Module
This module contains all the preprocessing, training, and prediction functions
for the AI-Powered Resume Classifier.
"""

import re
import pickle
from pathlib import Path

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class ResumePreprocessor:
    """Handle all text preprocessing for resume data"""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    def clean_text(self, text):
        """Remove special characters and numbers"""
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return text

    def lowercase_text(self, text):
        """Convert text to lowercase"""
        return text.lower()

    def remove_stopwords_and_tokenize(self, text):
        """Remove stopwords and tokenize"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return filtered_words

    def apply_stemming(self, word_list):
        """Apply stemming to word list"""
        return [self.stemmer.stem(word) for word in word_list]

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Remove URLs
        text = self.remove_urls(text)
        # Clean special characters
        text = self.clean_text(text)
        # Lowercase
        text = self.lowercase_text(text)
        # Tokenize and remove stopwords
        words = self.remove_stopwords_and_tokenize(text)
        # Apply stemming
        words = self.apply_stemming(words)
        # Join back to string
        return ' '.join(words)


class ResumeClassifier:
    """Main classifier class for resume classification"""

    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.preprocessor = ResumePreprocessor()
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.model = None
        self.is_trained = False

    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the resume classifier"""
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)

        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )

        # Initialize and train model
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                random_state=random_state
            )
        elif self.model_type == 'knn':
            self.model = OneVsRestClassifier(KNeighborsClassifier())
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        )

        return {
            'accuracy': accuracy,
            'report': report,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape
        }

    def predict(self, resume_text):
        """Predict resume category"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Preprocess
        processed = self.preprocessor.preprocess(resume_text)

        # Vectorize
        X_tfidf = self.tfidf_vectorizer.transform([processed])

        # Predict
        prediction = self.model.predict(X_tfidf)[0]
        probabilities = None

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_tfidf)[0]
            probabilities = dict(zip(self.label_encoder.classes_, probs))

        # Get category name
        category = self.label_encoder.inverse_transform([prediction])[0]

        return {
            'category': category,
            'encoded_label': prediction,
            'probabilities': probabilities
        }

    def save_model(self, model_dir='models'):
        """Save trained model and components"""
        Path(model_dir).mkdir(exist_ok=True)

        model_path = Path(model_dir) / f'{self.model_type}_model.pkl'
        vectorizer_path = Path(model_dir) / 'tfidf_vectorizer.pkl'
        encoder_path = Path(model_dir) / 'label_encoder.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        return {
            'model': str(model_path),
            'vectorizer': str(vectorizer_path),
            'encoder': str(encoder_path)
        }

    def load_model(self, model_dir='models'):
        """Load saved model and components"""
        model_path = Path(model_dir) / f'{self.model_type}_model.pkl'
        vectorizer_path = Path(model_dir) / 'tfidf_vectorizer.pkl'
        encoder_path = Path(model_dir) / 'label_encoder.pkl'

        if not all([model_path.exists(), vectorizer_path.exists(), encoder_path.exists()]):
            return False

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.is_trained = True
        return True

    def get_categories(self):
        """Get list of all job categories"""
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)
