#!/usr/bin/env python3
"""
Training Script with Groq API Fallback
Trains ML model and sets up LLM fallback system
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Utilities
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodeReviewModel:
    def __init__(self, use_groq_fallback=True):
        self.use_groq_fallback = use_groq_fallback
        self.vectorizer = None
        self.model = None
        self.feature_names = []
        self.class_weights = None
        self.threshold = 0.5
        
        # Load Groq API keys if fallback is enabled
        if self.use_groq_fallback:
            self.groq_api_keys = self.load_groq_keys()
            self.current_key_idx = 0
        
        # Initialize text processor
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def load_groq_keys(self):
        """Load Groq API keys from environment"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            import os
            
            keys_str = os.getenv('GROQ_API_KEYS', '')
            if keys_str:
                keys = [k.strip() for k in keys_str.split(',') if k.strip()]
                logger.info(f"Loaded {len(keys)} Groq API keys")
                return keys
            else:
                logger.warning("No Groq API keys found in environment")
                return []
        except:
            return []
    
    def get_next_groq_key(self):
        """Rotate through available Groq API keys"""
        if not self.groq_api_keys:
            return None
        
        key = self.groq_api_keys[self.current_key_idx]
        self.current_key_idx = (self.current_key_idx + 1) % len(self.groq_api_keys)
        return key
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for commit messages"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important symbols
        text = re.sub(r'[^\w\s#\-@]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        
        # Keep important commit-related terms
        commit_terms = {
            'fix', 'bug', 'error', 'issue', 'feature', 'add', 'remove', 'update',
            'patch', 'resolve', 'implement', 'support', 'memory', 'leak', 'crash',
            'fail', 'test', 'doc', 'refactor', 'optimize', 'security', 'vulnerability'
        }
        
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                # Keep if it's a commit term or has reasonable length
                if token in commit_terms or len(token) >= 3:
                    # Handle common variations
                    if token.endswith('ing'):
                        token = token[:-3]
                    elif token.endswith('ed'):
                        token = token[:-2]
                    filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def extract_additional_features(self, df):
        """Extract engineered features from commit data"""
        logger.info("Extracting additional features...")
        
        # Text-based features
        df['message_length'] = df['message'].str.len()
        df['message_word_count'] = df['message'].str.split().str.len()
        df['has_issue_ref'] = df['message'].str.contains(r'#\d+', na=False)
        df['has_error_keyword'] = df['message'].str.contains(
            r'error|exception|fail|crash|bug|fix', 
            case=False, 
            na=False
        )
        
        # Code change features
        df['code_change_length'] = df['code_changes'].str.len().fillna(0)
        df['has_test_file'] = df['files_list'].str.contains(
            'test|spec', 
            case=False, 
            na=False
        )
        
        # Ratio features
        df['change_density'] = np.where(
            df['message_length'] > 0,
            df['total_changes'] / df['message_length'],
            0
        )
        
        # Time features (if available)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        else:
            df['hour_of_day'] = 0
            df['day_of_week'] = 0
        
        # File extension features
        def get_file_extensions(files_str):
            if pd.isna(files_str):
                return []
            extensions = []
            for file in files_str.split(','):
                if '.' in file:
                    ext = file.split('.')[-1].strip()
                    if ext and len(ext) <= 5:
                        extensions.append(ext)
            return list(set(extensions))
        
        df['file_extensions'] = df['files_list'].apply(get_file_extensions)
        df['num_extensions'] = df['file_extensions'].str.len()
        df['has_py_files'] = df['file_extensions'].apply(
            lambda exts: 'py' in exts if isinstance(exts, list) else False
        )
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix for training"""
        logger.info("Preprocessing commit messages...")
        
        # Preprocess text
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Extract additional features
        df = self.extract_additional_features(df)
        
        # Combine text features
        df['combined_text'] = df['processed_message']
        if 'code_changes' in df.columns:
            has_code_mask = df['code_changes'].notna() & (df['code_changes'] != '')
            df.loc[has_code_mask, 'combined_text'] = (
                df['processed_message'] + ' [CODE] ' + 
                df['code_changes'].str[:300]  # Limit code changes
            )
        
        # Text vectorization
        logger.info("Vectorizing text features...")
        self.vectorizer = TfidfVectorizer(
            max_features=800,  # Keep it reasonable for 4GB RAM
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        X_text = self.vectorizer.fit_transform(df['combined_text'])
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        
        # Prepare numeric features
        numeric_features = [
            'message_length', 'message_word_count', 'files_changed',
            'total_changes', 'code_change_length', 'change_density',
            'hour_of_day', 'day_of_week', 'num_extensions'
        ]
        
        # Only include features that exist
        existing_numeric = [f for f in numeric_features if f in df.columns]
        X_numeric = df[existing_numeric].fillna(0).values
        
        # Prepare binary features
        binary_features = [
            'has_issue_ref', 'has_error_keyword', 'has_test_file', 'has_py_files'
        ]
        
        existing_binary = [f for f in binary_features if f in df.columns]
        X_binary = df[existing_binary].fillna(False).astype(int).values
        
        # Combine all features
        X = np.hstack([X_text.toarray(), X_numeric, X_binary])
        
        # Update feature names
        self.feature_names.extend(existing_numeric)
        self.feature_names.extend(existing_binary)
        
        # Target variable
        y = df['is_bug_fix'].astype(int).values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        return X, y, df
    
    def handle_class_imbalance(self, X, y):
        """Handle imbalanced classes using SMOTE"""
        logger.info("Checking class balance...")
        
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Calculate class weights for Random Forest
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        logger.info(f"Class weights: {self.class_weights}")
        
        # Apply SMOTE if severe imbalance
        if min(counts) / max(counts) < 0.3:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts) - 1))
            X, y = smote.fit_resample(X, y)
            logger.info(f"After SMOTE - Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train(self, df):
        """Train the model with cross-validation"""
        logger.info("Starting model training...")
        
        # Prepare features
        X, y, df = self.prepare_features(df)
        
        # Handle class imbalance
        X, y = self.handle_class_imbalance(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Testing samples: {X_test.shape[0]}")
        logger.info(f"Bug fixes in train: {y_train.mean():.2%}")
        logger.info(f"Bug fixes in test: {y_test.mean():.2%}")
        
        # Train Random Forest with class weights
        logger.info("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,  # Use all cores
            random_state=42,
            verbose=1
        )
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        logger.info(f"CV ROC-AUC scores: {cv_scores}")
        logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        logger.info("\n" + "="*50)
        logger.info("MODEL EVALUATION")
        logger.info("="*50)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Not Bug', 'Bug Fix'])
        logger.info(f"\nClassification Report:\n{report}")
        
        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"ROC-AUC Score: {auc:.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info(f"\nTop 15 most important features:")
            for i in range(min(15, len(indices))):
                idx = indices[i]
                if idx < len(self.feature_names):
                    logger.info(f"  {i+1:2d}. {self.feature_names[idx]:30s} {importances[idx]:.4f}")
        
        # Find optimal threshold
        self.find_optimal_threshold(y_test, y_pred_proba)
        
        return self
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal probability threshold"""
        from sklearn.metrics import f1_score
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Optimal probability threshold: {self.threshold:.3f} (F1: {best_f1:.3f})")
        
        return best_threshold
    
    def query_groq_api(self, commit_message, code_changes="", context=""):
        """Query Groq API for code review analysis"""
        if not self.groq_api_keys:
            return None
        
        try:
            from groq import Groq
            
            api_key = self.get_next_groq_key()
            if not api_key:
                return None
            
            client = Groq(api_key=api_key)
            
            prompt = f"""As a senior code reviewer, analyze this commit and determine if it's likely a bug fix.

Commit Message: {commit_message}

Code Changes: {code_changes[:1000] if code_changes else "No code changes provided"}

Context: {context}

Please analyze and respond with ONLY a JSON object in this exact format:
{{
  "is_bug_fix": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "risk_level": "LOW/MEDIUM/HIGH",
  "suggestions": ["suggestion1", "suggestion2"]
}}

Focus on:
1. Does the commit message indicate bug fixing?
2. Do code changes look like fixes (error handling, null checks, etc.)?
3. What's the potential risk?
"""
            
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Try to parse as is
                return json.loads(result_text)
                
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None
    
    def predict_with_fallback(self, commit_message, code_changes="", context=""):
        """Predict with ML model and Groq fallback"""
        # First, get ML prediction
        ml_result = self.predict_ml(commit_message, code_changes)
        
        # Determine if we need LLM fallback
        confidence = abs(ml_result['probability'] - 0.5) * 2  # Convert to 0-1 confidence
        
        llm_min_conf = float(os.getenv('LLM_MIN_CONFIDENCE', 0.4))
        llm_max_conf = float(os.getenv('LLM_MAX_CONFIDENCE', 0.6))
        
        use_llm = (
            self.use_groq_fallback and 
            self.groq_api_keys and
            llm_min_conf <= confidence <= llm_max_conf
        )
        
        if use_llm:
            logger.info("Using Groq API fallback for uncertain prediction")
            llm_result = self.query_groq_api(commit_message, code_changes, context)
            
            if llm_result:
                # Combine ML and LLM results
                combined_confidence = (confidence + llm_result.get('confidence', 0.5)) / 2
                
                return {
                    'is_bug_fix': llm_result.get('is_bug_fix', ml_result['is_bug_fix']),
                    'probability': llm_result.get('confidence', ml_result['probability']),
                    'confidence': 'HIGH' if combined_confidence > 0.7 else 'MEDIUM',
                    'risk_level': llm_result.get('risk_level', 'MEDIUM'),
                    'suggestions': llm_result.get('suggestions', []),
                    'reasoning': llm_result.get('reasoning', ''),
                    'source': 'groq_llm',
                    'ml_confidence': confidence,
                    'combined_confidence': combined_confidence
                }
        
        # Return ML result with source info
        ml_result['source'] = 'ml_model'
        ml_result['ml_confidence'] = confidence
        return ml_result
    
    def predict_ml(self, commit_message, code_changes=""):
        """Make prediction using ML model only"""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess
        processed_msg = self.preprocess_text(commit_message)
        combined_text = processed_msg
        
        if code_changes:
            combined_text += ' [CODE] ' + str(code_changes)[:300]
        
        # Transform text
        X_text = self.vectorizer.transform([combined_text])
        
        # Prepare additional features
        message_length = len(commit_message)
        message_word_count = len(commit_message.split())
        has_error_keyword = any(word in commit_message.lower() for word in 
                              ['error', 'exception', 'fail', 'crash', 'bug', 'fix'])
        has_issue_ref = '#' in commit_message
        
        # Estimate files changed from diff
        files_changed = 1
        if code_changes and 'diff --git' in code_changes:
            files_changed = code_changes.count('diff --git')
        
        # Create feature vector
        additional_features = np.array([[
            message_length,
            message_word_count,
            files_changed,
            len(code_changes) if code_changes else 0,
            has_error_keyword,
            has_issue_ref
        ]])
        
        # Combine features
        X = np.hstack([X_text.toarray(), additional_features])
        
        # Predict
        probability = self.model.predict_proba(X)[0, 1]
        prediction = probability > self.threshold
        
        # Determine confidence
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.7:
            confidence_level = 'HIGH'
        elif confidence_score > 0.4:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'
        
        # Risk assessment
        if prediction and probability > 0.7:
            risk = "HIGH - Requires thorough review"
        elif prediction:
            risk = "MEDIUM - Review recommended"
        elif probability < 0.3:
            risk = "LOW - Routine change"
        else:
            risk = "MODERATE - Standard review"
        
        return {
            'is_bug_fix': bool(prediction),
            'probability': float(probability),
            'confidence': confidence_level,
            'risk_level': risk,
            'suggestions': self.generate_ml_suggestions(commit_message, code_changes, probability)
        }
    
    def generate_ml_suggestions(self, commit_message, code_changes, probability):
        """Generate review suggestions based on ML analysis"""
        suggestions = []
        
        # Check commit message quality
        if len(commit_message) < 10:
            suggestions.append("Commit message is too short. Provide more context.")
        elif len(commit_message.split()) < 3:
            suggestions.append("Add more details to commit message.")
        
        # Check for common patterns
        if probability > 0.7:
            suggestions.append("High bug probability detected. Please add test cases.")
            suggestions.append("Consider peer review before merging.")
        
        if 'TODO' in code_changes.upper() or 'FIXME' in code_changes.upper():
            suggestions.append("Remove or address TODO/FIXME comments before merging.")
        
        if code_changes and code_changes.count('\n') > 200:
            suggestions.append("Large diff detected. Consider splitting into smaller commits.")
        
        if not code_changes and probability > 0.5:
            suggestions.append("No code changes detected. Verify this is intentional.")
        
        if len(suggestions) == 0:
            suggestions.append("Standard code review recommended.")
        
        return suggestions
    
    def save(self, path="models/code_review_model.pkl"):
        """Save model and components"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'class_weights': self.class_weights,
            'use_groq_fallback': self.use_groq_fallback,
            'groq_api_keys': self.groq_api_keys if hasattr(self, 'groq_api_keys') else [],
            'training_date': datetime.now().isoformat()
        }
        
        # Save full model
        joblib.dump(model_data, path)
        logger.info(f"Full model saved to {path}")
        
        # Save lightweight version for inference
        light_path = path.replace('.pkl', '_light.pkl')
        light_data = {
            'model': self.model,
            'vectorizer_vocab': self.vectorizer.vocabulary_,
            'vectorizer_idf': self.vectorizer.idf_,
            'feature_names': self.feature_names[:100],  # Keep only important ones
            'threshold': self.threshold,
            'use_groq_fallback': self.use_groq_fallback
        }
        
        with open(light_path, 'wb') as f:
            pickle.dump(light_data, f, protocol=4)
        
        logger.info(f"Lightweight model saved to {light_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'features_count': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'threshold': self.threshold,
            'has_groq_fallback': self.use_groq_fallback and bool(self.groq_api_keys)
        }
        
        metadata_path = path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return path
    
    def load(self, path="models/code_review_model.pkl"):
        """Load saved model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data.get('feature_names', [])
        self.threshold = model_data.get('threshold', 0.5)
        self.class_weights = model_data.get('class_weights')
        self.use_groq_fallback = model_data.get('use_groq_fallback', True)
        
        if self.use_groq_fallback:
            self.groq_api_keys = self.load_groq_keys()
            self.current_key_idx = 0
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Features: {len(self.feature_names)}, Threshold: {self.threshold}")
        
        return self

def main():
    """Main training function"""
    logger.info("Starting CodeGuard AI Model Training")
    logger.info("="*50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load dataset
    data_path = os.getenv('DATA_PATH', 'data/commits_dataset.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please run data collection script first:")
        logger.info("python scripts/collect_data.py")
        return
    
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset loaded: {len(df)} commits")
    logger.info(f"Bug fixes: {df['is_bug_fix'].sum()} ({df['is_bug_fix'].mean():.1%})")
    
    # Check if we have enough data
    if len(df) < 50:
        logger.warning(f"Very small dataset ({len(df)} samples). Results may not be reliable.")
    
    # Train model
    use_groq = os.getenv('USE_LLM_FALLBACK', 'true').lower() == 'true'
    model = CodeReviewModel(use_groq_fallback=use_groq)
    
    logger.info(f"Training with Groq fallback: {use_groq}")
    if use_groq and model.groq_api_keys:
        logger.info(f"Available Groq API keys: {len(model.groq_api_keys)}")
    
    # Train
    model.train(df)
    
    # Save model
    model_path = os.getenv('MODEL_PATH', 'models/code_review_model.pkl')
    model.save(model_path)
    
    # Test with examples
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE PREDICTIONS")
    logger.info("="*50)
    
    examples = [
        {
            "message": "Fix memory leak in data parser when handling null values",
            "code": "diff --git a/parser.c b/parser.c\n@@ -15,6 +15,9 @@ void parse_data(char* input) {\n     Data* data = malloc(sizeof(Data));\n     if (input == NULL) {\n         return;\n+    }\n+    if (data == NULL) {\n+        return; // Memory allocation failed\n     }"
        },
        {
            "message": "Add new feature for JSON export",
            "code": "def export_json(data):\n    return json.dumps(data)"
        },
        {
            "message": "Fix segmentation fault in array access",
            "code": "- return array[index];\n+ if (index < 0 || index >= array_size) return NULL;\n+ return array[index];"
        },
        {
            "message": "Update documentation",
            "code": ""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        logger.info(f"\nExample {i}: '{example['message'][:50]}...'")
        
        # Try with fallback
        result = model.predict_with_fallback(
            example['message'], 
            example['code'],
            "Example commit for testing"
        )
        
        logger.info(f"  Prediction: {'🚨 BUG FIX' if result['is_bug_fix'] else '✅ NOT BUG'}")
        logger.info(f"  Probability: {result['probability']:.3f}")
        logger.info(f"  Confidence: {result['confidence']}")
        logger.info(f"  Risk: {result['risk_level']}")
        logger.info(f"  Source: {result.get('source', 'unknown')}")
        
        if result.get('reasoning'):
            logger.info(f"  Reasoning: {result['reasoning'][:100]}...")
        
        if result.get('suggestions'):
            logger.info(f"  Suggestions: {result['suggestions'][0]}")
    
    logger.info("\n" + "="*50)
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Light model: {model_path.replace('.pkl', '_light.pkl')}")
    logger.info("Ready for deployment!")

if __name__ == "__main__":
    main()
