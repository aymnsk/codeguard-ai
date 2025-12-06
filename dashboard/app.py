#!/usr/bin/env python3
"""
CodeGuard AI Web Dashboard
Flask app with ML predictions and Groq API integration
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Flask imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# App configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'codeguard-ai-secret-key-2024'),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10MB max file size
    JSONIFY_PRETTYPRINT_REGULAR=True,
    TEMPLATES_AUTO_RELOAD=True
)

class CodeReviewAssistant:
    def __init__(self):
        self.ml_model = None
        self.vectorizer = None
        self.feature_names = []
        self.threshold = 0.5
        self.use_groq_fallback = False
        self.groq_api_keys = []
        self.current_key_idx = 0
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.getenv('MODEL_PATH', 'models/code_review_model_light.pkl')
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                # Try full model
                model_path = model_path.replace('_light.pkl', '.pkl')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"No model found at {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load components
            self.ml_model = model_data['model']
            
            # Recreate vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            if 'vectorizer_vocab' in model_data:
                self.vectorizer = TfidfVectorizer(vocabulary=model_data['vectorizer_vocab'])
                self.vectorizer.idf_ = model_data['vectorizer_idf']
            else:
                self.vectorizer = model_data.get('vectorizer')
            
            self.feature_names = model_data.get('feature_names', [])
            self.threshold = model_data.get('threshold', 0.5)
            self.use_groq_fallback = model_data.get('use_groq_fallback', False)
            
            # Load Groq keys if fallback is enabled
            if self.use_groq_fallback:
                self.groq_api_keys = self.load_groq_keys()
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Features: {len(self.feature_names)}")
            logger.info(f"   Threshold: {self.threshold}")
            logger.info(f"   Groq fallback: {self.use_groq_fallback}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def load_groq_keys(self):
        """Load Groq API keys from environment"""
        keys_str = os.getenv('GROQ_API_KEYS', '')
        if keys_str:
            keys = [k.strip() for k in keys_str.split(',') if k.strip()]
            logger.info(f"Loaded {len(keys)} Groq API keys")
            return keys
        return []
    
    def get_next_groq_key(self):
        """Rotate through available API keys"""
        if not self.groq_api_keys:
            return None
        
        key = self.groq_api_keys[self.current_key_idx]
        self.current_key_idx = (self.current_key_idx + 1) % len(self.groq_api_keys)
        return key
    
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        if not text:
            return ""
        
        import re
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s#]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def query_groq_api(self, commit_message, code_changes="", context=""):
        """Query Groq API for code review"""
        if not self.groq_api_keys:
            return None
        
        try:
            from groq import Groq
            
            api_key = self.get_next_groq_key()
            if not api_key:
                return None
            
            client = Groq(api_key=api_key)
            
            prompt = f"""As a senior code reviewer, analyze this commit:

Commit Message: {commit_message}

Code Changes: {code_changes[:800] if code_changes else "No code changes provided"}

Context: {context}

Respond with ONLY a JSON object:
{{
  "is_bug_fix": boolean,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "risk_level": "LOW/MEDIUM/HIGH",
  "suggestions": ["suggestion1", "suggestion2"],
  "review_focus": ["area1", "area2"]
}}"""
            
            response = client.chat.completions.create(
                model=os.getenv('GROQ_MODEL', 'llama3-70b-8192'),
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer. Return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.getenv('GROQ_TEMPERATURE', 0.3)),
                max_tokens=int(os.getenv('GROQ_MAX_TOKENS', 500))
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try direct parse
                return json.loads(result_text)
                
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None
    
    def predict_ml(self, commit_message, code_changes=""):
        """Make prediction using ML model"""
        if self.ml_model is None or self.vectorizer is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        processed = self.preprocess_text(commit_message)
        combined = processed
        
        if code_changes:
            combined += ' ' + str(code_changes)[:500]
        
        # Transform
        X_text = self.vectorizer.transform([combined])
        
        # Additional features
        message_len = len(commit_message)
        files_changed = 1
        
        if code_changes and 'diff --git' in code_changes:
            files_changed = code_changes.count('diff --git')
        
        # Create feature array
        if hasattr(X_text, 'toarray'):
            X_text_dense = X_text.toarray()
        else:
            X_text_dense = X_text
        
        # Simple feature vector (adjust based on your model)
        X_extra = np.array([[message_len, files_changed]])
        X = np.hstack([X_text_dense, X_extra])
        
        # Predict
        probability = self.ml_model.predict_proba(X)[0, 1]
        is_bug = probability > self.threshold
        
        # Confidence
        conf_score = abs(probability - 0.5) * 2
        if conf_score > 0.7:
            confidence = "HIGH"
        elif conf_score > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Risk
        if is_bug and probability > 0.7:
            risk = "HIGH - Requires thorough review"
        elif is_bug:
            risk = "MEDIUM - Review recommended"
        elif probability < 0.3:
            risk = "LOW - Routine change"
        else:
            risk = "MODERATE - Standard review"
        
        # Suggestions
        suggestions = self.generate_suggestions(commit_message, code_changes, probability)
        
        return {
            'is_bug_fix': bool(is_bug),
            'probability': float(probability),
            'confidence': confidence,
            'risk_level': risk,
            'suggestions': suggestions,
            'source': 'ml_model'
        }
    
    def generate_suggestions(self, message, diff, probability):
        """Generate review suggestions"""
        suggestions = []
        
        # Message quality
        if len(message) < 15:
            suggestions.append("Commit message is too brief. Add more context.")
        
        if not any(word in message.lower() for word in ['fix', 'add', 'remove', 'update', 'refactor', 'implement']):
            suggestions.append("Consider using more descriptive verbs in commit message.")
        
        # Diff analysis
        if diff:
            lines = diff.count('\n')
            if lines > 300:
                suggestions.append("Large change detected. Consider splitting into smaller commits.")
            
            if 'TODO' in diff.upper() or 'FIXME' in diff.upper():
                suggestions.append("Address TODO/FIXME comments before merging.")
            
            if '//' in diff and 'TODO' not in diff.upper():
                suggestions.append("Consider removing or documenting commented code.")
        
        # Probability-based
        if probability > 0.8:
            suggestions.append("High bug probability. Review error handling and edge cases.")
            suggestions.append("Add or update test cases for this change.")
        elif probability > 0.6:
            suggestions.append("Moderate bug risk. Review logic and data flow.")
        elif probability < 0.2:
            suggestions.append("Low risk change. Quick review should suffice.")
        
        if not suggestions:
            suggestions.append("Standard code review recommended.")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def predict(self, commit_message, code_changes="", use_llm=None):
        """Make prediction with optional LLM fallback"""
        # Determine if we should use LLM
        if use_llm is None:
            use_llm = self.use_groq_fallback
        
        # Get ML prediction
        ml_result = self.predict_ml(commit_message, code_changes)
        
        # Calculate ML confidence
        ml_confidence = abs(ml_result['probability'] - 0.5) * 2
        
        # Check if we need LLM fallback
        llm_min = float(os.getenv('LLM_MIN_CONFIDENCE', 0.4))
        llm_max = float(os.getenv('LLM_MAX_CONFIDENCE', 0.6))
        
        should_use_llm = (
            use_llm and 
            self.groq_api_keys and
            llm_min <= ml_confidence <= llm_max
        )
        
        if should_use_llm:
            logger.info("Using Groq API for uncertain prediction")
            llm_result = self.query_groq_api(commit_message, code_changes, "Web interface analysis")
            
            if llm_result:
                # Combine results
                combined_prob = (ml_result['probability'] + llm_result.get('confidence', 0.5)) / 2
                
                result = {
                    'is_bug_fix': llm_result.get('is_bug_fix', ml_result['is_bug_fix']),
                    'probability': float(combined_prob),
                    'confidence': 'HIGH' if combined_prob > 0.8 or combined_prob < 0.2 else 'MEDIUM',
                    'risk_level': llm_result.get('risk_level', ml_result['risk_level']),
                    'suggestions': llm_result.get('suggestions', ml_result['suggestions']),
                    'reasoning': llm_result.get('reasoning', ''),
                    'review_focus': llm_result.get('review_focus', []),
                    'source': 'groq_llm',
                    'ml_confidence': ml_confidence,
                    'llm_confidence': llm_result.get('confidence', 0.5)
                }
                
                # Add LLM-specific suggestions if not present
                if not result['suggestions'] and llm_result.get('suggestions'):
                    result['suggestions'] = llm_result['suggestions']
                
                return result
        
        # Return ML result
        ml_result['ml_confidence'] = ml_confidence
        return ml_result

# Initialize assistant
try:
    assistant = CodeReviewAssistant()
    MODEL_LOADED = True
    logger.info("✅ CodeReviewAssistant initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize assistant: {e}")
    MODEL_LOADED = False
    assistant = None

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', 
                         model_loaded=MODEL_LOADED,
                         groq_enabled=assistant.use_groq_fallback if MODEL_LOADED else False)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction API endpoint"""
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 503
    
    try:
        data = request.json
        commit_message = data.get('commit_message', '').strip()
        code_diff = data.get('code_diff', '').strip()
        use_llm = data.get('use_llm', None)
        
        if not commit_message:
            return jsonify({
                'error': 'Commit message is required',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = assistant.predict(commit_message, code_diff, use_llm)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['message_length'] = len(commit_message)
        result['code_length'] = len(code_diff)
        result['groq_available'] = bool(assistant.groq_api_keys)
        
        # Log prediction
        logger.info(f"Prediction: message='{commit_message[:50]}...', "
                   f"bug_fix={result['is_bug_fix']}, prob={result['probability']:.3f}")
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Advanced analysis with LLM (always uses Groq if available)"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        commit_message = data.get('commit_message', '').strip()
        code_diff = data.get('code_diff', '').strip()
        
        if not commit_message:
            return jsonify({'error': 'Commit message is required'}), 400
        
        # Always try to use Groq for detailed analysis
        if assistant.groq_api_keys:
            llm_result = assistant.query_groq_api(
                commit_message, 
                code_diff,
                "Detailed code review requested"
            )
            
            if llm_result:
                # Get ML prediction for comparison
                ml_result = assistant.predict_ml(commit_message, code_diff)
                
                return jsonify({
                    'status': 'success',
                    'analysis': {
                        'llm': llm_result,
                        'ml': ml_result,
                        'comparison': {
                            'agree': llm_result.get('is_bug_fix') == ml_result['is_bug_fix'],
                            'ml_confidence': abs(ml_result['probability'] - 0.5) * 2,
                            'llm_confidence': llm_result.get('confidence', 0.5)
                        }
                    }
                })
        
        # Fallback to ML only
        ml_result = assistant.predict_ml(commit_message, code_diff)
        
        return jsonify({
            'status': 'success',
            'analysis': {
                'ml': ml_result,
                'llm': None,
                'note': 'Groq API not available or failed'
            }
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'groq_available': bool(assistant.groq_api_keys) if MODEL_LOADED else False,
        'groq_keys_count': len(assistant.groq_api_keys) if MODEL_LOADED else 0,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    if MODEL_LOADED:
        status['model_info'] = {
            'features': len(assistant.feature_names),
            'threshold': assistant.threshold,
            'groq_fallback': assistant.use_groq_fallback
        }
    
    return jsonify(status)

@app.route('/api/stats')
def stats():
    """Statistics endpoint"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get some example predictions
    examples = [
        "Fix memory leak in parser",
        "Add new API endpoint",
        "Update documentation",
        "Fix segmentation fault",
        "Refactor code structure"
    ]
    
    stats_data = {
        'model': {
            'features': len(assistant.feature_names),
            'threshold': assistant.threshold,
            'groq_fallback': assistant.use_groq_fallback,
            'groq_keys': len(assistant.groq_api_keys)
        },
        'examples': []
    }
    
    # Test with examples
    for example in examples:
        try:
            result = assistant.predict_ml(example, "")
            stats_data['examples'].append({
                'message': example,
                'prediction': result['is_bug_fix'],
                'probability': result['probability']
            })
        except:
            pass
    
    return jsonify(stats_data)

@app.route('/api/version')
def version():
    """Version endpoint"""
    return jsonify({
        'name': 'CodeGuard AI',
        'version': '1.0.0',
        'description': 'ML + LLM Hybrid Code Review Assistant',
        'author': 'Your Name',
        'github': 'https://github.com/yourusername/codeguard-ai'
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'status': 'error'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large', 'status': 'error'}), 413

# ==================== MAIN ====================

def ensure_templates():
    """Ensure template files exist"""
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Create index.html if it doesn't exist
    index_path = templates_dir / 'index.html'
    if not index_path.exists():
        with open(index_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeGuard AI - ML-Powered Code Review</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            color: white;
        }
        
        .logo {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        
        h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .tagline {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-title {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }
        
        .card-title i {
            margin-right: 10px;
            font-size: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4a5568;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin-top: 10px;
        }
        
        .btn:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(0,0,0,0.1);
        }
        
        .btn-secondary {
            background: #48bb78;
        }
        
        .btn-danger {
            background: #f56565;
        }
        
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            margin-top: 30px;
            display: none;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .prediction-badge {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .bug-fix {
            background: #fed7d7;
            color: #c53030;
        }
        
        .not-bug {
            background: #c6f6d5;
            color: #276749;
        }
        
        .probability-bar {
            height: 25px;
            background: #e2e8f0;
            border-radius: 12px;
            margin: 20px 0;
            overflow: hidden;
            position: relative;
        }
        
        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #f56565);
            width: 0%;
            transition: width 1s ease;
            border-radius: 12px;
        }
        
        .probability-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: 600;
            color: #2d3748;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .detail-item {
            padding: 15px;
            background: #f7fafc;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .detail-label {
            font-size: 0.9rem;
            color: #718096;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2d3748;
        }
        
        .suggestions {
            margin-top: 25px;
        }
        
        .suggestion-item {
            padding: 12px 15px;
            background: #fff5f5;
            border-left: 4px solid #f56565;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            align-items: center;
        }
        
        .suggestion-item i {
            margin-right: 10px;
            color: #f56565;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #e2e8f0;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .status-healthy {
            background: #c6f6d5;
            color: #276749;
        }
        
        .status-unhealthy {
            background: #fed7d7;
            color: #c53030;
        }
        
        .api-info {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        footer {
            text-align: center;
            padding: 30px 0;
            color: white;
            opacity: 0.8;
            font-size: 0.9rem;
        }
        
        .toggle-switch {
            display: flex;
            align-items: center;
            margin: 15px 0;
        }
        
        .toggle-label {
            margin-left: 10px;
            font-weight: 500;
        }
        
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        
        .source-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 10px;
            background: #bee3f8;
            color: #2c5282;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-robot"></i>
            </div>
            <h1>CodeGuard AI</h1>
            <p class="tagline">ML + LLM Hybrid Code Review Assistant</p>
            <div>
                <span id="model-status" class="status-badge status-unhealthy">Model: Loading...</span>
                <span id="groq-status" class="status-badge status-unhealthy">Groq: Offline</span>
            </div>
        </header>
        
        <div class="main-content">
            <div class="card">
                <div class="card-title">
                    <i class="fas fa-code-branch"></i>
                    <h2>Code Review Analyzer</h2>
                </div>
                
                <div class="form-group">
                    <label for="commit-message">
                        <i class="fas fa-comment-alt"></i> Commit Message *
                    </label>
                    <textarea id="commit-message" rows="4" placeholder="Enter commit message...">Fix memory leak in data parser when handling null values</textarea>
                </div>
                
                <div class="form-group">
                    <label for="code-diff">
                        <i class="fas fa-file-code"></i> Code Diff (Optional)
                    </label>
                    <textarea id="code-diff" rows="10" placeholder="Paste git diff here...">diff --git a/parser.c b/parser.c
@@ -15,6 +15,9 @@ void parse_data(char* input) {
     Data* data = malloc(sizeof(Data));
     if (input == NULL) {
         return;
+    }
+    if (data == NULL) {
+        return; // Memory allocation failed
     }
     // Parse logic here</textarea>
                </div>
                
                <div class="toggle-switch">
                    <label class="switch">
                        <input type="checkbox" id="use-llm" checked>
                        <span class="slider"></span>
                    </label>
                    <span class="toggle-label">Use Groq AI for uncertain cases</span>
                </div>
                
                <button class="btn" onclick="analyzeCode()">
                    <i class="fas fa-search"></i> Analyze Code Review
                </button>
                
                <button class="btn btn-secondary" onclick="analyzeWithLLM()">
                    <i class="fas fa-brain"></i> Detailed Analysis (Groq AI)
                </button>
                
                <button class="btn btn-danger" onclick="clearForm()">
                    <i class="fas fa-trash"></i> Clear Form
                </button>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <i class="fas fa-chart-line"></i>
                    <h2>How It Works</h2>
                </div>
                
                <div style="line-height: 1.8;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">🤖 Dual Analysis System</h3>
                    
                    <div style="margin-bottom: 20px; padding: 15px; background: #f0f4ff; border-radius: 10px;">
                        <h4><i class="fas fa-microchip"></i> Machine Learning Model</h4>
                        <p>Trained on thousands of commits to detect bug fixes with 80%+ accuracy.</p>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li>Fast predictions (under 100ms)</li>
                            <li>Works offline</li>
                            <li>Lightweight (4GB RAM friendly)</li>
                        </ul>
                    </div>
                    
                    <div style="padding: 15px; background: #f0fff4; border-radius: 10px;">
                        <h4><i class="fas fa-brain"></i> Groq AI (Llama 3 70B)</h4>
                        <p>When ML is uncertain, we use Groq's LLM for detailed analysis.</p>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li>Understands complex code patterns</li>
                            <li>Provides detailed reasoning</li>
                            <li>Multiple API key rotation</li>
                        </ul>
                    </div>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px;">🎯 Use Cases</h3>
                    <ul style="margin-left: 20px;">
                        <li>Prioritize code reviews</li>
                        <li>Identify high-risk commits</li>
                        <li>Improve commit message quality</li>
                        <li>Catch bug patterns automatically</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>Analyzing your code...</h3>
            <p>ML model is processing your commit message and code changes</p>
        </div>
        
        <div class="result-card" id="result-card">
            <div class="result-header">
                <h2><i class="fas fa-chart-bar"></i> Analysis Result</h2>
                <div id="prediction-badge" class="prediction-badge">Loading...</div>
            </div>
            
            <div class="probability-bar">
                <div class="probability-fill" id="probability-fill"></div>
                <div class="probability-text" id="probability-text">0%</div>
            </div>
            
            <div class="details-grid">
                <div class="detail-item">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value" id="confidence">-</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Risk Level</div>
                    <div class="detail-value" id="risk-level">-</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Source</div>
                    <div class="detail-value" id="source">-</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Analysis Time</div>
                    <div class="detail-value" id="analysis-time">-</div>
                </div>
            </div>
            
            <div id="llm-reasoning" style="display: none; margin: 20px 0; padding: 15px; background: #f7fafc; border-radius: 10px;">
                <h4><i class="fas fa-lightbulb"></i> AI Reasoning</h4>
                <p id="reasoning-text"></p>
            </div>
            
            <div class="suggestions">
                <h3><i class="fas fa-clipboard-check"></i> Review Suggestions</h3>
                <div id="suggestions-list"></div>
            </div>
        </div>
        
        <div class="api-info">
            <h3><i class="fas fa-plug"></i> API Endpoints</h3>
            <p><strong>POST /api/predict</strong> - Get prediction (ML + optional Groq fallback)</p>
            <p><strong>POST /api/analyze</strong> - Detailed analysis with Groq AI</p>
            <p><strong>GET /api/health</strong> - Health check</p>
            <p><strong>GET /api/stats</strong> - Model statistics</p>
        </div>
        
        <footer>
            <p>CodeGuard AI v1.0.0 | ML + LLM Hybrid Code Review | Built with ❤️ for developers</p>
            <p>Deploy on Railway, Render, or your own server</p>
        </footer>
    </div>
    
    <script>
        // Check health on load
        window.onload = function() {
            checkHealth();
        };
        
        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const modelStatus = document.getElementById('model-status');
                const groqStatus = document.getElementById('groq-status');
                
                if (data.model_loaded) {
                    modelStatus.textContent = 'Model: Loaded';
                    modelStatus.className = 'status-badge status-healthy';
                }
                
                if (data.groq_available) {
                    groqStatus.textContent = `Groq: ${data.groq_keys_count} keys`;
                    groqStatus.className = 'status-badge status-healthy';
                } else {
                    groqStatus.textContent = 'Groq: Not available';
                    groqStatus.className = 'status-badge status-unhealthy';
                }
                
            } catch (error) {
                console.error('Health check failed:', error);
            }
        }
        
        async function analyzeCode() {
            const commitMessage = document.getElementById('commit-message').value;
            const codeDiff = document.getElementById('code-diff').value;
            const useLLM = document.getElementById('use-llm').checked;
            
            if (!commitMessage.trim()) {
                alert('Please enter a commit message');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';
            
            try {
                const startTime = Date.now();
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        commit_message: commitMessage,
                        code_diff: codeDiff,
                        use_llm: useLLM
                    })
                });
                
                const data = await response.json();
                const endTime = Date.now();
                
                if (data.status === 'error') {
                    throw new Error(data.error);
                }
                
                displayResult(data.prediction, endTime - startTime);
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        async function analyzeWithLLM() {
            const commitMessage = document.getElementById('commit-message').value;
            const codeDiff = document.getElementById('code-diff').value;
            
            if (!commitMessage.trim()) {
                alert('Please enter a commit message');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        commit_message: commitMessage,
                        code_diff: codeDiff
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'error') {
                    throw new Error(data.error);
                }
                
                if (data.analysis.llm) {
                    // Display LLM result
                    const prediction = {
                        ...data.analysis.llm,
                        source: 'groq_llm',
                        probability: data.analysis.llm.confidence || 0.5
                    };
                    
                    displayResult(prediction, 0);
                    
                    // Show reasoning
                    if (data.analysis.llm.reasoning) {
                        document.getElementById('llm-reasoning').style.display = 'block';
                        document.getElementById('reasoning-text').textContent = 
                            data.analysis.llm.reasoning;
                    }
                    
                } else {
                    // Fallback to ML
                    displayResult(data.analysis.ml, 0);
                }
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResult(prediction, timeMs) {
            const resultCard = document.getElementById('result-card');
            const badge = document.getElementById('prediction-badge');
            const probabilityFill = document.getElementById('probability-fill');
            const probabilityText = document.getElementById('probability-text');
            
            // Update badge
            if (prediction.is_bug_fix) {
                badge.textContent = '🚨 POTENTIAL BUG FIX';
                badge.className = 'prediction-badge bug-fix';
            } else {
                badge.textContent = '✅ ROUTINE CHANGE';
                badge.className = 'prediction-badge not-bug';
            }
            
            // Update probability
            const probability = prediction.probability * 100;
            probabilityFill.style.width = probability + '%';
            probabilityText.textContent = probability.toFixed(1) + '%';
            
            // Update details
            document.getElementById('confidence').textContent = prediction.confidence;
            document.getElementById('risk-level').textContent = prediction.risk_level;
            document.getElementById('source').innerHTML = 
                prediction.source.toUpperCase() + 
                (prediction.source === 'groq_llm' ? ' <span class="source-badge">AI</span>' : '');
            
            document.getElementById('analysis-time').textContent = 
                timeMs > 0 ? timeMs + 'ms' : '<1s';
            
            // Update suggestions
            const suggestionsList = document.getElementById('suggestions-list');
            suggestionsList.innerHTML = '';
            
            if (prediction.suggestions && prediction.suggestions.length > 0) {
                prediction.suggestions.forEach(suggestion => {
                    const div = document.createElement('div');
                    div.className = 'suggestion-item';
                    div.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${suggestion}`;
                    suggestionsList.appendChild(div);
                });
            } else {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                div.innerHTML = '<i class="fas fa-check-circle"></i> No specific suggestions';
                suggestionsList.appendChild(div);
            }
            
            // Hide LLM reasoning if not present
            if (!prediction.reasoning) {
                document.getElementById('llm-reasoning').style.display = 'none';
            }
            
            // Show result card
            resultCard.style.display = 'block';
            
            // Scroll to result
            resultCard.scrollIntoView({ behavior: 'smooth' });
        }
        
        function clearForm() {
            document.getElementById('commit-message').value = '';
            document.getElementById('code-diff').value = '';
            document.getElementById('result-card').style.display = 'none';
        }
        
        // Example commit messages
        const examples = [
            "Fix memory leak in data parser when handling null values",
            "Add support for JSON export in API",
            "Update documentation for new authentication system",
            "Fix segmentation fault in array boundary check",
            "Implement rate limiting for user endpoints",
            "Refactor database connection pooling",
            "Fix race condition in concurrent file access",
            "Add unit tests for payment processing"
        ];
        
        // Load a random example
        function loadExample() {
            const randomExample = examples[Math.floor(Math.random() * examples.length)];
            document.getElementById('commit-message').value = randomExample;
        }
        
        // Press Ctrl+E to load example
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'e') {
                e.preventDefault();
                loadExample();
            }
        });
    </script>
</body>
</html>''')
        
        # Create static directory
        static_dir = Path(__file__).parent / 'static'
        static_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Template files ensured")

if __name__ == '__main__':
    # Ensure templates exist
    ensure_templates()
    
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting CodeGuard AI on http://{host}:{port}")
    logger.info(f"📊 Model loaded: {MODEL_LOADED}")
    logger.info(f"🤖 Groq fallback: {assistant.use_groq_fallback if MODEL_LOADED else False}")
    
    app.run(host=host, port=port, debug=debug)
