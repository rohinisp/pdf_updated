import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Basic configurations
app.config.update(
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', 'development-key'),
    DEBUG=os.environ.get('FLASK_DEBUG', '1') == '1',
    CACHE_TYPE="SimpleCache",
    CACHE_DEFAULT_TIMEOUT=7200
)

# Enable CORS
CORS(app)

# Initialize cache
cache = Cache(app)

# Basic routes
@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return render_template('error.html', error="Internal server error"), 500

@app.route('/api/test')
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        'status': 'success',
        'message': 'API is working'
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.is_json:
        return jsonify({'error': 'Page not found'}), 404
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    if request.is_json:
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('error.html', error="Internal server error"), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)