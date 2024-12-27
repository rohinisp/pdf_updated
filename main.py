import os
import logging
from dotenv import load_dotenv
from app import app

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    try:
        logger.info("Starting Flask application server...")
        port = int(os.environ.get('PORT', 5000))

        # Check for required environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            logger.info("Application will start but AI features may be limited")

        # Start the Flask server
        logger.info(f"Starting Flask server on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True
        )

    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()