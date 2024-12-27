import logging
import time
import mimetypes
from typing import List, Dict, Any, Optional
from flask import render_template, request, jsonify, abort
import requests
from werkzeug.utils import secure_filename
from pdf_extractor import PDFExtractor
from llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all routes with the Flask app"""

    @app.route('/')
    def index():
        """Main page"""
        try:
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error rendering index page: {str(e)}")
            return render_template('error.html', error="Internal server error"), 500

    @app.route('/api/test', methods=['GET'])
    def test_endpoint():
        """Test endpoint to verify API functionality"""
        try:
            return jsonify({
                'status': 'success',
                'message': 'API is working',
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error in test endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def _is_valid_pdf(content: bytes) -> bool:
        """Validate PDF content"""
        try:
            # Use built-in mimetypes instead of python-magic
            if content.startswith(b'%PDF-'):
                return True

            # As fallback, try to detect MIME type
            import io
            temp_buffer = io.BytesIO(content)
            mime_type, _ = mimetypes.guess_type('test.pdf')
            return mime_type == 'application/pdf'
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

    def _download_pdf(url: str, max_size: int = 100 * 1024 * 1024) -> Optional[bytes]:
        """Download PDF with size limit and validation using memory-efficient streaming"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            with requests.get(url, headers=headers, stream=True, verify=False, timeout=30) as response:
                response.raise_for_status()

                # Check content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    logger.info(f"Downloading PDF of size: {size_mb:.2f}MB")
                    if int(content_length) > max_size:
                        raise ValueError(f"File too large: {size_mb:.1f}MB. Maximum size is {max_size/(1024*1024):.1f}MB")

                # Use BytesIO for memory-efficient accumulation
                import io
                buffer = io.BytesIO()
                chunk_size = 1024 * 1024  # 1MB chunks
                size = 0
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size += len(chunk)
                    if size > max_size:
                        raise ValueError(f"File too large. Maximum size is {max_size/(1024*1024):.1f}MB")
                    buffer.write(chunk)
                    if content_length:
                        progress = (size / int(content_length)) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")

                logger.info(f"Successfully downloaded {size/(1024*1024):.2f}MB")
                return buffer.getvalue()

        except requests.RequestException as e:
            logger.error(f"PDF download failed: {str(e)}")
            raise ValueError(f"Failed to download PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            raise ValueError(f"Error downloading PDF: {str(e)}")

    @app.route('/api/process-pdf', methods=['POST'])
    def process_pdf():
        """Unified endpoint for processing PDFs with questions"""
        try:
            # Initialize response metadata
            response_metadata = {
                'title': 'Untitled Document',
                'page_count': 0,
                'file_size': 0,
                'processing_time': None
            }

            # Get PDF content
            pdf_content = None
            start_time = time.time()

            if request.is_json:
                data = request.get_json()
                logger.debug("Processing JSON request")

                # Validate URL
                pdf_url = data.get('url')
                if not pdf_url:
                    return jsonify({'error': 'URL is required'}), 400

                # Handle both string and array URL formats
                url = pdf_url[1] if isinstance(pdf_url, list) and len(pdf_url) > 1 else pdf_url
                if not isinstance(url, str):
                    return jsonify({'error': 'Invalid URL format'}), 400

                # Download PDF
                logger.info(f"Downloading PDF from: {url}")
                try:
                    pdf_content = _download_pdf(url)
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400

                # Extract questions
                questions = []
                if isinstance(data.get('questions'), list):
                    questions = data['questions']
                elif data.get('question'):
                    questions = [data['question']]
                else:
                    # Look for numbered questions
                    i = 0
                    while str(i) in data:
                        questions.append(data[str(i)])
                        i += 1

            elif request.files and 'file' in request.files:
                file = request.files['file']
                if not file or not file.filename:
                    return jsonify({'error': 'No file selected'}), 400

                # Validate file size
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)

                max_size = 100 * 1024 * 1024  # 100MB limit
                if file_size > max_size:
                    return jsonify({'error': f'File too large ({file_size/(1024*1024):.1f}MB). Maximum size is {max_size/(1024*1024)}MB'}), 400

                pdf_content = file.read()
                questions = request.form.getlist('questions[]') or [request.form.get('question')] if request.form.get('question') else []

            else:
                return jsonify({'error': 'No URL or file provided'}), 400

            # Validate PDF content
            if not pdf_content or not _is_valid_pdf(pdf_content):
                return jsonify({'error': 'Invalid PDF format'}), 400

            # Extract text content
            logger.info(f"Processing PDF content: {len(pdf_content)} bytes")
            try:
                content, cache_key = PDFExtractor.extract_from_bytes(pdf_content)
                if not isinstance(content, dict) or 'raw_text' not in content:
                    raise ValueError("Invalid content structure")

                text = content['raw_text']
                response_metadata.update({
                    'title': content.get('title', 'Untitled Document'),
                    'page_count': content.get('metadata', {}).get('pages', 0),
                    'file_size': len(pdf_content),
                    'processing_time': time.time() - start_time
                })

            except Exception as e:
                logger.error(f"Text extraction failed: {str(e)}")
                return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500

            # Process questions if provided
            if questions and text:
                try:
                    logger.info(f"Processing {len(questions)} questions")
                    answers = LLMProcessor.process_questions(text, questions)

                    return jsonify({
                        'text': text,
                        'cache_key': cache_key,
                        'success': True,
                        'answers': answers,
                        'metadata': {
                            'document': response_metadata,
                            'processing': {
                                'questions_count': len(questions),
                                'cached_responses': sum(1 for a in answers if a.get('status') == 'success'),
                                'timestamp': int(time.time())
                            }
                        }
                    })

                except Exception as e:
                    logger.error(f"Question processing failed: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }), 500

            # Return extracted text if no questions
            return jsonify({
                'text': text,
                'cache_key': cache_key,
                'success': True,
                'metadata': response_metadata
            })

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/process-multiple-sources', methods=['POST'])
    def process_multiple_sources():
        """Process content from both PDF and webpage sources"""
        try:
            data = request.get_json()
            if not data or 'pdf_url' not in data or 'webpage_url' not in data:
                return jsonify({'error': 'Both PDF URL and webpage URL are required'}), 400

            pdf_url = data['pdf_url']
            webpage_url = data['webpage_url']

            logger.info(f"Processing multiple sources - PDF: {pdf_url}, Webpage: {webpage_url}")

            combined_content = PDFExtractor.extract_from_multiple_sources(pdf_url, webpage_url)

            # Process question if provided
            if 'question' in data:
                try:
                    text_to_use = (combined_content['content']['pdf_text'] or '') + '\n' + (combined_content['content']['web_text'] or '')
                    answer = LLMProcessor.get_answer(text_to_use, data['question'])
                    return jsonify({
                        'content': combined_content,
                        'answer': answer
                    })
                except Exception as e:
                    logger.error(f"Question answering failed: {str(e)}")
                    return jsonify({'error': 'Failed to process question'}), 500

            return jsonify(combined_content)

        except ValueError as e:
            logger.error(f"Value error in process_multiple_sources: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error in process_multiple_sources: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    return app