import logging
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from pypdf import PdfReader
from io import BytesIO
from app import cache
from langchain import embeddings, document_loaders, text_splitter, vectorstores

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PDFExtractor:
    @staticmethod
    def generate_cache_key(content: bytes) -> str:
        """Generate a cache key based on PDF content"""
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def process_document(cls, file_path: str) -> Tuple[Dict[str, any], str]:
        """Process PDF using PyPDF for complete content extraction"""
        try:
            import time
            start_time = time.time()

            # Load PDF using PyPDF directly for more reliable extraction
            logger.info(f"Loading PDF from path: {file_path}")
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                num_pages = len(reader.pages)
                logger.info(f"PDF loaded: {num_pages} pages")
                
                # Extract text from each page with complete content extraction
                pages_content = []
                for i, page in enumerate(reader.pages):
                    try:
                        # Extract text with better whitespace handling
                        content = page.extract_text(layout=True)
                        content = ' '.join(content.split())  # Normalize whitespace
                        
                        if content:
                            # Format page content with clear separation
                            page_text = f"Page {i+1}:\n{content}\n"
                            pages_content.append(page_text)
                            logger.debug(f"Successfully extracted page {i+1} - Length: {len(content)} chars")
                        else:
                            logger.warning(f"No content extracted from page {i+1}")
                            # Try alternate extraction method
                            try:
                                content = '\n'.join(text.get_text() for text in page.extract_text_lines())
                                if content:
                                    page_text = f"Page {i+1}:\n{content}\n"
                                    pages_content.append(page_text)
                                    logger.debug(f"Extracted page {i+1} using alternate method")
                            except Exception as inner_e:
                                logger.error(f"Alternate extraction failed for page {i+1}: {str(inner_e)}")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                        continue
                
            load_time = time.time() - start_time
            logger.info(f"PDF processed in {load_time:.2f}s ({num_pages/load_time:.1f} pages/s)")

            # Join all page content with clear separation and chunking for large content
            raw_text = "\n\n".join(pages_content)
            logger.debug(f"Total extracted content length: {len(raw_text)}")

            # Extract metadata if available
            title = 'PDF Document'
            try:
                if reader.metadata:
                    title = reader.metadata.get('/Title', 'PDF Document')
            except Exception as e:
                logger.warning(f"Failed to extract metadata: {str(e)}")

            # Create structured content dictionary with enhanced metadata
            structured_content = {
                'title': title,
                'metadata': {
                    'source': reader.metadata.get('/Creator', 'Unknown') if reader.metadata else 'Unknown',
                    'pages': num_pages,
                    'processing_time': load_time,
                    'content_length': len(raw_text),
                    'average_page_length': len(raw_text) // max(num_pages, 1),
                    'processed_at': int(time.time())
                },
                'raw_text': raw_text
            }

            # Generate cache key from raw text
            cache_key = cls.generate_cache_key(raw_text.encode('utf-8'))

            # Store document content in cache with automatic cleanup
            cache.set(
                f"doc_content_{cache_key}",
                structured_content,
                timeout=3600  # 1 hour cache
            )
            
            # Cleanup any associated chunks after processing
            try:
                reader.stream.close()
                if hasattr(reader, '_garbage'):
                    reader._garbage.clear()
            except Exception as e:
                logger.warning(f"Cleanup warning: {str(e)}")
                # Non-critical error, continue processing

            logger.info(f"Document processed successfully. Cache key: {cache_key}")
            return structured_content, cache_key

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Error processing document: {str(e)}")


    @classmethod
    def extract_from_bytes(cls, pdf_bytes: bytes) -> Tuple[Dict[str, any], str]:
        """Extract text from PDF bytes with improved extraction"""
        try:
            # Generate cache key first
            cache_key = cls.generate_cache_key(pdf_bytes)

            # Check cache
            cached_content = cache.get(f"doc_content_{cache_key}")
            if cached_content:
                logger.info("Returning cached content")
                return cached_content, cache_key

            # Process PDF directly from bytes with enhanced extraction
            import io
            import time
            start_time = time.time()

            # Read PDF from bytes with proper encoding
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            logger.info(f"PDF loaded from bytes: {num_pages} pages")

            # Extract text from each page with complete content handling
            pages_content = []
            for i, page in enumerate(reader.pages):
                try:
                    # Extract text with better whitespace handling
                    content = page.extract_text(layout=True)
                    content = ' '.join(content.split())  # Normalize whitespace
                    
                    if content:
                        # Format page content with clear separation
                        page_text = f"Page {i+1}:\n{content}\n"
                        pages_content.append(page_text)
                        logger.debug(f"Successfully extracted page {i+1} - Length: {len(content)} chars")
                    else:
                        logger.warning(f"No content extracted from page {i+1}")
                        # Try alternate extraction method
                        try:
                            content = '\n'.join(text.get_text() for text in page.extract_text_lines())
                            if content:
                                page_text = f"Page {i+1}:\n{content}\n"
                                pages_content.append(page_text)
                                logger.debug(f"Extracted page {i+1} using alternate method")
                        except Exception as inner_e:
                            logger.error(f"Alternate extraction failed for page {i+1}: {str(inner_e)}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                    continue

            load_time = time.time() - start_time
            logger.info(f"PDF processed in {load_time:.2f}s ({num_pages/load_time:.1f} pages/s)")

            # Join all page content
            raw_text = "\n\n".join(pages_content)
            logger.debug(f"Total extracted content length: {len(raw_text)}")

            # Create structured content
            content = {
                'title': 'PDF Document',  # Default title
                'metadata': {
                    'pages': num_pages,
                    'processing_time': load_time,
                    'content_length': len(raw_text)
                },
                'raw_text': raw_text
            }

            # Cache the content
            cache.set(
                f"doc_content_{cache_key}",
                content,
                timeout=3600  # 1 hour cache
            )

            return content, cache_key

        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    @staticmethod
    def semantic_search(cache_key: str, query: str, k: int = 5) -> List[dict]:
        """
        Perform semantic search on previously processed PDF content
        Args:
            cache_key: The cache key for the vector store
            query: The search query
            k: Number of results to return (default: 3)
        Returns:
            List of dictionaries containing search results with scores
        """
        try:
            # Verify cache key is valid
            if not cache_key:
                raise ValueError("Invalid cache key provided")

            # Retrieve document chunks from cache
            cached_chunks = cache.get(f"doc_chunks_{cache_key}")
            if not cached_chunks:
                logger.error(f"Document chunks not found for cache key: {cache_key}")
                raise ValueError("Please extract the PDF content again to enable search functionality")

            # Recreate documents from cached chunks
            documents = []
            from langchain.schema import Document
            for content, metadata in cached_chunks:
                documents.append(Document(page_content=content, metadata=metadata))

            # Create vector store from documents
            embeddings = embeddings.OpenAIEmbeddings()
            vector_store = vectorstores.FAISS.from_documents(documents, embeddings)

            # Perform similarity search with scores
            logger.debug(f"Performing semantic search with query: {query}")
            results = vector_store.similarity_search_with_score(query, k=k)

            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'score': float(score),  # Convert numpy float to Python float
                    'metadata': doc.metadata
                })

            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise ValueError(f"Error performing semantic search: {str(e)}")

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import vectorstores
from langchain import embeddings