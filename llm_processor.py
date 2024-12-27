import logging
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from datetime import timedelta
from openai import OpenAI
from app import cache

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMProcessor:
    _client = None
    MODEL_NAME = "gpt-4o-mini"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
    TEMPERATURE = 0.3
    MAX_TOKENS = 300
    CACHE_DURATION = timedelta(hours=24)
    BATCH_SIZE = 10  # Number of questions to process in one API call
    
    @classmethod
    def _get_client(cls) -> OpenAI:
        """Get or create OpenAI client with lazy initialization"""
        if not cls._client:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            cls._client = OpenAI(api_key=api_key)
        return cls._client

    @staticmethod
    def _generate_cache_key(content: str) -> str:
        """Generate a stable cache key for content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    @classmethod
    def _get_cached_result(cls, text: str, question: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available"""
        try:
            cache_key = f"qa_{cls._generate_cache_key(text)}_{cls._generate_cache_key(question)}"
            cached = cache.get(cache_key)
            if cached:
                return json.loads(cached)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        return None

    @classmethod
    def _cache_result(cls, text: str, question: str, result: Dict[str, Any]) -> None:
        """Cache successful result"""
        try:
            if result.get('confidence', 0) >= 0.4:  # Only cache confident results
                cache_key = f"qa_{cls._generate_cache_key(text)}_{cls._generate_cache_key(question)}"
                cache.set(
                    cache_key,
                    json.dumps(result),
                    timeout=int(cls.CACHE_DURATION.total_seconds())
                )
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")

    @classmethod
    def get_relevant_context(cls, text: str, questions: List[str], max_chars: int = 1000000) -> str:
        """Extract relevant context for questions with size limits"""
        try:
            # Normalize and combine questions
            query = " ".join(q.lower() for q in questions)
            
            # Extract meaningful keywords
            stopwords = {'what', 'when', 'where', 'who', 'how', 'why', 'is', 'are', 
                        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            keywords = set(word for word in query.split() 
                         if len(word) > 2 and word not in stopwords)

            # Split text into manageable chunks
            chunks = text.split('\n\n')
            if not chunks:
                return text[:max_chars]

            # Score chunks based on relevance
            scored_chunks = []
            for chunk in chunks:
                chunk_lower = chunk.lower()
                
                # Calculate relevance score
                keyword_score = sum(3 for keyword in keywords if keyword in chunk_lower)
                spec_score = 2 if any(indicator in chunk_lower for indicator in 
                                    ['specifications', 'technical', 'specs']) else 0
                
                if keyword_score + spec_score > 0:
                    scored_chunks.append((keyword_score + spec_score, chunk))

            # Select relevant chunks within size limit
            selected_text = ""
            current_length = 0
            
            for _, chunk in sorted(scored_chunks, reverse=True):
                chunk_len = len(chunk) + 2  # +2 for newlines
                if current_length + chunk_len > max_chars:
                    break
                selected_text += chunk + "\n\n"
                current_length += chunk_len

            return selected_text.strip() if selected_text else text[:max_chars]

        except Exception as e:
            logger.warning(f"Context selection failed: {str(e)}")
            return text[:max_chars]

    @classmethod
    def process_questions(cls, text: str, questions: List[str], min_confidence: float = 0.4) -> List[Dict[str, Any]]:
        """Process multiple questions efficiently with batching and caching"""
        try:
            # Input validation
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid text input")
            if not isinstance(questions, list) or not questions:
                raise ValueError("Questions must be a non-empty list")

            # Initialize results storage
            results = [None] * len(questions)
            uncached_questions = []

            # Check cache first
            for i, question in enumerate(questions):
                if not isinstance(question, str):
                    question = str(question)
                cached = cls._get_cached_result(text, question)
                if cached:
                    results[i] = cached
                else:
                    uncached_questions.append((i, question))

            # Process uncached questions in batches
            if uncached_questions:
                for batch_start in range(0, len(uncached_questions), cls.BATCH_SIZE):
                    batch = uncached_questions[batch_start:batch_start + cls.BATCH_SIZE]
                    
                    # Get relevant context for batch
                    batch_questions = [q for _, q in batch]
                    relevant_text = cls.get_relevant_context(text, batch_questions)
                    
                    # Prepare batch prompt
                    questions_prompt = "\n".join(
                        f"{i+1}. {q}" for i, (_, q) in enumerate(batch)
                    )
                    
                    prompt = (
                        'Analyze the text and answer all questions with specific technical details.\n\n'
                        f'TEXT:\n"""\n{relevant_text}\n"""\n\n'
                        f'QUESTIONS:\n{questions_prompt}\n\n'
                        'Format response as JSON:\n'
                        '{\n  "answers": [\n    {\n'
                        '      "question_index": number,\n'
                        '      "answer": string,\n'
                        '      "confidence": number,\n'
                        '      "context": string\n    }\n  ]\n}'
                    )

                    try:
                        # Get API response
                        response = cls._get_client().chat.completions.create(
                            model=cls.MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a precise document analyzer. Format all responses as JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=cls.TEMPERATURE,
                            max_tokens=cls.MAX_TOKENS * len(batch)
                        )

                        # Parse response
                        content = response.choices[0].message.content
                        parsed = json.loads(content)

                        if not isinstance(parsed.get('answers'), list):
                            raise ValueError("Invalid response format")

                        # Process each answer
                        for answer in parsed['answers']:
                            if not all(k in answer for k in ['question_index', 'answer', 'confidence', 'context']):
                                continue

                            # Map response index to original question index
                            batch_index = answer['question_index'] - 1
                            if batch_index < 0 or batch_index >= len(batch):
                                continue

                            orig_index = batch[batch_index][0]
                            result = {
                                'answer': str(answer['answer']),
                                'confidence': float(min(max(float(answer['confidence']), 0), 1)),
                                'context': str(answer['context'])[:500],
                                'question_index': orig_index,
                                'status': 'success' if float(answer['confidence']) >= min_confidence else 'low_confidence'
                            }

                            # Cache and store result
                            cls._cache_result(text, batch[batch_index][1], result)
                            results[orig_index] = result

                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        # Handle batch failure
                        for orig_index, _ in batch:
                            if results[orig_index] is None:
                                results[orig_index] = {
                                    'answer': "Processing error occurred",
                                    'confidence': 0,
                                    'context': str(e),
                                    'question_index': orig_index,
                                    'status': 'error'
                                }

            # Ensure all questions have results
            return [result if result is not None else {
                'answer': "Failed to process question",
                'confidence': 0,
                'context': "Processing error",
                'question_index': i,
                'status': 'error'
            } for i, result in enumerate(results)]

        except Exception as e:
            logger.error(f"Question processing failed: {str(e)}")
            return [{
                'answer': "Processing error",
                'confidence': 0,
                'context': str(e),
                'question_index': i,
                'status': 'error'
            } for i in range(len(questions))]

    @classmethod
    def get_answer(cls, text: str, question: str) -> Dict[str, Any]:
        """Get answer for a single question"""
        results = cls.process_questions(text, [question])
        return results[0]
