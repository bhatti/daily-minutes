"""
Ollama Service for Local LLM Integration

This service provides integration with Ollama for running LLMs locally.
Supports multiple models and handles connection management.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import httpx
from ollama import AsyncClient, Client
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OllamaConfig(BaseModel):
    """Configuration for Ollama service."""

    host: str = Field(
        default_factory=lambda: __import__('os').getenv('OLLAMA_HOST', 'http://localhost:11434'),
        description="Ollama server host"
    )
    model: str = Field(
        default_factory=lambda: __import__('os').getenv('OLLAMA_MODEL', 'qwen2.5:1.5b'),
        description="Default model (reads from OLLAMA_MODEL env var)"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(4000, gt=0, description="Maximum tokens (increased for comprehensive output)")
    timeout: int = Field(60, gt=0, description="Request timeout in seconds")

    # Model-specific settings
    context_window: int = Field(8192, description="Context window size (increased for qwen2.5)")
    num_ctx: int = Field(4096, description="Number of context tokens (increased)")
    num_predict: int = Field(512, description="Number of tokens to predict (increased for comprehensive output)")
    top_k: int = Field(40, description="Top-k sampling")
    top_p: float = Field(0.9, description="Top-p (nucleus) sampling")

    # Available models
    available_models: List[str] = Field(
        default_factory=lambda: [
            "qwen2.5:7b",        # Qwen 2.5 7B (BEST for comprehensive output)
            "qwen2.5:3b",        # Qwen 2.5 3B (faster, good quality)
            "llama3.2",          # Latest Llama 3.2 (3B)
            "llama3.2:1b",       # Smaller 1B version
            "mistral",           # Mistral 7B
            "phi3",              # Microsoft Phi-3
            "gemma2:2b",         # Google Gemma 2B
            "qwen2.5-coder",     # Code-focused model
            "nomic-embed-text",  # Embedding model
        ],
        description="List of available models"
    )


class OllamaResponse(BaseModel):
    """Response from Ollama API."""

    model: str
    content: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    @property
    def generation_time_ms(self) -> Optional[float]:
        """Get generation time in milliseconds."""
        if self.eval_duration:
            return self.eval_duration / 1_000_000  # Convert nanoseconds to ms
        return None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second."""
        if self.eval_count and self.eval_duration:
            return self.eval_count / (self.eval_duration / 1_000_000_000)
        return None


class OllamaService:
    """Service for interacting with Ollama API."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama service."""
        self.config = config or OllamaConfig()
        self.client = Client(host=self.config.host)
        self.async_client = AsyncClient(host=self.config.host)
        self._is_available = None
        self._available_models = []

        logger.info("ollama_service_initialized",
                   host=self.config.host,
                   model=self.config.model)

    async def check_availability(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.host}/api/tags")
                if response.status_code == 200:
                    self._is_available = True
                    data = response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    logger.info("ollama_available",
                               models_count=len(self._available_models))
                    return True
        except Exception as e:
            logger.warning("ollama_not_available", error=str(e))
            self._is_available = False
        return False

    async def list_models(self) -> List[str]:
        """List available models."""
        if not self._available_models:
            await self.check_availability()
        return self._available_models

    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull a model from Ollama registry with progress tracking.

        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback function(status, progress_pct, total, completed)
        """
        try:
            logger.info("pulling_model", model=model_name)

            # Check if model already exists
            current_models = await self.list_models()
            if model_name in current_models or any(model_name in m for m in current_models):
                logger.info("model_already_exists", model=model_name)
                return True

            # Pull model - ollama library doesn't support streaming properly in async
            # So we'll use a simple pull without progress for now
            await self.async_client.pull(model_name)

            # Refresh available models
            await self.check_availability()

            logger.info("model_pulled", model=model_name)
            return True

        except Exception as e:
            logger.error("model_pull_failed", model=model_name, error=str(e))
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """Generate text using Ollama."""

        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        # Check if model is available
        if model not in await self.list_models():
            logger.warning("model_not_available", model=model)
            # Try to pull the model
            if not await self.pull_model(model):
                raise ValueError(f"Model {model} is not available and couldn't be pulled")

        try:
            # Prepare options
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            }

            # Generate response
            start_time = datetime.now()

            response = await self.async_client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
                stream=False
            )

            generation_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info("generation_complete",
                       model=model,
                       prompt_length=len(prompt),
                       response_length=len(response['response']),
                       time_ms=generation_time)

            return OllamaResponse(
                model=model,
                content=response['response'],
                total_duration=response.get('total_duration'),
                load_duration=response.get('load_duration'),
                prompt_eval_count=response.get('prompt_eval_count'),
                eval_count=response.get('eval_count'),
                eval_duration=response.get('eval_duration')
            )

        except Exception as e:
            logger.error("generation_failed",
                        model=model,
                        error=str(e))
            raise

    async def embed(
        self,
        text: Union[str, List[str]],
        model: str = "nomic-embed-text"
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text."""

        # Ensure embedding model is available
        if model not in await self.list_models():
            logger.info("pulling_embedding_model", model=model)
            await self.pull_model(model)

        try:
            # Handle single text or batch
            texts = [text] if isinstance(text, str) else text

            embeddings = []
            for t in texts:
                response = await self.async_client.embeddings(
                    model=model,
                    prompt=t
                )
                embeddings.append(response['embedding'])

            logger.info("embeddings_generated",
                       model=model,
                       count=len(embeddings))

            # Return single embedding or list
            return embeddings[0] if isinstance(text, str) else embeddings

        except Exception as e:
            logger.error("embedding_failed",
                        model=model,
                        error=str(e))
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> OllamaResponse:
        """Chat with Ollama using conversation history."""

        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        try:
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }

            response = await self.async_client.chat(
                model=model,
                messages=messages,
                options=options
            )

            logger.info("chat_complete",
                       model=model,
                       messages_count=len(messages))

            return OllamaResponse(
                model=model,
                content=response['message']['content'],
                total_duration=response.get('total_duration'),
                eval_count=response.get('eval_count'),
                eval_duration=response.get('eval_duration')
            )

        except Exception as e:
            logger.error("chat_failed",
                        model=model,
                        error=str(e))
            raise

    async def summarize_text(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """Summarize text using Ollama."""

        prompts = {
            "concise": f"Summarize the following text in {max_length} words or less. Be concise and capture the key points:\n\n{text}",
            "bullet": f"Summarize the following text as bullet points (max {max_length} words total):\n\n{text}",
            "technical": f"Provide a technical summary of the following text in {max_length} words or less:\n\n{text}",
            "executive": f"Provide an executive summary of the following text in {max_length} words or less:\n\n{text}"
        }

        prompt = prompts.get(style, prompts["concise"])

        response = await self.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more focused summaries
            max_tokens=max_length * 2  # Approximate token count
        )

        return response.content

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""

        prompt = f"""Analyze the sentiment of the following text.
        Provide a JSON response with:
        - sentiment: positive, negative, or neutral
        - confidence: 0.0 to 1.0
        - key_emotions: list of detected emotions

        Text: {text}

        Response (JSON only):"""

        response = await self.generate(
            prompt=prompt,
            temperature=0.1,  # Very low temperature for consistent format
            max_tokens=200
        )

        try:
            # Parse JSON response
            # Extract JSON from response (might have extra text)
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            # Fallback to simple parsing
            content_lower = response.content.lower()
            if "positive" in content_lower:
                sentiment = "positive"
            elif "negative" in content_lower:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "confidence": 0.7,
                "key_emotions": []
            }

    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """Extract keywords from text."""

        prompt = f"""Extract the {max_keywords} most important keywords from the following text.
        Return only the keywords as a comma-separated list.

        Text: {text}

        Keywords:"""

        response = await self.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=100
        )

        # Parse keywords
        keywords = [k.strip() for k in response.content.split(',')]
        return keywords[:max_keywords]

    async def classify_text(
        self,
        text: str,
        categories: List[str]
    ) -> Dict[str, float]:
        """Classify text into categories."""

        categories_str = ", ".join(categories)
        prompt = f"""Classify the following text into one or more of these categories: {categories_str}

        Provide confidence scores (0.0 to 1.0) for each category.
        Return as JSON format.

        Text: {text}

        Classification (JSON):"""

        response = await self.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=200
        )

        try:
            # Parse JSON response
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)

                # Ensure all categories have scores
                for category in categories:
                    if category not in result:
                        result[category] = 0.0

                return result
        except:
            # Fallback - equal distribution
            return {cat: 1.0/len(categories) for cat in categories}

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "host": self.config.host,
            "default_model": self.config.model,
            "is_available": self._is_available,
            "available_models": self._available_models,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout
            }
        }


# Singleton instance
_ollama_service = None
_last_model_env = None

def get_ollama_service(config: Optional[OllamaConfig] = None) -> OllamaService:
    """Get or create Ollama service instance.

    Recreates service if OLLAMA_MODEL env var changes.
    """
    global _ollama_service, _last_model_env

    import os
    current_model_env = os.getenv('OLLAMA_MODEL')

    # Recreate if env var changed or service doesn't exist
    if _ollama_service is None or _last_model_env != current_model_env:
        _ollama_service = OllamaService(config)
        _last_model_env = current_model_env

    return _ollama_service