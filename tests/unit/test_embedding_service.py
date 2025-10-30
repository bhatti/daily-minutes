"""Unit tests for Embedding Service.

Following TDD - write tests first, then implement.
Tests the Ollama-based embedding generation service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.memory.embedding_service import EmbeddingService, EmbeddingResponse


class TestEmbeddingResponse:
    """Test EmbeddingResponse data class."""

    def test_embedding_response_creation(self):
        """Test creating an embedding response."""
        embedding = [0.1, 0.2, 0.3]
        response = EmbeddingResponse(
            embedding=embedding,
            model="nomic-embed-text",
            dimensions=3,
        )

        assert response.embedding == embedding
        assert response.model == "nomic-embed-text"
        assert response.dimensions == 3

    def test_embedding_response_length(self):
        """Test that embedding dimensions match actual length."""
        embedding = [0.1] * 768  # Typical embedding size
        response = EmbeddingResponse(
            embedding=embedding,
            model="nomic-embed-text",
            dimensions=768,
        )

        assert len(response.embedding) == response.dimensions


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization."""

    def test_service_initialization_with_defaults(self):
        """Test creating service with default parameters."""
        service = EmbeddingService()

        assert service.model == "nomic-embed-text"
        assert service.base_url == "http://localhost:11434"
        assert service.embedding_cache is not None

    def test_service_initialization_with_custom_params(self):
        """Test creating service with custom parameters."""
        service = EmbeddingService(
            model="custom-model",
            base_url="http://custom:8080",
        )

        assert service.model == "custom-model"
        assert service.base_url == "http://custom:8080"

    def test_service_has_cache(self):
        """Test that service initializes with cache."""
        service = EmbeddingService()

        assert hasattr(service, 'embedding_cache')
        assert isinstance(service.embedding_cache, dict)


class TestEmbeddingGeneration:
    """Test embedding generation."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_embedding_success(self, MockOllama):
        """Test successful embedding generation."""
        # Mock Ollama service
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()
        result = await service.generate_embedding("Test text")

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.dimensions == 5
        mock_ollama.generate_embedding.assert_called_once_with(
            text="Test text",
            model="nomic-embed-text"
        )

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_embedding_empty_text(self, MockOllama):
        """Test embedding generation with empty text."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embedding("")

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_embedding_whitespace_text(self, MockOllama):
        """Test embedding generation with whitespace-only text."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embedding("   \n\t   ")

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_embedding_long_text(self, MockOllama):
        """Test embedding generation with long text (truncation)."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1] * 768
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()
        long_text = "word " * 10000  # Very long text

        result = await service.generate_embedding(long_text)

        # Should truncate to max tokens
        assert isinstance(result, EmbeddingResponse)
        called_text = mock_ollama.generate_embedding.call_args[1]['text']
        assert len(called_text) <= service.max_tokens * 4  # Rough token estimate

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_embedding_error_handling(self, MockOllama):
        """Test error handling during embedding generation."""
        mock_ollama = MagicMock()
        mock_ollama.generate_embedding = AsyncMock(
            side_effect=Exception("Ollama connection error")
        )
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()

        with pytest.raises(Exception, match="Ollama connection error"):
            await service.generate_embedding("Test text")


class TestBatchEmbeddingGeneration:
    """Test batch embedding generation."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_batch_embeddings_success(self, MockOllama):
        """Test successful batch embedding generation."""
        mock_ollama = MagicMock()

        # Mock different embeddings for each text
        def mock_embed(text, model):
            embeddings_map = {
                "Text 1": [0.1, 0.2],
                "Text 2": [0.3, 0.4],
                "Text 3": [0.5, 0.6],
            }
            response = MagicMock()
            response.embedding = embeddings_map.get(text, [0.0, 0.0])
            return response

        mock_ollama.generate_embedding = AsyncMock(side_effect=mock_embed)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()
        texts = ["Text 1", "Text 2", "Text 3"]
        results = await service.generate_batch_embeddings(texts)

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResponse) for r in results)
        assert results[0].embedding == [0.1, 0.2]
        assert results[1].embedding == [0.3, 0.4]
        assert results[2].embedding == [0.5, 0.6]

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_batch_embeddings_empty_list(self, MockOllama):
        """Test batch embedding with empty list."""
        service = EmbeddingService()

        results = await service.generate_batch_embeddings([])

        assert results == []

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_batch_embeddings_single_item(self, MockOllama):
        """Test batch embedding with single item."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()
        results = await service.generate_batch_embeddings(["Single text"])

        assert len(results) == 1
        assert results[0].embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_generate_batch_embeddings_filters_empty(self, MockOllama):
        """Test batch embedding filters out empty texts."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()
        texts = ["Valid text", "", "  ", "Another valid text"]
        results = await service.generate_batch_embeddings(texts)

        # Should only generate embeddings for valid texts
        assert len(results) == 2
        assert mock_ollama.generate_embedding.call_count == 2


class TestEmbeddingCache:
    """Test embedding caching functionality."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_cache_hit(self, MockOllama):
        """Test that cached embeddings are returned."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()

        # First call - should hit Ollama
        result1 = await service.generate_embedding("Test text")
        assert mock_ollama.generate_embedding.call_count == 1

        # Second call - should use cache
        result2 = await service.generate_embedding("Test text")
        assert mock_ollama.generate_embedding.call_count == 1  # No additional call

        # Results should be identical
        assert result1.embedding == result2.embedding

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_cache_miss_different_text(self, MockOllama):
        """Test that different texts generate new embeddings."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()

        # First text
        await service.generate_embedding("Text 1")
        assert mock_ollama.generate_embedding.call_count == 1

        # Different text - should call Ollama again
        await service.generate_embedding("Text 2")
        assert mock_ollama.generate_embedding.call_count == 2

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_clear_cache(self, MockOllama):
        """Test clearing the embedding cache."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService()

        # Generate and cache
        await service.generate_embedding("Test text")
        assert len(service.embedding_cache) == 1

        # Clear cache
        service.clear_cache()
        assert len(service.embedding_cache) == 0

        # Next call should hit Ollama again
        await service.generate_embedding("Test text")
        assert mock_ollama.generate_embedding.call_count == 2

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_cache_size_limit(self, MockOllama):
        """Test that cache respects size limits."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService(max_cache_size=3)

        # Generate 5 embeddings
        for i in range(5):
            await service.generate_embedding(f"Text {i}")

        # Cache should not exceed max size
        assert len(service.embedding_cache) <= 3


class TestEmbeddingNormalization:
    """Test embedding normalization."""

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_normalize_embedding(self, MockOllama):
        """Test normalizing embeddings to unit length."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        # Non-normalized embedding
        mock_response.embedding = [3.0, 4.0]  # Length = 5
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService(normalize=True)
        result = await service.generate_embedding("Test")

        # Should be normalized to unit length
        import math
        length = math.sqrt(sum(x**2 for x in result.embedding))
        assert abs(length - 1.0) < 0.0001  # Close to 1.0

    @pytest.mark.asyncio
    @patch('src.memory.embedding_service.OllamaService')
    async def test_no_normalization(self, MockOllama):
        """Test embeddings without normalization."""
        mock_ollama = MagicMock()
        mock_response = MagicMock()
        mock_response.embedding = [3.0, 4.0]
        mock_ollama.generate_embedding = AsyncMock(return_value=mock_response)
        MockOllama.return_value = mock_ollama

        service = EmbeddingService(normalize=False)
        result = await service.generate_embedding("Test")

        # Should keep original values
        assert result.embedding == [3.0, 4.0]
