"""
RAG (Retrieval-Augmented Generation) Service

This service provides vector storage and semantic search capabilities
using ChromaDB and integrates with Ollama for embeddings.
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.models.news import NewsArticle
from src.services.ollama_service import OllamaService, get_ollama_service

logger = get_logger(__name__)


class RAGConfig(BaseModel):
    """Configuration for RAG service."""

    # ChromaDB settings
    persist_directory: str = Field("./data/chroma", description="ChromaDB persistence directory")
    collection_name: str = Field("daily_minutes", description="Default collection name")

    # Embedding settings
    embedding_model: str = Field("nomic-embed-text", description="Embedding model to use")
    embedding_batch_size: int = Field(10, description="Batch size for embeddings")

    # Search settings
    max_results: int = Field(10, description="Maximum search results")
    min_similarity: float = Field(0.5, description="Minimum similarity score")

    # Context settings
    max_context_length: int = Field(4000, description="Maximum context length in tokens")
    context_window_size: int = Field(3, description="Number of documents to include in context")


class DocumentMetadata(BaseModel):
    """Metadata for stored documents."""

    doc_id: str = Field(..., description="Document ID")
    source: str = Field(..., description="Document source")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    doc_type: str = Field("article", description="Document type")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResult(BaseModel):
    """Search result with document and metadata."""

    doc_id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    embedding: Optional[List[float]] = None


class RAGService:
    """
    RAG service for semantic search and context generation.

    Uses ChromaDB for vector storage and Ollama for embeddings.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        ollama_service: Optional[OllamaService] = None
    ):
        """Initialize RAG service."""
        self.config = config or RAGConfig()
        self.ollama = ollama_service or get_ollama_service()

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        self.collection = self._initialize_collection()

        logger.info("rag_service_initialized",
                   collection=self.config.collection_name,
                   persist_dir=self.config.persist_directory)

    def _initialize_collection(self):
        """Initialize or get ChromaDB collection."""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(
                name=self.config.collection_name
            )
            logger.info("collection_loaded",
                       name=self.config.collection_name,
                       count=collection.count())
        except:
            # Create new collection
            collection = self.chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("collection_created", name=self.config.collection_name)

        return collection

    async def add_article(self, article: NewsArticle) -> str:
        """Add a news article to the vector store."""

        # Generate document ID
        doc_id = self._generate_doc_id(article)

        # Check if already exists
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            logger.debug("article_already_indexed", doc_id=doc_id)
            return doc_id

        # Prepare document content
        content = self._prepare_article_content(article)

        # Generate embedding
        embedding = await self.ollama.embed(
            text=content,
            model=self.config.embedding_model
        )

        # Prepare metadata
        priority = article.priority
        if isinstance(priority, str):
            priority_str = priority
        else:
            priority_str = priority.value if hasattr(priority, 'value') else str(priority)

        metadata = {
            "doc_id": doc_id,
            "source": str(article.source),
            "source_name": article.source_name or "",
            "title": article.title,
            "url": str(article.url),  # Convert HttpUrl to string
            "author": article.author or "",
            "published_at": article.published_at.isoformat() if article.published_at else "",
            "priority": priority_str,
            "relevance_score": article.relevance_score,
            "tags": json.dumps(article.tags),
            "doc_type": "news_article",
            "indexed_at": datetime.now().isoformat()
        }

        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )

        logger.info("article_indexed",
                   doc_id=doc_id,
                   title=article.title[:50])

        return doc_id

    async def add_articles_batch(self, articles: List[NewsArticle]) -> List[str]:
        """Add multiple articles in batch."""

        doc_ids = []
        documents = []
        embeddings = []
        metadatas = []

        for article in articles:
            # Generate document ID
            doc_id = self._generate_doc_id(article)

            # Check if already exists
            existing = self.collection.get(ids=[doc_id])
            if existing['ids']:
                continue

            # Prepare content
            content = self._prepare_article_content(article)

            # Generate embedding
            embedding = await self.ollama.embed(
                text=content,
                model=self.config.embedding_model
            )

            # Prepare metadata
            priority = article.priority
            if isinstance(priority, str):
                priority_str = priority
            else:
                priority_str = priority.value if hasattr(priority, 'value') else str(priority)

            metadata = {
                "doc_id": doc_id,
                "source": str(article.source),
                "source_name": article.source_name or "",
                "title": article.title,
                "url": str(article.url),  # Convert HttpUrl to string
                "author": article.author or "",
                "published_at": article.published_at.isoformat() if article.published_at else "",
                "priority": priority_str,
                "relevance_score": article.relevance_score,
                "tags": json.dumps(article.tags),
                "doc_type": "news_article",
                "indexed_at": datetime.now().isoformat()
            }

            doc_ids.append(doc_id)
            documents.append(content)
            embeddings.append(embedding)
            metadatas.append(metadata)

        if doc_ids:
            # Add batch to collection
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info("articles_batch_indexed", count=len(doc_ids))

        return doc_ids

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""

        max_results = max_results or self.config.max_results

        # Generate query embedding
        query_embedding = await self.ollama.embed(
            text=query,
            model=self.config.embedding_model
        )

        # Prepare filters
        where = {}
        if filter_metadata:
            for key, value in filter_metadata.items():
                where[key] = value

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity from distance (cosine)
            # Distance ranges from 0 to 2, where 0 is identical
            similarity = 1.0 - (results['distances'][0][i] / 2.0)

            if similarity >= self.config.min_similarity:
                search_results.append(SearchResult(
                    doc_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    similarity=similarity
                ))

        logger.info("search_completed",
                   query=query[:50],
                   results=len(search_results))

        return search_results

    async def search_articles(
        self,
        query: str,
        source: Optional[str] = None,
        priority: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """Search specifically for news articles."""

        filters = {"doc_type": "news_article"}

        if source:
            filters["source"] = source

        if priority:
            filters["priority"] = priority

        return await self.search(
            query=query,
            max_results=max_results,
            filter_metadata=filters
        )

    async def get_similar_articles(
        self,
        article: NewsArticle,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Find articles similar to a given article."""

        # Use article content as query
        query = self._prepare_article_content(article)

        # Search for similar articles, excluding the same article
        results = await self.search_articles(
            query=query,
            max_results=max_results + 1  # Get extra in case we need to filter
        )

        # Filter out the same article
        doc_id = self._generate_doc_id(article)
        results = [r for r in results if r.doc_id != doc_id]

        return results[:max_results]

    async def generate_context(
        self,
        query: str,
        max_documents: Optional[int] = None
    ) -> Tuple[str, List[SearchResult]]:
        """
        Generate context for RAG by retrieving relevant documents.

        Returns:
            Tuple of (context_string, search_results)
        """

        max_documents = max_documents or self.config.context_window_size

        # Search for relevant documents
        results = await self.search(
            query=query,
            max_results=max_documents
        )

        if not results:
            return "", []

        # Build context string
        context_parts = []
        for i, result in enumerate(results):
            metadata = result.metadata
            context_parts.append(
                f"[Document {i+1}]\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Source: {metadata.get('source_name', metadata.get('source', 'Unknown'))}\n"
                f"Content: {result.content[:500]}...\n"
            )

        context = "\n---\n".join(context_parts)

        logger.info("context_generated",
                   query=query[:50],
                   documents=len(results))

        return context, results

    async def answer_with_context(
        self,
        question: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG - retrieve context and generate response.

        Returns dictionary with answer and source documents.
        """

        # Generate context
        context, sources = await self.generate_context(question)

        if not context:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "context_used": False
            }

        # Default system prompt for RAG
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Always cite the document number when using information from the context. "
                "If the context doesn't contain relevant information, say so."
            )

        # Build prompt with context
        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided above. Cite document numbers where applicable."""

        # Generate answer using Ollama
        response = await self.ollama.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3,  # Lower temperature for factual responses
            max_tokens=500
        )

        return {
            "answer": response.content,
            "sources": [
                {
                    "title": s.metadata.get("title"),
                    "url": s.metadata.get("url"),
                    "source": s.metadata.get("source_name"),
                    "similarity": s.similarity
                }
                for s in sources
            ],
            "context_used": True,
            "documents_retrieved": len(sources)
        }

    def _generate_doc_id(self, article: NewsArticle) -> str:
        """Generate unique document ID for an article."""
        # Use URL as unique identifier (convert to string first)
        return hashlib.md5(str(article.url).encode()).hexdigest()

    def _prepare_article_content(self, article: NewsArticle) -> str:
        """Prepare article content for embedding."""
        parts = [
            f"Title: {article.title}",
            f"Source: {article.source_name or article.source}"
        ]

        if article.description:
            parts.append(f"Description: {article.description}")

        if article.tags:
            parts.append(f"Tags: {', '.join(article.tags)}")

        return "\n".join(parts)

    async def delete_old_documents(self, days: int = 30):
        """Delete documents older than specified days."""

        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        # Query all documents to check dates
        all_docs = self.collection.get(
            include=["metadatas"]
        )

        ids_to_delete = []
        for i, metadata in enumerate(all_docs['metadatas']):
            indexed_at = metadata.get('indexed_at')
            if indexed_at and indexed_at < cutoff_str:
                ids_to_delete.append(all_docs['ids'][i])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info("old_documents_deleted",
                       count=len(ids_to_delete),
                       cutoff_days=days)

        return len(ids_to_delete)

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        collection_count = self.collection.count()

        # Get sample of metadata to analyze
        sample = self.collection.get(
            limit=100,
            include=["metadatas"]
        )

        sources = {}
        priorities = {}
        doc_types = {}

        for metadata in sample['metadatas']:
            # Count sources
            source = metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

            # Count priorities
            priority = metadata.get('priority', 'unknown')
            priorities[priority] = priorities.get(priority, 0) + 1

            # Count document types
            doc_type = metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            "collection_name": self.config.collection_name,
            "total_documents": collection_count,
            "embedding_model": self.config.embedding_model,
            "sources_distribution": sources,
            "priorities_distribution": priorities,
            "doc_types_distribution": doc_types,
            "config": {
                "max_results": self.config.max_results,
                "min_similarity": self.config.min_similarity,
                "context_window_size": self.config.context_window_size
            }
        }


# Singleton instance
_rag_service = None

def get_rag_service(
    config: Optional[RAGConfig] = None,
    ollama_service: Optional[OllamaService] = None
) -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(config, ollama_service)
    return _rag_service