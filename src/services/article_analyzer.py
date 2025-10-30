"""ArticleAnalyzer - AI-powered article analysis and key learnings extraction."""

import asyncio
import hashlib
import json
from typing import Dict, List, Any, Optional
from src.core.logging import get_logger
from src.database.sqlite_manager import get_db_manager

logger = get_logger(__name__)


class ArticleAnalyzer:
    """
    Analyzer for extracting insights and key learnings from articles.

    Uses AI (Ollama) to generate:
    - Concise analysis (2-3 sentences about what the article covers)
    - Key learnings/takeaways (3-5 bullet points)
    - Category classification
    - Impact assessment
    """

    def __init__(self):
        """Initialize ArticleAnalyzer."""
        self.sqlite_mgr = get_db_manager()

    def _generate_cache_key(self, url: str) -> str:
        """
        Generate cache key for article analysis.

        Args:
            url: Article URL

        Returns:
            Cache key
        """
        return hashlib.md5(url.encode()).hexdigest()

    def _generate_analysis_prompt(self, title: str, content: Optional[str] = None) -> str:
        """
        Generate prompt for article analysis.

        Args:
            title: Article title
            content: Article content (optional)

        Returns:
            Analysis prompt
        """
        base_text = f"Title: {title}"
        if content:
            # Limit content to avoid token limits
            content_preview = content[:2000] if len(content) > 2000 else content
            base_text += f"\n\nContent: {content_preview}"

        prompt = f"""Analyze the following article:

{base_text}

Provide a comprehensive analysis in this EXACT format:

Analysis: [Write 4-6 DETAILED sentences with specific facts, metrics, and context. MINIMUM 300 characters required. NO meta descriptions. Start with WHO/WHAT, include numbers, explain WHY it matters and WHAT the implications are.]

❌ REJECTED (too short, only 150 chars): "Tailscale releases Peer Relays feature enabling direct mesh connectivity, reducing latency by 40-60ms."
✅ REQUIRED (300+ chars with details): "Tailscale releases Peer Relays feature enabling direct mesh connectivity between nodes without central coordination servers, reducing latency by 40-60ms in cross-region scenarios. This eliminates the overhead of relay servers that previously handled traffic between nodes behind NATs or firewalls. The feature leverages DERP (Designated Encrypted Relay for Packets) protocol fallback only when direct connections fail. This improvement particularly benefits distributed teams working across multiple regions, as it can reduce round-trip times from 120ms to under 60ms. The update represents Tailscale's ongoing commitment to zero-configuration networking while maintaining security through WireGuard encryption."

Key Learnings:
- [Specific fact 1 with numbers/context - minimum 10 words]
- [Specific fact 2 with numbers/context - minimum 10 words]
- [Specific fact 3 with numbers/context - minimum 10 words]
- [At least 3-5 detailed, actionable insights]

TLDR: [ONE sentence, 15-25 words, capturing the core finding. Be specific with numbers/names.]

Category: [AI/Technology/Business/Science/Health/etc.]

Impact: [High/Medium/Low with 1-2 sentence justification]

MANDATORY LENGTH REQUIREMENTS:
- Analysis section: MINIMUM 300 characters (count them!)
- Each key learning: MINIMUM 10 words
- TLDR: EXACTLY 15-25 words
- If Analysis is under 300 chars, ADD MORE DETAILS about impact, context, implications

CRITICAL RULES:
- NO meta phrases: "The article", "appears to", "seems to", "discusses"
- START WITH DIRECT FACTS: company/person names, specific claims
- INCLUDE: numbers, percentages, dates, metrics, comparisons
- EXPLAIN: Why it matters, what changed, who benefits, what's next
"""

        return prompt

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Ollama response into structured format.

        Args:
            response: Raw response from Ollama

        Returns:
            Structured analysis dictionary
        """
        result = {
            'analysis': '',
            'tldr': '',
            'key_learnings': [],
            'category': 'General',
            'impact': 'medium'
        }

        try:
            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Detect sections
                if line.lower().startswith('analysis:'):
                    current_section = 'analysis'
                    result['analysis'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('tldr:'):
                    current_section = 'tldr'
                    result['tldr'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('key learning'):
                    current_section = 'key_learnings'
                elif line.lower().startswith('category:'):
                    current_section = 'category'
                    result['category'] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('impact:'):
                    current_section = 'impact'
                    result['impact'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('-') or line.startswith('•'):
                    # Key learning bullet point
                    if current_section == 'key_learnings':
                        learning = line.lstrip('-•').strip()
                        if learning:
                            result['key_learnings'].append(learning)
                elif current_section == 'analysis' and line:
                    # Continue analysis if multi-line
                    result['analysis'] += ' ' + line
                elif current_section == 'tldr' and line:
                    # Continue TLDR if multi-line
                    result['tldr'] += ' ' + line

            # Ensure we have at least some analysis
            if not result['analysis']:
                result['analysis'] = "Unable to generate detailed analysis from the available content."

            # Ensure we have at least one key learning
            if not result['key_learnings']:
                result['key_learnings'] = ["See full article for detailed information"]

        except Exception as e:
            logger.error("parse_analysis_failed", error=str(e))
            result['analysis'] = f"Error parsing analysis: {str(e)}"

        return result

    async def analyze_article(
        self,
        title: str,
        content: Optional[str] = None,
        url: Optional[str] = None,
        use_cache: bool = True,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze article and extract insights.

        Args:
            title: Article title
            content: Article content (optional)
            url: Article URL for caching
            use_cache: Whether to use cached results
            model: Ollama model to use (defaults to OLLAMA_MODEL env var)

        Returns:
            Dictionary with analysis, key_learnings, category, impact
        """
        # Check cache first
        if use_cache and url:
            cache_key = self._generate_cache_key(url)
            cached = await self.sqlite_mgr.get_cache(cache_key)
            if cached:
                logger.info("analysis_cache_hit", url=url)
                return cached

        try:
            # Import here to avoid circular dependency
            try:
                from src.services.ollama_service import get_ollama_service
                ollama_service = get_ollama_service()
            except ImportError:
                logger.warning("ollama_not_available")
                return {
                    'error': 'AI analysis not available - Ollama service not running',
                    'analysis': 'AI analysis unavailable. Please start Ollama service.',
                    'key_learnings': []
                }

            # Use default model from OllamaService if not specified
            if model is None:
                model = ollama_service.config.model

            # Generate prompt
            prompt = self._generate_analysis_prompt(title, content)

            # Call Ollama
            logger.info("analyzing_article", title=title[:50], model=model)

            # OllamaService.chat() expects messages as List[Dict]
            response = await ollama_service.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model
            )

            # Parse response - OllamaResponse has 'content' field
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict) and 'content' in response:
                response_text = response['content']
            else:
                response_text = str(response)

            result = self._parse_analysis_response(response_text)

            # Cache result for 100 days (analysis doesn't change once generated)
            # Can be cleared manually via Settings if model changes
            if use_cache and url:
                cache_key = self._generate_cache_key(url)
                await self.sqlite_mgr.set_cache(
                    cache_key,
                    result,
                    expires_in_seconds=100 * 24 * 3600  # 100 days
                )

            logger.info("article_analyzed",
                       title=title[:50],
                       learnings_count=len(result.get('key_learnings', [])),
                       category=result.get('category'))

            return result

        except Exception as e:
            logger.error("analysis_failed", error=str(e), title=title[:50])
            return {
                'error': str(e),
                'analysis': f'Analysis failed: {str(e)[:100]}',
                'key_learnings': []
            }

    async def analyze_batch(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        use_cache: bool = True,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple articles in parallel.

        Args:
            articles: List of article dicts with 'title', 'content', 'url'
            max_concurrent: Maximum concurrent analyses
            use_cache: Whether to use cached results
            model: Ollama model to use (defaults to OLLAMA_MODEL env var)

        Returns:
            List of analysis results
        """
        # Get default model from OllamaService if not specified
        if model is None:
            from src.services.ollama_service import get_ollama_service
            model = get_ollama_service().config.model

        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(article_data: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                return await self.analyze_article(
                    title=article_data.get('title', ''),
                    content=article_data.get('content'),
                    url=article_data.get('url'),
                    use_cache=use_cache,
                    model=model
                )

        tasks = [analyze_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'analysis': f'Analysis failed: {str(result)[:100]}',
                    'key_learnings': []
                })
            else:
                processed_results.append(result)

        return processed_results


# Singleton instance
_article_analyzer: Optional[ArticleAnalyzer] = None


def get_article_analyzer() -> ArticleAnalyzer:
    """
    Get or create ArticleAnalyzer instance.

    Returns:
        ArticleAnalyzer instance
    """
    global _article_analyzer
    if _article_analyzer is None:
        _article_analyzer = ArticleAnalyzer()
    return _article_analyzer
