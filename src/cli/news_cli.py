#!/usr/bin/env python
"""CLI interface for manually testing news aggregation functionality."""

import asyncio
import json
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from src.connectors.hackernews import HackerNewsConnector
from src.connectors.rss import RSSConnector
from src.agents.news_agent import NewsAgent
from src.models.news import Priority, DataSource
from src.core.logging import get_logger

logger = get_logger(__name__)
console = Console()


@click.group()
def cli():
    """Daily Minutes News CLI - Manual testing interface."""
    pass


@cli.command()
@click.option('--story-type', default='top', type=click.Choice(['top', 'best', 'new', 'ask', 'show', 'job']),
              help='Type of HackerNews stories to fetch')
@click.option('--max-stories', default=10, type=int, help='Maximum number of stories to fetch')
@click.option('--min-score', default=50, type=int, help='Minimum score for stories')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'markdown']),
              help='Output format')
def hackernews(story_type: str, max_stories: int, min_score: int, output_format: str):
    """Fetch and display HackerNews articles."""

    async def fetch():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Fetching {story_type} stories from HackerNews...", total=None)

            connector = HackerNewsConnector(
                story_type=story_type,
                max_stories=max_stories,
                min_score=min_score
            )

            articles = await connector.execute_async()
            progress.update(task, completed=True)

            return articles

    try:
        articles = asyncio.run(fetch())

        if not articles:
            console.print("[yellow]No articles found matching criteria[/yellow]")
            return

        if output_format == 'json':
            # JSON output
            data = [
                {
                    'title': a.title,
                    'url': str(a.url),
                    'author': a.author,
                    'published': a.published_at.isoformat() if a.published_at else None,
                    'score': a.metadata.get('score', 0),
                    'comments': a.metadata.get('comments', 0),
                    'priority': a.priority.value if hasattr(a.priority, 'value') else str(a.priority),
                    'relevance': a.relevance_score,
                    'tags': a.tags
                }
                for a in articles
            ]
            console.print_json(json.dumps(data, indent=2))

        elif output_format == 'markdown':
            # Markdown output
            md_content = f"# HackerNews {story_type.title()} Stories\n\n"
            md_content += f"*Fetched {len(articles)} articles on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"

            for a in articles:
                md_content += f"## [{a.title}]({a.url})\n"
                md_content += f"- **Author**: {a.author}\n"
                md_content += f"- **Score**: {a.metadata.get('score', 0)} | **Comments**: {a.metadata.get('comments', 0)}\n"
                md_content += f"- **Priority**: {a.priority.value if hasattr(a.priority, 'value') else str(a.priority)} | **Relevance**: {a.relevance_score:.2f}\n"
                if a.tags:
                    md_content += f"- **Tags**: {', '.join(a.tags)}\n"
                md_content += "\n"

            console.print(Markdown(md_content))

        else:
            # Table output (default)
            table = Table(title=f"HackerNews {story_type.title()} Stories")
            table.add_column("Title", style="cyan", no_wrap=False)
            table.add_column("Score", justify="right", style="magenta")
            table.add_column("Comments", justify="right", style="blue")
            table.add_column("Priority", style="green")
            table.add_column("Relevance", justify="right", style="yellow")
            table.add_column("Tags", style="dim")

            for article in articles:
                table.add_row(
                    article.title[:80] + ("..." if len(article.title) > 80 else ""),
                    str(article.metadata.get('score', 0)),
                    str(article.metadata.get('comments', 0)),
                    article.priority.value if hasattr(article.priority, 'value') else str(article.priority),
                    f"{article.relevance_score:.2f}",
                    ", ".join(article.tags[:3]) if article.tags else ""
                )

            console.print(table)
            console.print(f"\n[green]Total articles: {len(articles)}[/green]")

    except Exception as e:
        console.print(f"[red]Error fetching HackerNews articles: {e}[/red]")
        logger.error("hackernews_fetch_error", error=str(e))


@cli.command()
@click.option('--max-per-feed', default=5, type=int, help='Maximum articles per RSS feed')
@click.option('--add-feed', multiple=True, help='Add custom RSS feed URL')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'grouped']),
              help='Output format')
def rss(max_per_feed: int, add_feed: tuple, output_format: str):
    """Fetch and display RSS feed articles."""

    async def fetch():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching RSS feeds...", total=None)

            connector = RSSConnector(max_articles_per_feed=max_per_feed)

            # Add custom feeds if provided
            for feed_url in add_feed:
                connector.add_feed(feed_url)

            articles = await connector.fetch_all_feeds()
            progress.update(task, completed=True)

            # Get feed status
            status = connector.get_feed_status()

            return articles, status

    try:
        articles, status = asyncio.run(fetch())

        if not articles:
            console.print("[yellow]No articles found from RSS feeds[/yellow]")
            return

        # Show feed status summary
        console.print(Panel(
            f"[cyan]Feeds Processed:[/cyan] {status['feeds_processed']}\n"
            f"[green]Active Feeds:[/green] {status['active_feeds']}\n"
            f"[yellow]Articles Fetched:[/yellow] {status['articles_fetched']}\n"
            f"[red]Feeds with Errors:[/red] {status['feeds_with_errors']}",
            title="RSS Feed Status",
            border_style="blue"
        ))

        if output_format == 'json':
            # JSON output
            data = [
                {
                    'title': a.title,
                    'url': str(a.url),
                    'source': a.source_name,
                    'published': a.published_at.isoformat() if a.published_at else None,
                    'description': a.description[:200] if a.description else None,
                    'tags': a.tags
                }
                for a in articles
            ]
            console.print_json(json.dumps(data, indent=2))

        elif output_format == 'grouped':
            # Group by source
            by_source = {}
            for article in articles:
                source = article.source_name
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(article)

            for source, source_articles in by_source.items():
                console.print(f"\n[bold cyan]{source}[/bold cyan] ({len(source_articles)} articles)")
                for a in source_articles[:3]:  # Show first 3 from each source
                    console.print(f"  • {a.title[:80]}")
                if len(source_articles) > 3:
                    console.print(f"    [dim]... and {len(source_articles) - 3} more[/dim]")

        else:
            # Table output (default)
            table = Table(title="RSS Feed Articles")
            table.add_column("Source", style="cyan")
            table.add_column("Title", style="white", no_wrap=False)
            table.add_column("Published", style="green")
            table.add_column("Tags", style="dim")

            for article in articles[:20]:  # Limit to 20 for readability
                table.add_row(
                    article.source_name,
                    article.title[:60] + ("..." if len(article.title) > 60 else ""),
                    article.published_at.strftime("%m/%d %H:%M") if article.published_at else "N/A",
                    ", ".join(article.tags[:2]) if article.tags else ""
                )

            console.print(table)
            if len(articles) > 20:
                console.print(f"\n[dim]Showing 20 of {len(articles)} total articles[/dim]")

    except Exception as e:
        console.print(f"[red]Error fetching RSS feeds: {e}[/red]")
        logger.error("rss_fetch_error", error=str(e))


@cli.command()
@click.option('--max-articles', default=20, type=int, help='Maximum total articles to return')
@click.option('--sources', default='all', type=click.Choice(['all', 'hackernews', 'rss']),
              help='News sources to use')
@click.option('--enable-rag', is_flag=True, help='Enable RAG for similar article detection')
@click.option('--enable-rl', is_flag=True, help='Enable reinforcement learning personalization')
def agent(max_articles: int, sources: str, enable_rag: bool, enable_rl: bool):
    """Run the full NewsAgent with orchestration."""

    async def run_agent():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing NewsAgent...", total=None)

            # Create agent with proper configuration
            config = {
                'enable_rag': enable_rag,
                'enable_preferences': enable_rl  # Map enable_rl to enable_preferences
            }

            # Configure connectors based on sources
            if sources == 'hackernews':
                config['hn_connector'] = HackerNewsConnector(max_stories=max_articles)
                config['rss_connector'] = None
            elif sources == 'rss':
                config['hn_connector'] = None
                config['rss_connector'] = RSSConnector(max_articles_per_feed=max_articles // 5)
            else:  # all
                config['hn_connector'] = HackerNewsConnector(max_stories=max_articles // 2)
                config['rss_connector'] = RSSConnector(max_articles_per_feed=max_articles // 10)

            agent = NewsAgent(**config)

            progress.update(task, description="Running news aggregation workflow...")
            articles = await agent.run()

            progress.update(task, description="Generating summary...")
            summary = await agent.generate_summary(articles)

            progress.update(task, completed=True)

            return articles, summary

    try:
        articles, summary = asyncio.run(run_agent())

        # Display summary
        console.print(Panel(
            summary.generate_brief(),
            title="News Summary",
            border_style="green"
        ))

        # Display top articles
        if articles:
            table = Table(title="Top Articles (Ranked by Importance)")
            table.add_column("#", style="dim")
            table.add_column("Title", style="cyan", no_wrap=False)
            table.add_column("Source", style="magenta")
            table.add_column("Priority", style="green")
            table.add_column("Score", justify="right", style="yellow")

            for i, article in enumerate(articles[:10], 1):
                importance = article.calculate_importance()
                table.add_row(
                    str(i),
                    article.title[:60] + ("..." if len(article.title) > 60 else ""),
                    article.source_name,
                    article.priority.value if hasattr(article.priority, 'value') else str(article.priority),
                    f"{importance:.2f}"
                )

            console.print(table)

            # Show workflow state
            console.print(f"\n[green]Workflow completed:[/green] {agent.workflow_state}")
            console.print(f"[blue]Total articles processed:[/blue] {len(articles)}")

    except Exception as e:
        console.print(f"[red]Error running NewsAgent: {e}[/red]")
        logger.error("agent_run_error", error=str(e))


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, type=int, help='Maximum search results')
def search(query: str, limit: int):
    """Search for specific news articles."""

    async def search_news():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Searching for '{query}'...", total=None)

            # Create agent with minimal config for search
            hn_connector = HackerNewsConnector(max_stories=1)
            articles = await hn_connector.search_stories(query, limit=limit)

            progress.update(task, completed=True)
            return articles

    try:
        articles = asyncio.run(search_news())

        if not articles:
            console.print(f"[yellow]No articles found for query: {query}[/yellow]")
            return

        console.print(f"\n[green]Found {len(articles)} articles matching '{query}'[/green]\n")

        for i, article in enumerate(articles, 1):
            console.print(f"[cyan]{i}. {article.title}[/cyan]")
            console.print(f"   [dim]{article.url}[/dim]")
            console.print(f"   Source: {article.source_name} | "
                        f"Published: {article.published_at.strftime('%Y-%m-%d') if article.published_at else 'N/A'}")
            if article.description:
                console.print(f"   [italic]{article.description[:100]}...[/italic]")
            console.print()

    except Exception as e:
        console.print(f"[red]Error searching news: {e}[/red]")
        logger.error("search_error", error=str(e))


@cli.command()
def trending():
    """Get trending topics from HackerNews."""

    async def get_trending():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing trending topics...", total=None)

            # Create connector for trending topics
            hn_connector = HackerNewsConnector(max_stories=50)
            topics = await hn_connector.get_trending_topics()

            progress.update(task, completed=True)
            return topics

    try:
        topics = asyncio.run(get_trending())

        if not topics:
            console.print("[yellow]No trending topics found[/yellow]")
            return

        console.print(Panel(
            "\n".join([f"[cyan]#{i}[/cyan] {topic}" for i, topic in enumerate(topics, 1)]),
            title="Trending Topics",
            border_style="magenta"
        ))

    except Exception as e:
        console.print(f"[red]Error getting trending topics: {e}[/red]")
        logger.error("trending_error", error=str(e))


@cli.command()
def test_connection():
    """Test connectivity to news sources."""

    async def test():
        results = {}

        # Test HackerNews
        try:
            hn = HackerNewsConnector(max_stories=1)
            stories = await hn.fetch_story_ids()
            results['HackerNews'] = {
                'status': 'OK' if stories else 'No data',
                'details': f"Can fetch story IDs: {len(stories) > 0}"
            }
        except Exception as e:
            results['HackerNews'] = {
                'status': 'Error',
                'details': str(e)
            }

        # Test RSS feeds
        try:
            rss = RSSConnector(max_articles_per_feed=1)
            # Just test one feed
            test_feed_url = "https://feeds.arstechnica.com/arstechnica/index"
            articles = await rss.fetch_from_source(test_feed_url)
            results['RSS Feeds'] = {
                'status': 'OK' if articles else 'No data',
                'details': f"Test feed returned {len(articles)} articles"
            }
        except Exception as e:
            results['RSS Feeds'] = {
                'status': 'Error',
                'details': str(e)
            }

        return results

    console.print("[cyan]Testing connectivity to news sources...[/cyan]\n")

    try:
        results = asyncio.run(test())

        table = Table(title="Connection Test Results")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")

        for source, result in results.items():
            status_color = "green" if result['status'] == 'OK' else "red" if result['status'] == 'Error' else "yellow"
            table.add_row(
                source,
                f"[{status_color}]{result['status']}[/{status_color}]",
                result['details']
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error testing connections: {e}[/red]")


if __name__ == '__main__':
    cli()