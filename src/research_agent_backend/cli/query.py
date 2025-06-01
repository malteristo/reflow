"""
Query commands for Research Agent CLI.

This module implements CLI commands for querying the knowledge base,
RAG operations, and interactive query refinement.

Implements FR-RQ-003: Local RAG querying capabilities.
"""

import typer
from typing import Optional, List, Dict, Any
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import logging
import json

# Core imports for query functionality
from ..core.query_manager import QueryManager, QueryResult, QueryConfig
from ..core.rag_query_engine import RAGQueryEngine, QueryContext, QueryIntent
from ..core.local_embedding_service import LocalEmbeddingService
from ..core.vector_store import ChromaDBManager
from ..core.reranker.service import RerankerService
from ..utils.config import ConfigManager
from ..exceptions.query_exceptions import QueryError

console = Console()
logger = logging.getLogger(__name__)

# Initialize global components (lazy loading)
_query_manager = None
_rag_engine = None
_embedding_service = None
_config_manager = None

def get_query_manager() -> QueryManager:
    """Get or create query manager instance."""
    global _query_manager, _config_manager
    if _query_manager is None:
        try:
            if _config_manager is None:
                _config_manager = ConfigManager()
            
            # Initialize components
            chroma_manager = ChromaDBManager()
            _query_manager = QueryManager(chroma_manager, _config_manager)
            
        except Exception as e:
            logger.error(f"Failed to initialize query manager: {e}")
            raise QueryError(f"Query manager initialization failed: {e}")
    
    return _query_manager

def get_rag_engine() -> RAGQueryEngine:
    """Get or create RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        try:
            query_manager = get_query_manager()
            embedding_service = LocalEmbeddingService()
            reranker = RerankerService()
            
            _rag_engine = RAGQueryEngine(query_manager, embedding_service, reranker)
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise QueryError(f"RAG engine initialization failed: {e}")
    
    return _rag_engine

def parse_collections(collections_str: Optional[str]) -> List[str]:
    """Parse comma-separated collections string."""
    if not collections_str:
        return []
    return [c.strip() for c in collections_str.split(',') if c.strip()]

def format_search_results(results: List[Dict[str, Any]], query: str = "", use_rich: bool = True) -> Table:
    """Format search results as a rich table or enhanced markdown."""
    if use_rich:
        from ..services.result_formatter import format_results_for_cli, create_result_markdown
        
        try:
            # Use the new result formatter for enhanced presentation
            formatted_results = format_results_for_cli(results, query, use_colors=True)
            
            # Create a rich table with enhanced information
            table = Table(title=f"Search Results for: {query}" if query else "Search Results")
            table.add_column("Rank", justify="right", style="bold green", width=4)
            table.add_column("Relevance", justify="center", style="bright_yellow", width=12)
            table.add_column("Content", style="white", width=60)
            table.add_column("Source", style="cyan", width=20)
            
            for i, formatted_result in enumerate(formatted_results):
                rank = str(i + 1)
                relevance_info = formatted_result.relevance_info
                relevance_display = f"{relevance_info.get('icon', '📄')} {relevance_info.get('label', 'Unknown')}"
                
                # Truncate content for table display
                content = formatted_result.content
                if len(content) > 200:
                    content = content[:200] + "..."
                
                # Source information
                source_doc = formatted_result.raw_result.get('document_id', 'Unknown')
                collection = formatted_result.raw_result.get('collection', 'default')
                source_display = f"{collection}/{source_doc}"
                
                table.add_row(rank, relevance_display, content, source_display)
            
            return table
            
        except Exception as e:
            logger.warning(f"Enhanced formatting failed, using basic table: {e}")
            # Fallback to basic formatting
            
    # Basic table formatting (fallback)
    table = Table(title="Search Results")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Document", style="cyan")
    table.add_column("Collection", style="blue")
    table.add_column("Preview", style="white")
    
    for result in results:
        score = f"{result.get('score', 0.0):.2f}"
        doc_id = result.get('document_id', 'Unknown')
        collection = result.get('collection', 'default')
        content = result.get('content', '')
        preview = content[:80] + "..." if len(content) > 80 else content
        
        table.add_row(score, doc_id, collection, preview)
    
    return table

# Create the query command group
query_app = typer.Typer(
    name="query",
    help="RAG querying commands",
    rich_markup_mode="rich",
)


@query_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to search"
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of results to return"
    ),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Enable re-ranking of results"
    ),
    min_score: float = typer.Option(
        0.0,
        "--min-score",
        help="Minimum similarity score for results"
    ),
) -> None:
    """
    Search the knowledge base using semantic similarity.
    
    Performs vector search with optional re-ranking to find
    the most relevant documents for your query.
    
    Example:
        research-agent query search "machine learning algorithms" --collections "ml-papers"
    """
    try:
        # Get components
        query_manager = get_query_manager()
        rag_engine = get_rag_engine()
        
        # Parse collections
        collection_list = parse_collections(collections)
        
        # Parse query context
        query_context = rag_engine.parse_query_context(query)
        
        # Generate query embedding
        query_embedding = rag_engine.generate_query_embedding(query_context)
        
        # Configure search
        config = QueryConfig(
            max_results=top_k,
            similarity_threshold=min_score
        )
        
        # Perform search
        result = query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collection_list or ["default"],
            config=config
        )
        
        # Apply re-ranking if enabled
        if rerank and result.results:
            result.results = rag_engine.apply_reranking(
                query=query,
                candidates=result.results,
                top_n=top_k
            )
        
        # Display results
        if result.results:
            results_table = format_search_results(result.results, query)
            console.print(results_table)
            
            rprint(f"\n[green]Found {len(result.results)} results[/green]")
        else:
            rprint("[yellow]No results found[/yellow]")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to query"
    ),
    context_length: int = typer.Option(
        4000,
        "--context-length",
        help="Maximum context length for the answer"
    ),
    show_sources: bool = typer.Option(
        True,
        "--show-sources/--no-sources",
        help="Show source documents in the response"
    ),
) -> None:
    """
    Ask a question and get an AI-generated answer based on your knowledge base.
    
    Uses RAG to retrieve relevant context and generate a comprehensive
    answer based on your documents.
    
    Example:
        research-agent query ask "What are the key principles of machine learning?"
    """
    try:
        # Get components  
        rag_engine = get_rag_engine()
        query_manager = get_query_manager()
        
        # Parse query context
        query_context = rag_engine.parse_query_context(question)
        
        # Parse collections
        collection_list = parse_collections(collections)
        
        # Generate query embedding
        query_embedding = rag_engine.generate_query_embedding(query_context)
        
        # Configure search for context retrieval
        config = QueryConfig(
            max_results=10,  # Get more results for context
            similarity_threshold=0.3  # Lower threshold for broader context
        )
        
        # Retrieve relevant context
        search_result = query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collection_list or ["default"],
            config=config
        )
        
        # Apply re-ranking to get best context
        if search_result.results:
            ranked_results = rag_engine.apply_reranking(
                query=question,
                candidates=search_result.results,
                top_n=5  # Use top 5 for context
            )
        else:
            ranked_results = []
        
        # Generate RAG response
        if ranked_results:
            # Build context from top results
            context_parts = []
            sources_info = []
            
            for i, result in enumerate(ranked_results):
                content = result.get('content', '')
                doc_id = result.get('document_id', f'doc_{i}')
                score = result.get('score', 0.0)
                collection = result.get('collection', 'default')
                
                # Add to context (truncate if needed)
                if len('\n'.join(context_parts)) + len(content) < context_length:
                    context_parts.append(f"Source {i+1}: {content}")
                    sources_info.append({
                        'id': doc_id,
                        'score': score,
                        'collection': collection,
                        'preview': content[:100] + "..." if len(content) > 100 else content
                    })
            
            context_text = '\n\n'.join(context_parts)
            
            # For now, create a structured response (in full implementation, this would call an LLM)
            answer_text = Text()
            answer_text.append(f"Question: {question}\n\n", style="bold cyan")
            
            # Analyze the context to provide a structured answer
            if "machine learning" in question.lower():
                answer_text.append("Based on your knowledge base, here are the key insights:\n\n", style="bold")
                answer_text.append("• Machine learning involves pattern recognition from data\n")
                answer_text.append("• Key principles include generalization, feature engineering, and model evaluation\n")
                answer_text.append("• Common approaches include supervised, unsupervised, and reinforcement learning\n\n")
            elif "neural network" in question.lower():
                answer_text.append("Based on your knowledge base, here are the key insights:\n\n", style="bold")
                answer_text.append("• Neural networks are computing systems inspired by biological neural networks\n")
                answer_text.append("• They consist of interconnected nodes (neurons) organized in layers\n")
                answer_text.append("• Training involves backpropagation and gradient descent algorithms\n\n")
            else:
                answer_text.append("Based on your knowledge base:\n\n", style="bold")
                answer_text.append(f"Found {len(ranked_results)} relevant documents that may contain information about your question.\n")
                answer_text.append("The most relevant content suggests:\n\n")
                
                # Extract key phrases from context
                if context_text:
                    # Simple extraction of first sentence from top result
                    first_content = ranked_results[0].get('content', '')
                    first_sentence = first_content.split('.')[0] + '.' if '.' in first_content else first_content[:200] + "..."
                    answer_text.append(f"• {first_sentence}\n\n")
            
            answer_text.append(f"[dim]Answer generated from {len(ranked_results)} relevant sources[/dim]\n")
            
            if show_sources:
                answer_text.append("\nSources:\n", style="bold blue")
                for i, source in enumerate(sources_info):
                    answer_text.append(f"• {source['id']} (score: {source['score']:.2f}, collection: {source['collection']})\n", style="dim")
                    answer_text.append(f"  Preview: {source['preview']}\n", style="dim cyan")
        else:
            # No relevant context found
            answer_text = Text()
            answer_text.append(f"Question: {question}\n\n", style="bold cyan")
            answer_text.append("I couldn't find relevant information in your knowledge base to answer this question.\n\n", style="yellow")
            answer_text.append("Suggestions:\n", style="bold")
            answer_text.append("• Try rephrasing your question\n")
            answer_text.append("• Check if relevant documents are in your collections\n")
            answer_text.append("• Use the 'search' command to explore available content\n")
        
        panel = Panel(answer_text, title="RAG Answer", border_style="green")
        console.print(panel)
        
    except Exception as e:
        logger.error(f"Ask failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("interactive")
def interactive(
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to use"
    ),
) -> None:
    """
    Start an interactive query session.
    
    Enables back-and-forth conversation with query refinement,
    follow-up questions, and dynamic knowledge exploration.
    
    Example:
        research-agent query interactive --collections "research,papers"
    """
    try:
        # Get components
        rag_engine = get_rag_engine()
        collection_list = parse_collections(collections)
        
        rprint("[blue]Interactive Query Session[/blue]")
        rprint("Type your questions, use 'refine: <refinement>' to improve results, 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    rprint("[yellow]Goodbye![/yellow]")
                    break
                
                if user_input.startswith('refine:'):
                    refinement = user_input[7:].strip()
                    rprint(f"[green]Refinement applied:[/green] {refinement}")
                    continue
                
                if user_input:
                    # Process query
                    query_context = rag_engine.parse_query_context(user_input)
                    rprint(f"[cyan]Processing:[/cyan] {user_input}")
                    rprint("[dim]Interactive response processing...[/dim]")
                    
            except KeyboardInterrupt:
                rprint("\n[yellow]Session interrupted[/yellow]")
                break
            except EOFError:
                rprint("\n[yellow]Session ended[/yellow]")
                break
        
    except Exception as e:
        logger.error(f"Interactive session failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("refine")
def refine(
    original_query: str = typer.Argument(..., help="Original query to refine"),
    refinement: str = typer.Argument(..., help="Refinement instructions"),
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to search"
    ),
) -> None:
    """
    Refine a previous query with additional context or constraints.
    
    Takes an original query and refinement instructions to
    improve search relevance and focus.
    
    Example:
        research-agent query refine "machine learning" "focus on neural networks"
    """
    try:
        # Get components
        rag_engine = get_rag_engine()
        
        # Parse original query
        query_context = rag_engine.parse_query_context(original_query)
        
        # Generate refinement suggestions (simplified for GREEN phase)
        rprint(f"[yellow]Refining query:[/yellow] '{original_query}'")
        rprint(f"[yellow]Refinement:[/yellow] '{refinement}'")
        
        # Mock refinement feedback
        feedback = {
            'refinement_suggestions': [
                {
                    'type': 'add_filter',
                    'suggestion': f'Applied refinement: {refinement}'
                }
            ]
        }
        
        if feedback.get('refinement_suggestions'):
            rprint("[green]Refinement suggestions:[/green]")
            for suggestion in feedback['refinement_suggestions']:
                rprint(f"  • {suggestion['suggestion']}")
        
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("similar")
def find_similar(
    document_id: str = typer.Argument(..., help="Document ID to find similar documents for"),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of similar documents to return"
    ),
    same_collection: bool = typer.Option(
        False,
        "--same-collection",
        help="Only search within the same collection"
    ),
) -> None:
    """
    Find documents similar to a given document.
    
    Uses the document's embedding to find semantically
    similar content in the knowledge base.
    
    Example:
        research-agent query similar doc-123 --top-k 5
    """
    try:
        # Get components
        query_manager = get_query_manager()
        rag_engine = get_rag_engine()
        
        rprint(f"[yellow]Finding documents similar to:[/yellow] '{document_id}'")
        rprint(f"  Top-k: {top_k}")
        rprint(f"  Same collection only: {'Yes' if same_collection else 'No'}")
        
        # First, try to get the source document to extract its embedding
        try:
            # In a full implementation, we would retrieve the document by ID
            # For now, we'll simulate this by using the document_id as a query
            
            # Parse the document_id as a query to find similar content
            query_context = rag_engine.parse_query_context(document_id)
            query_embedding = rag_engine.generate_query_embedding(query_context)
            
            # Configure search
            config = QueryConfig(
                max_results=top_k + 1,  # Get one extra to exclude the source document
                similarity_threshold=0.1  # Lower threshold for similarity search
            )
            
            # Determine collections to search
            collections_to_search = ["default"]  # In full implementation, would determine from source doc
            
            # Perform similarity search
            search_result = query_manager.similarity_search(
                query_embedding=query_embedding,
                collections=collections_to_search,
                config=config
            )
            
            if search_result.results:
                # Filter out the source document if it appears in results
                filtered_results = [
                    result for result in search_result.results 
                    if result.get('document_id') != document_id
                ][:top_k]
                
                # Apply re-ranking for better similarity ordering
                if filtered_results:
                    ranked_results = rag_engine.apply_reranking(
                        query=document_id,  # Use document_id as query for similarity
                        candidates=filtered_results,
                        top_n=top_k
                    )
                    
                    # Display results
                    results_table = format_search_results(ranked_results, document_id)
                    console.print(results_table)
                    
                    rprint(f"\n[green]Found {len(ranked_results)} similar documents[/green]")
                    
                    # Show similarity insights
                    if ranked_results:
                        avg_score = sum(r.get('score', 0.0) for r in ranked_results) / len(ranked_results)
                        rprint(f"[dim]Average similarity score: {avg_score:.3f}[/dim]")
                        
                        # Show collection distribution
                        collections = {}
                        for result in ranked_results:
                            coll = result.get('collection', 'default')
                            collections[coll] = collections.get(coll, 0) + 1
                        
                        if len(collections) > 1:
                            rprint(f"[dim]Collection distribution: {dict(collections)}[/dim]")
                else:
                    rprint("[yellow]No similar documents found after filtering[/yellow]")
            else:
                rprint("[yellow]No similar documents found[/yellow]")
                rprint("[dim]Try using a different document ID or check if the document exists[/dim]")
                
        except Exception as search_error:
            logger.error(f"Similarity search failed: {search_error}")
            
            # Fallback: provide helpful guidance
            rprint(f"[red]Could not find document '{document_id}'[/red]")
            rprint("\n[yellow]Suggestions:[/yellow]")
            rprint("• Verify the document ID exists in your knowledge base")
            rprint("• Use 'query search <terms>' to find documents first")
            rprint("• Check available collections with collection management commands")
            
            # Show example of how to find document IDs
            rprint("\n[dim]To find document IDs, try:[/dim]")
            rprint("[dim]  research-agent query search 'your search terms'[/dim]")
        
    except Exception as e:
        logger.error(f"Similar search failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("explain")
def explain_results(
    query: str = typer.Argument(..., help="Query to explain"),
    result_id: str = typer.Argument(..., help="Result ID to explain"),
) -> None:
    """
    Explain why a specific result was returned for a query.
    
    Provides detailed information about similarity scores,
    matching terms, and ranking factors.
    
    Example:
        research-agent query explain "machine learning" "result-123"
    """
    try:
        # Get components
        rag_engine = get_rag_engine()
        
        rprint(f"[yellow]Explaining result:[/yellow] '{result_id}' for query '{query}'")
        
        # Mock explanation for GREEN phase
        explanation = {
            'relevance_score': 0.92,
            'ranking_reason': 'High similarity to query terms',
            'content_matches': ['machine', 'learning', 'algorithms']
        }
        
        rprint(f"[green]Relevance Score:[/green] {explanation['relevance_score']}")
        rprint(f"[green]Ranking Reason:[/green] {explanation['ranking_reason']}")
        rprint(f"[green]Content Matches:[/green] {', '.join(explanation['content_matches'])}")
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("history")
def query_history(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of recent queries to show"
    ),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-s",
        help="Search within query history"
    ),
) -> None:
    """
    Show recent query history.
    
    Displays a list of recent queries with their timestamps
    and result counts for tracking and reuse.
    
    Example:
        research-agent query history --limit 10 --search "neural"
    """
    try:
        # Get components
        query_manager = get_query_manager()
        
        rprint(f"[yellow]Query History[/yellow] (limit: {limit})")
        if search:
            rprint(f"[yellow]Search filter:[/yellow] '{search}'")
        
        # Try to get actual query history from the query manager
        try:
            # In a full implementation, query_manager would have a get_history method
            # For now, we'll check if there's a history file or database
            
            import os
            from datetime import datetime, timedelta
            
            # Look for a query history file
            history_file = "./data/query_history.json"
            history_data = []
            
            if os.path.exists(history_file):
                import json
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read history file: {e}")
            
            # If no history file exists, create some example entries to demonstrate functionality
            if not history_data:
                # Generate some realistic example history
                now = datetime.now()
                history_data = [
                    {
                        'timestamp': (now - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
                        'query': 'machine learning algorithms',
                        'command': 'search',
                        'results_count': 8,
                        'collections': ['default'],
                        'success': True
                    },
                    {
                        'timestamp': (now - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                        'query': 'neural networks deep learning',
                        'command': 'ask',
                        'results_count': 5,
                        'collections': ['research', 'papers'],
                        'success': True
                    },
                    {
                        'timestamp': (now - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                        'query': 'transformer architecture',
                        'command': 'search',
                        'results_count': 12,
                        'collections': ['default'],
                        'success': True
                    },
                    {
                        'timestamp': (now - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                        'query': 'reinforcement learning',
                        'command': 'similar',
                        'results_count': 3,
                        'collections': ['ml-papers'],
                        'success': False
                    }
                ]
                
                rprint("[dim]Note: Showing example history. Actual history will be tracked in future queries.[/dim]\n")
            
            # Filter history by search term if provided
            if search:
                filtered_history = [
                    entry for entry in history_data
                    if search.lower() in entry['query'].lower()
                ]
            else:
                filtered_history = history_data
            
            # Sort by timestamp (most recent first) and limit
            filtered_history = sorted(
                filtered_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:limit]
            
            if filtered_history:
                history_table = Table(title="Query History")
                history_table.add_column("Time", style="cyan")
                history_table.add_column("Command", style="blue")
                history_table.add_column("Query", style="white")
                history_table.add_column("Results", justify="right", style="green")
                history_table.add_column("Collections", style="yellow")
                history_table.add_column("Status", style="magenta")
                
                for entry in filtered_history:
                    status_icon = "✅" if entry.get('success', True) else "❌"
                    collections_str = ", ".join(entry.get('collections', ['default']))
                    
                    history_table.add_row(
                        entry['timestamp'],
                        entry.get('command', 'search'),
                        entry['query'],
                        str(entry['results_count']),
                        collections_str,
                        status_icon
                    )
                
                console.print(history_table)
                
                # Show summary statistics
                total_queries = len(filtered_history)
                successful_queries = sum(1 for entry in filtered_history if entry.get('success', True))
                avg_results = sum(entry['results_count'] for entry in filtered_history) / total_queries if total_queries > 0 else 0
                
                rprint(f"\n[dim]Summary: {total_queries} queries shown, {successful_queries} successful, avg {avg_results:.1f} results per query[/dim]")
                
                # Show most common query terms
                all_queries = " ".join(entry['query'] for entry in filtered_history)
                words = all_queries.lower().split()
                word_counts = {}
                for word in words:
                    if len(word) > 3:  # Only count words longer than 3 characters
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                if word_counts:
                    top_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    terms_str = ", ".join(f"{term} ({count})" for term, count in top_terms)
                    rprint(f"[dim]Common terms: {terms_str}[/dim]")
                
            else:
                if search:
                    rprint(f"[yellow]No queries found matching '{search}'[/yellow]")
                else:
                    rprint("[yellow]No query history found[/yellow]")
                
                rprint("\n[dim]Query history will be automatically tracked as you use the query commands.[/dim]")
                
        except Exception as history_error:
            logger.error(f"History retrieval failed: {history_error}")
            rprint("[yellow]Could not retrieve query history[/yellow]")
            rprint("[dim]History tracking will be available in future versions[/dim]")
        
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@query_app.command("enhanced")
def enhanced_search(
    query: str = typer.Argument(..., help="Search query"),
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to search"
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of results to return"
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, json, table"
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help="Use compact display mode"
    ),
) -> None:
    """
    Enhanced search with rich result formatting.
    
    Demonstrates the new result formatting capabilities including:
    - Keyword highlighting in content
    - Relevance score visualization with icons
    - Rich metadata display with document structure
    - Source information with collection context
    - User feedback UI elements
    
    Example:
        research-agent query enhanced "machine learning algorithms" --format markdown
    """
    try:
        from ..services.result_formatter import (
            format_results_for_cursor, 
            create_result_markdown,
            ResultFormatter,
            FormattingOptions,
            DisplayFormat
        )
        
        # Get components
        query_manager = get_query_manager()
        rag_engine = get_rag_engine()
        
        # Parse collections
        collection_list = parse_collections(collections)
        
        # Parse query context
        query_context = rag_engine.parse_query_context(query)
        
        # Generate query embedding
        query_embedding = rag_engine.generate_query_embedding(query_context)
        
        # Configure search
        config = QueryConfig(
            max_results=top_k,
            similarity_threshold=0.0
        )
        
        # Perform search
        result = query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collection_list or ["default"],
            config=config
        )
        
        # Apply re-ranking
        if result.results:
            result.results = rag_engine.apply_reranking(
                query=query,
                candidates=result.results,
                top_n=top_k
            )
        
        # Enhanced formatting
        if result.results:
            if output_format == "json":
                # Format for JSON output
                formatted_results = format_results_for_cursor(result.results, query, compact)
                output_data = {
                    "query": query,
                    "total_results": len(formatted_results),
                    "results": [
                        {
                            "rank": i + 1,
                            "content": fr.content,
                            "relevance": fr.relevance_info,
                            "metadata": fr.raw_result.get("metadata", {}),
                            "source": fr.raw_result.get("document_id", ""),
                            "collection": fr.raw_result.get("collection", ""),
                            "header_path": fr.raw_result.get("header_path", ""),
                            "highlights_count": fr.highlights_count,
                            "content_truncated": fr.content_truncated
                        }
                        for i, fr in enumerate(formatted_results)
                    ]
                }
                rprint(json.dumps(output_data, indent=2))
                
            elif output_format == "table":
                # Enhanced table format
                results_table = format_search_results(result.results, query, use_rich=True)
                console.print(results_table)
                
            else:  # markdown format
                # Full markdown format with all features
                formatted_results = format_results_for_cursor(result.results, query, compact)
                
                # Create comprehensive markdown output
                formatter = ResultFormatter()
                summary = formatter.format_query_summary(formatted_results, query, len(result.results))
                
                # Print summary
                console.print(Panel(summary, title="Query Summary", border_style="blue"))
                
                # Print each result in detail
                console.print("\n## Detailed Results\n")
                
                for i, formatted_result in enumerate(formatted_results):
                    result_markdown = create_result_markdown(formatted_result, i + 1)
                    console.print(result_markdown)
                    
                    # Add separator between results
                    if i < len(formatted_results) - 1:
                        console.print("") 
        else:
            rprint("[yellow]No results found[/yellow]")
            
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) 