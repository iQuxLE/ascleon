"""
HPO Agent for accessing Human Phenotype Ontology information.
"""
from typing import Optional, Dict, Any, List

from pydantic_ai import Agent, Tool, System, RunContext

from ascleon.agents.hpo.hpo_config import HPODependencies, get_config
from ascleon.agents.hpo.hpo_tools import (
    get_phenotype_definition,
    search_phenotype_terms,
    get_phenotype_relationships,
    analyze_phenotype_overlap
)


HPO_SYSTEM_PROMPT = """
You are a Human Phenotype Ontology (HPO) specialist agent. 
The HPO provides a standardized vocabulary of phenotypic abnormalities encountered in human disease.

As an HPO agent, you can:
1. Provide detailed definitions and information about HPO terms
2. Search for HPO terms that match descriptions or keywords
3. Retrieve relationships between HPO terms (parents, children, siblings)
4. Analyze sets of phenotypes to identify patterns and relationships

When responding to user queries:
- Always provide HPO IDs in the format HP:nnnnnnn when referring to HPO terms
- Include both the term ID and label/name when discussing HPO terms
- Offer detailed explanations of phenotype characteristics when relevant
- Use clear, concise medical terminology appropriate for the audience
- Help users find the most specific HPO terms that match their descriptions

Your goal is to help users accurately describe phenotypes using standard HPO terminology.
"""


class hpo_agent(Agent[HPODependencies]):
    """HPO agent for retrieving detailed phenotype information from the HPO."""
    
    system = System(HPO_SYSTEM_PROMPT)
    
    tools = [
        Tool(get_phenotype_definition),
        Tool(search_phenotype_terms),
        Tool(get_phenotype_relationships),
        Tool(analyze_phenotype_overlap),
    ]


def run_sync(query: str, deps: Optional[HPODependencies] = None, **kwargs) -> Dict[str, Any]:
    """Run the HPO agent synchronously.
    
    Args:
        query: The user query
        deps: Optional dependencies, will be created if not provided
        **kwargs: Additional arguments to pass to the agent
        
    Returns:
        Dict[str, Any]: The agent response
    """
    # Get config if not provided
    if deps is None:
        deps = get_config()
        
    # Update model if provided
    if "model" in kwargs and kwargs["model"]:
        deps.model = kwargs["model"]
    
    # Override paths if provided
    if "chromadb_path" in kwargs and kwargs["chromadb_path"]:
        deps.chromadb_path = kwargs["chromadb_path"]
    if "collection_name" in kwargs and kwargs["collection_name"]:
        deps.collection_name = kwargs["collection_name"]
    
    # Run the agent
    response = hpo_agent().run(query, deps=deps)
    return response
