"""
MCP wrapper for the HPO agent.
"""
import json
import os
from typing import Dict, Any, Optional, List

from pydantic_ai import mcp, RunContext

from ascleon.agents.hpo.hpo_config import HPODependencies, get_config
from ascleon.agents.hpo.hpo_tools import (
    get_phenotype_definition,
    search_phenotype_terms,
    get_phenotype_relationships,
    analyze_phenotype_overlap
)


class HPOMCPTools:
    """MCP-compatible tools for the HPO agent."""
    
    def __init__(self, deps: Optional[HPODependencies] = None):
        """Initialize the HPO MCP tools.
        
        Args:
            deps: Optional dependencies, will be created if not provided
        """
        self.deps = deps or get_config()
        self.context = RunContext(deps=self.deps)

    @mcp.tool("get_hpo_definition")
    async def get_definition(self, phenotype_id: str) -> str:
        """Get the definition and details for an HPO term.
        
        Args:
            phenotype_id: The HPO ID to look up (e.g., HP:0001250)
            
        Returns:
            str: The phenotype definition and details as JSON
        """
        result = await get_phenotype_definition(self.context, phenotype_id)
        return json.dumps(result, indent=2)

    @mcp.tool("search_hpo_terms")
    async def search_terms(self, search_term: str, limit: int = 5) -> str:
        """Search for HPO terms matching a description.
        
        Args:
            search_term: The text to search for in phenotype terms
            limit: Maximum number of results to return (default: 5)
            
        Returns:
            str: The search results as JSON
        """
        result = await search_phenotype_terms(self.context, search_term, limit)
        return json.dumps(result, indent=2)

    @mcp.tool("get_hpo_relationships")
    async def get_relationships(self, phenotype_id: str, relationship_type: str = "parent") -> str:
        """Get related phenotype terms (parents, children, or siblings).
        
        Args:
            phenotype_id: The HPO term ID to look up relationships for
            relationship_type: The type of relationship ("parent", "child", or "sibling")
            
        Returns:
            str: The related terms as JSON
        """
        result = await get_phenotype_relationships(self.context, phenotype_id, relationship_type)
        return json.dumps(result, indent=2)

    @mcp.tool("analyze_phenotypes")
    async def analyze_phenotypes(self, phenotype_ids: List[str]) -> str:
        """Analyze a set of phenotypes to identify patterns and relationships.
        
        Args:
            phenotype_ids: List of HPO IDs to analyze
            
        Returns:
            str: The analysis results as JSON
        """
        result = await analyze_phenotype_overlap(self.context, phenotype_ids)
        return json.dumps(result, indent=2)
