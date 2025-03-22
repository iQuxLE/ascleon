"""
MCP wrapper for the exomiser agent.
"""
import os
from typing import Dict, Any, List, Optional

import mcp
from mcp import Config

from ascleon.agents.exomiser.exomiser_tools import (
    list_exomiser_results,
    read_exomiser_result,
    find_matching_phenopacket,
    extract_phenotypes_from_phenopacket,
    rerank_exomiser_results,
    get_result_and_phenopacket,
    perform_reranking
)
from ascleon.agents.exomiser.exomiser_config import ExomiserDependencies
from pydantic_ai import RunContext


class ExomiserAgentContext:
    """Context for the MCP Exomiser agent."""
    
    def __init__(
        self,
        exomiser_results_path: Optional[str] = None,
        phenopackets_path: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the agent context.
        
        Args:
            exomiser_results_path: Optional path to Exomiser results
            phenopackets_path: Optional path to phenopackets
            model: Optional model name to use
        """
        self.deps = ExomiserDependencies()
        if exomiser_results_path:
            self.deps.exomiser_results_path = exomiser_results_path
        if phenopackets_path:
            self.deps.phenopackets_path = phenopackets_path
        if model:
            self.deps.model = model


def create_exomiser_tools() -> List[mcp.Tool]:
    """
    Create MCP tools for the Exomiser agent.
    
    Returns:
        List[mcp.Tool]: List of MCP tools
    """
    return [
        # List available Exomiser result files
        mcp.tools.Function(
            function=lambda ctx, **kwargs: list_exomiser_results(RunContext(deps=ctx.deps)),
            name="list_exomiser_results",
            description="List available Exomiser result files",
            parameters={}
        ),
        
        # Read a specific Exomiser result file
        mcp.tools.Function(
            function=lambda ctx, filename, **kwargs: read_exomiser_result(RunContext(deps=ctx.deps), filename),
            name="read_exomiser_result",
            description="Read an Exomiser result file",
            parameters={
                "filename": {
                    "type": "string",
                    "description": "The name of the Exomiser result file to read"
                }
            }
        ),
        
        # Find matching phenopacket
        mcp.tools.Function(
            function=lambda ctx, exomiser_filename, **kwargs: find_matching_phenopacket(RunContext(deps=ctx.deps), exomiser_filename),
            name="find_matching_phenopacket",
            description="Find the phenopacket that corresponds to the given Exomiser result file",
            parameters={
                "exomiser_filename": {
                    "type": "string",
                    "description": "The name of the Exomiser result file"
                }
            }
        ),
        
        # Extract phenotypes from phenopacket
        mcp.tools.Function(
            function=lambda ctx, phenopacket_data, **kwargs: extract_phenotypes_from_phenopacket(phenopacket_data),
            name="extract_phenotypes_from_phenopacket",
            description="Extract relevant phenotype information from a phenopacket",
            parameters={
                "phenopacket_data": {
                    "type": "object",
                    "description": "The parsed phenopacket data"
                }
            }
        ),
        
        # Rerank Exomiser results
        mcp.tools.Function(
            function=lambda ctx, exomiser_results, phenotype_data, **kwargs: rerank_exomiser_results(
                RunContext(deps=ctx.deps), exomiser_results, phenotype_data
            ),
            name="rerank_exomiser_results",
            description="Use an AI model to rerank Exomiser results based on phenopacket data",
            parameters={
                "exomiser_results": {
                    "type": "array",
                    "description": "The parsed Exomiser results"
                },
                "phenotype_data": {
                    "type": "object",
                    "description": "The structured phenotype data extracted from the phenopacket"
                }
            }
        ),
        
        # Get result and phenopacket data
        mcp.tools.Function(
            function=lambda ctx, exomiser_filename, **kwargs: get_result_and_phenopacket(
                RunContext(deps=ctx.deps), exomiser_filename
            ),
            name="get_result_and_phenopacket",
            description="Get both Exomiser results and matching phenopacket data for analysis",
            parameters={
                "exomiser_filename": {
                    "type": "string",
                    "description": "The name of the Exomiser result file"
                }
            }
        ),
        
        # Complete reranking workflow
        mcp.tools.Function(
            function=lambda ctx, exomiser_filename, **kwargs: perform_reranking(
                RunContext(deps=ctx.deps), exomiser_filename
            ),
            name="perform_reranking",
            description="Complete workflow to rerank Exomiser results based on phenopacket data",
            parameters={
                "exomiser_filename": {
                    "type": "string",
                    "description": "The name of the Exomiser result file"
                }
            }
        )
    ]


def get_system_prompt() -> str:
    """
    Get the system prompt for the Exomiser MCP agent.
    
    Returns:
        str: The system prompt
    """
    return """
You are an AI assistant specialized in analyzing and reranking Exomiser disease gene prioritization results 
based on phenopacket data, particularly focusing on disease onset information.

Exomiser is a tool for prioritizing variants and genes from exome or genome sequencing data, producing TSV files with 
ranked disease candidates. You can help with:

- Listing available Exomiser result files for analysis
- Retrieving and analyzing specific Exomiser result files
- Finding matching phenopacket data for Exomiser results
- Reranking Exomiser results using phenotype and onset information
- Providing explanations for why certain diseases might be more compatible with a patient's presentation

You can use different functions to work with Exomiser results and phenopackets:

- `list_exomiser_results` to see available Exomiser result files
- `read_exomiser_result` to retrieve data from a specific Exomiser file
- `find_matching_phenopacket` to get the corresponding phenopacket
- `perform_reranking` to execute the complete workflow of fetching data and reranking results

When analyzing results, pay special attention to:
1. Disease onset information in the phenopacket
2. The compatibility between onset and each candidate disease
3. The typical age of presentation for each candidate disease
4. The likelihood of the disease manifesting with the specific onset pattern

Your reranking should prioritize diseases that are most consistent with both the phenotypes AND the onset information.
Present your reranked list clearly, explaining why certain diseases were moved up or down in the ranking.
"""


def generate_mcp_config(
    model: Optional[str] = None,
    exomiser_results_path: Optional[str] = None,
    phenopackets_path: Optional[str] = None
) -> Config:
    """
    Generate an MCP config for the Exomiser agent.
    
    Args:
        model: Optional model name to use
        exomiser_results_path: Optional path to Exomiser results
        phenopackets_path: Optional path to phenopackets
        
    Returns:
        Config: The MCP config
    """
    ctx = ExomiserAgentContext(
        exomiser_results_path=exomiser_results_path,
        phenopackets_path=phenopackets_path,
        model=model or "claude-3-opus-20240229"
    )
    
    return mcp.Config(
        tools=create_exomiser_tools(),
        context=ctx,
        system=get_system_prompt(),
        model=model or os.environ.get("EXOMISER_MODEL", "claude-3-opus-20240229")
    )