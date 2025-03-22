"""
MCP wrapper for the HPOA agent.
"""
import os
from typing import Dict, List, Any, Optional

import mcp
from mcp import Config

from ascleon.agents.hpoa.hpoa_tools import (
    load_hpoa_data,
    get_phenotype_frequency,
    get_disease_phenotypes,
    get_phenotype_onset,
    compare_phenotypes_to_disease,
    compare_phenotypes_to_disease_with_exclusions
)
from ascleon.agents.hpoa.hpoa_config import HPOADependencies
from pydantic_ai import RunContext


class HPOAAgentContext:
    """Context for the MCP HPOA agent."""
    
    def __init__(
        self,
        hpoa_path: Optional[str] = None
    ):
        """
        Initialize the HPOA agent context.
        
        Args:
            hpoa_path: Optional path to the HPOA file
        """
        self.deps = HPOADependencies()
        if hpoa_path:
            self.deps.hpoa_path = hpoa_path


def create_hpoa_tools() -> List[mcp.Tool]:
    """
    Create MCP tools for the HPOA agent.
    
    Returns:
        List[mcp.Tool]: List of MCP tools
    """
    return [
        # Load HPOA data
        mcp.tools.Function(
            function=lambda ctx, **kwargs: load_hpoa_data(RunContext(deps=ctx.deps)),
            name="load_hpoa_data",
            description="Load and parse the HPO Annotation file",
            parameters={}
        ),
        
        # Get phenotype frequency
        mcp.tools.Function(
            function=lambda ctx, phenotype_id, **kwargs: get_phenotype_frequency(
                RunContext(deps=ctx.deps), phenotype_id
            ),
            name="get_phenotype_frequency",
            description="Get frequency data for a phenotype across different diseases",
            parameters={
                "phenotype_id": {
                    "type": "string",
                    "description": "The HPO ID to look up (e.g., HP:0001250)"
                }
            }
        ),
        
        # Get disease phenotypes
        mcp.tools.Function(
            function=lambda ctx, disease_id, **kwargs: get_disease_phenotypes(
                RunContext(deps=ctx.deps), disease_id
            ),
            name="get_disease_phenotypes",
            description="Get phenotypes associated with a disease",
            parameters={
                "disease_id": {
                    "type": "string",
                    "description": "The disease ID to look up (e.g., OMIM:123456)"
                }
            }
        ),
        
        # Get phenotype onset
        mcp.tools.Function(
            function=lambda ctx, disease_id, phenotype_id, **kwargs: get_phenotype_onset(
                RunContext(deps=ctx.deps), disease_id, phenotype_id
            ),
            name="get_phenotype_onset",
            description="Get onset information for a phenotype in a disease",
            parameters={
                "disease_id": {
                    "type": "string",
                    "description": "The disease ID (e.g., OMIM:123456)"
                },
                "phenotype_id": {
                    "type": "string",
                    "description": "The phenotype ID (e.g., HP:0001250)"
                }
            }
        ),
        
        # Compare phenotypes to disease
        mcp.tools.Function(
            function=lambda ctx, disease_id, phenotype_ids, **kwargs: compare_phenotypes_to_disease(
                RunContext(deps=ctx.deps), disease_id, phenotype_ids
            ),
            name="compare_phenotypes_to_disease",
            description="Compare patient phenotypes with disease-associated phenotypes",
            parameters={
                "disease_id": {
                    "type": "string",
                    "description": "The disease ID to check (e.g., OMIM:123456)"
                },
                "phenotype_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of patient's phenotype IDs (e.g., [HP:0001250, HP:0001251])"
                }
            }
        ),
        
        # Compare phenotypes to disease with exclusions
        mcp.tools.Function(
            function=lambda ctx, disease_id, included_phenotype_ids, excluded_phenotype_ids, **kwargs: 
                compare_phenotypes_to_disease_with_exclusions(
                    RunContext(deps=ctx.deps), disease_id, included_phenotype_ids, excluded_phenotype_ids
                ),
            name="compare_phenotypes_to_disease_with_exclusions",
            description="Compare patient phenotypes (both included and excluded) with disease-associated phenotypes",
            parameters={
                "disease_id": {
                    "type": "string",
                    "description": "The disease ID to check (e.g., OMIM:123456)"
                },
                "included_phenotype_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of patient's observed phenotype IDs (e.g., [HP:0001250, HP:0001251])"
                },
                "excluded_phenotype_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of patient's excluded phenotype IDs (e.g., [HP:0001252, HP:0001253])"
                }
            }
        )
    ]


def get_system_prompt() -> str:
    """
    Get the system prompt for the HPOA MCP agent.
    
    Returns:
        str: The system prompt
    """
    return """
You are an AI assistant that analyzes Human Phenotype Ontology Annotation (HPOA) data.

HPOA contains information about:
- Phenotype-disease associations
- Frequency of phenotypes in diseases (how common a phenotype is within a disease)
- Age of onset for phenotypes in specific diseases
- Evidence codes supporting these associations

You can help with:
- Finding the frequency of a phenotype across different diseases
- Getting all phenotypes associated with a disease
- Comparing a patient's phenotypes with disease-associated phenotypes
- Analyzing how excluded phenotypes impact disease likelihood
- Finding onset information for phenotypes in diseases

When comparing phenotypes to diseases, pay special attention to:
1. Exclusion conflicts - phenotypes that are typically present in a disease but explicitly excluded in the patient
2. Frequency data - how common each phenotype is in each disease
3. Onset information - whether the onset matches expectations for the disease
"""


def generate_mcp_config(
    model: Optional[str] = None,
    hpoa_path: Optional[str] = None
) -> Config:
    """
    Generate an MCP config for the HPOA agent.
    
    Args:
        model: Optional model name to use
        hpoa_path: Optional path to the HPOA file
        
    Returns:
        Config: The MCP config
    """
    ctx = HPOAAgentContext(hpoa_path=hpoa_path)
    
    return mcp.Config(
        tools=create_hpoa_tools(),
        context=ctx,
        system=get_system_prompt(),
        model=model or os.environ.get("HPOA_MODEL", "claude-3-opus-20240229")
    )