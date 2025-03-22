"""
OMIM Agent for retrieving and analyzing Online Mendelian Inheritance in Man data.
"""
from typing import Optional, Dict, Any, List

from pydantic_ai import Agent, Tool, System, RunContext

from ascleon.agents.omim.omim_config import OMIMDependencies, get_config
from ascleon.agents.omim.omim_tools import (
    search_omim,
    get_omim_entry,
    get_clinical_synopsis,
    extract_onset_information,
    get_omim_for_disease,
    batch_get_omim_for_diseases,
    analyze_onset_compatibility
)


OMIM_SYSTEM_PROMPT = """
You are an OMIM specialist agent with access to the Online Mendelian Inheritance in Man (OMIM) database.

As an OMIM agent, you can:
1. Search for OMIM entries related to genetic disorders
2. Retrieve detailed information about specific OMIM entries
3. Extract clinical information like age of onset for diseases
4. Analyze compatibility between patient onset and disease onset information

When responding to user queries:
- Always provide OMIM IDs when referring to genetic disorders
- Include both the disorder name and MIM number when discussing genetic conditions
- Offer detailed explanations of inheritance patterns when relevant
- Help users find the most accurate onset information for diseases
- Prioritize clear clinical information that can aid in diagnosis

Your goal is to help users leverage OMIM data for accurate disease gene prioritization.
"""


class omim_agent(Agent[OMIMDependencies]):
    """OMIM agent for retrieving and analyzing OMIM disease information."""
    
    system = System(OMIM_SYSTEM_PROMPT)
    
    tools = [
        Tool(search_omim),
        Tool(get_omim_entry),
        Tool(get_clinical_synopsis),
        Tool(extract_onset_information),
        Tool(get_omim_for_disease),
        Tool(batch_get_omim_for_diseases),
        Tool(analyze_onset_compatibility),
    ]


def run_sync(query: str, deps: Optional[OMIMDependencies] = None, **kwargs) -> Dict[str, Any]:
    """Run the OMIM agent synchronously.
    
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
    
    # Override API key if provided
    if "api_key" in kwargs and kwargs["api_key"]:
        deps.api_key = kwargs["api_key"]
    
    # Run the agent
    response = omim_agent().run(query, deps=deps)
    return response
