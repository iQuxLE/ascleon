"""
MCP wrapper for the OMIM agent.
"""
import json
import os
from typing import Dict, Any, Optional, List, Union

from pydantic_ai import mcp, RunContext

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


class OmimMCPTools:
    """MCP-compatible tools for the OMIM agent."""
    
    def __init__(self, deps: Optional[OMIMDependencies] = None):
        """Initialize the OMIM MCP tools.
        
        Args:
            deps: Optional dependencies, will be created if not provided
        """
        self.deps = deps or get_config()
        self.context = RunContext(deps=self.deps)

    @mcp.tool("search_omim")
    async def search_omim_mcp(self, search_term: str, search_type: str = "entry") -> str:
        """Search OMIM for a term.
        
        Args:
            search_term: The term to search for
            search_type: The type of search (entry, clinical, gene)
            
        Returns:
            str: The search results as JSON
        """
        result = await search_omim(self.context, search_term, search_type)
        return json.dumps(result, indent=2)

    @mcp.tool("get_omim_entry")
    async def get_omim_entry_mcp(self, mim_number: Union[str, int], include: Optional[List[str]] = None) -> str:
        """Get detailed information for an OMIM entry.
        
        Args:
            mim_number: The MIM number to retrieve
            include: Optional sections to include (clinicalSynopsis, geneMap, etc.)
            
        Returns:
            str: The entry data as JSON
        """
        result = await get_omim_entry(self.context, mim_number, include)
        return json.dumps(result, indent=2)

    @mcp.tool("get_clinical_synopsis")
    async def get_clinical_synopsis_mcp(self, mim_number: Union[str, int]) -> str:
        """Get the clinical synopsis for an OMIM entry.
        
        Args:
            mim_number: The MIM number to retrieve
            
        Returns:
            str: The clinical synopsis data as JSON
        """
        result = await get_clinical_synopsis(self.context, mim_number)
        return json.dumps(result, indent=2)

    @mcp.tool("extract_onset_information")
    async def extract_onset_information_mcp(self, mim_number: Union[str, int]) -> str:
        """Extract onset information from an OMIM entry.
        
        Args:
            mim_number: The MIM number to retrieve
            
        Returns:
            str: The onset information as JSON
        """
        result = await extract_onset_information(self.context, mim_number)
        return json.dumps(result, indent=2)

    @mcp.tool("get_omim_for_disease")
    async def get_omim_for_disease_mcp(self, disease_id: str, disease_name: Optional[str] = None) -> str:
        """Get comprehensive OMIM information for a disease.
        
        Args:
            disease_id: The disease identifier (could be OMIM:123456 or other format)
            disease_name: Optional disease name to use for searching
            
        Returns:
            str: The OMIM information as JSON
        """
        result = await get_omim_for_disease(self.context, disease_id, disease_name)
        return json.dumps(result, indent=2)

    @mcp.tool("analyze_onset_compatibility")
    async def analyze_onset_compatibility_mcp(
        self, 
        patient_onset: Dict[str, Any],
        diseases: List[Dict[str, Any]]
    ) -> str:
        """Analyze compatibility between patient onset and disease onset information.
        
        Args:
            patient_onset: Dictionary with patient onset information (id, label)
            diseases: List of disease dictionaries with DISEASE_ID and DISEASE_NAME
            
        Returns:
            str: The compatibility analysis as JSON
        """
        result = await analyze_onset_compatibility(self.context, patient_onset, diseases)
        return json.dumps(result, indent=2)