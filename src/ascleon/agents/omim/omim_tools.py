"""
Tools for the OMIM agent to retrieve and analyze OMIM disease information.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import quote

import requests
from pydantic_ai import RunContext, ModelRetry

from ascleon.agents.omim.omim_config import OMIMDependencies


# Base URLs for OMIM API
OMIM_API_BASE_URL = "https://api.omim.org/api"
OMIM_ENTRY_BASE_URL = "https://www.omim.org/entry"


async def ensure_cache_dir(ctx: RunContext[OMIMDependencies]) -> str:
    """
    Ensure that the cache directory exists.
    
    Args:
        ctx: The run context with OMIM configuration
        
    Returns:
        str: The path to the cache directory
    """
    # Get path to cache directory
    if ctx.deps.workdir and ctx.deps.workdir.location:
        cache_base = Path(ctx.deps.workdir.location)
    else:
        cache_base = Path(os.getcwd())
    
    cache_dir = cache_base / ctx.deps.cache_dir
    
    # Create directory if it doesn't exist
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    return str(cache_dir)


async def load_cache(ctx: RunContext[OMIMDependencies]) -> Dict[str, Any]:
    """
    Load cached OMIM data from disk.
    
    Args:
        ctx: The run context with OMIM configuration
        
    Returns:
        Dict[str, Any]: The loaded cache
    """
    # Return existing cache if already loaded
    if ctx.deps.cache:
        return ctx.deps.cache
    
    # Ensure cache directory exists
    cache_dir = await ensure_cache_dir(ctx)
    cache_file = Path(cache_dir) / "omim_cache.json"
    
    # Load cache from disk
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                ctx.deps.cache = json.load(f)
        except json.JSONDecodeError:
            ctx.deps.cache = {}
    else:
        ctx.deps.cache = {}
    
    return ctx.deps.cache


async def save_cache(ctx: RunContext[OMIMDependencies]) -> None:
    """
    Save the current cache to disk.
    
    Args:
        ctx: The run context with OMIM configuration
    """
    # Ensure cache is loaded
    if not ctx.deps.cache:
        await load_cache(ctx)
    
    # Ensure cache directory exists
    cache_dir = await ensure_cache_dir(ctx)
    cache_file = Path(cache_dir) / "omim_cache.json"
    
    # Save cache to disk
    with open(cache_file, 'w') as f:
        json.dump(ctx.deps.cache, f, indent=2)


async def make_omim_api_request(
    ctx: RunContext[OMIMDependencies],
    endpoint: str,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Make a request to the OMIM API with appropriate rate limiting.
    
    Args:
        ctx: The run context with OMIM configuration
        endpoint: The API endpoint to request
        params: Additional query parameters
        
    Returns:
        Dict[str, Any]: The API response
    """
    # Ensure API key is set
    if not ctx.deps.api_key:
        raise ModelRetry("OMIM API key is not set. Please set the OMIM_API_KEY environment variable.")
    
    # Build the full URL
    url = f"{OMIM_API_BASE_URL}/{endpoint}"
    
    # Set up the parameters
    if params is None:
        params = {}
    
    # Add authentication
    params["apiKey"] = ctx.deps.api_key
    params["format"] = "json"
    
    # Create cache key for this request
    cache_key = f"api_{endpoint}_{json.dumps(params, sort_keys=True)}"
    
    # Check if in cache
    cache = await load_cache(ctx)
    if cache_key in cache:
        return cache[cache_key]
    
    # Make the API request with rate limiting
    try:
        # Implement basic rate limiting
        delay = 1.0 / ctx.deps.requests_per_second
        time.sleep(delay)
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Cache the result
        cache[cache_key] = data
        await save_cache(ctx)
        
        return data
    except requests.RequestException as e:
        raise ModelRetry(f"OMIM API request failed: {str(e)}")


async def search_omim(
    ctx: RunContext[OMIMDependencies],
    search_term: str,
    search_type: str = "entry"
) -> List[Dict[str, Any]]:
    """
    Search OMIM for a term.
    
    Args:
        ctx: The run context with OMIM configuration
        search_term: The term to search for
        search_type: The type of search (entry, clinical, gene)
        
    Returns:
        List[Dict[str, Any]]: The search results
    """
    # Make the API request
    response = await make_omim_api_request(
        ctx,
        f"search/{search_type}",
        {"search": search_term, "limit": 10}
    )
    
    # Extract the search results
    try:
        results = response["omim"]["searchResponse"]["entryList"]
        return results
    except KeyError:
        return []


async def get_omim_entry(
    ctx: RunContext[OMIMDependencies],
    mim_number: Union[str, int],
    include: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get detailed information for an OMIM entry.
    
    Args:
        ctx: The run context with OMIM configuration
        mim_number: The MIM number to retrieve
        include: Optional sections to include (clinicalSynopsis, geneMap, etc.)
        
    Returns:
        Dict[str, Any]: The entry data
    """
    # Format MIM number
    mim_number = str(mim_number).replace("OMIM:", "").strip()
    
    # Set up include parameters
    params = {}
    if include:
        params["include"] = ",".join(include)
    
    # Make the API request
    response = await make_omim_api_request(
        ctx,
        f"entry/{mim_number}",
        params
    )
    
    # Extract the entry data
    try:
        entry = response["omim"]["entryList"][0]["entry"]
        return entry
    except (KeyError, IndexError):
        raise ModelRetry(f"OMIM entry {mim_number} not found or could not be retrieved")


async def get_clinical_synopsis(
    ctx: RunContext[OMIMDependencies],
    mim_number: Union[str, int]
) -> Dict[str, Any]:
    """
    Get the clinical synopsis for an OMIM entry.
    
    Args:
        ctx: The run context with OMIM configuration
        mim_number: The MIM number to retrieve
        
    Returns:
        Dict[str, Any]: The clinical synopsis data
    """
    # Get the entry with clinical synopsis
    entry = await get_omim_entry(ctx, mim_number, include=["clinicalSynopsis"])
    
    # Extract the clinical synopsis
    try:
        synopsis = entry["clinicalSynopsis"]
        return synopsis
    except KeyError:
        return {}


async def extract_onset_information(
    ctx: RunContext[OMIMDependencies],
    mim_number: Union[str, int]
) -> Dict[str, Any]:
    """
    Extract onset information from an OMIM entry.
    
    This function analyzes the clinical synopsis and text sections
    to find age of onset information for a disease.
    
    Args:
        ctx: The run context with OMIM configuration
        mim_number: The MIM number to retrieve
        
    Returns:
        Dict[str, Any]: Structured onset information
    """
    # Format MIM number for consistency
    mim_number = str(mim_number).replace("OMIM:", "").strip()
    
    # Build cache key
    cache_key = f"onset_{mim_number}"
    
    # Check cache first
    cache = await load_cache(ctx)
    if cache_key in cache:
        return cache[cache_key]
    
    # Get entry with clinical synopsis and text sections
    try:
        entry = await get_omim_entry(
            ctx, 
            mim_number, 
            include=["clinicalSynopsis", "text:description", "text:clinicalFeatures"]
        )
    except ModelRetry:
        # If entry can't be retrieved, return empty result
        return {"onset": None, "onset_description": []}
    
    result = {
        "mim_number": mim_number,
        "title": entry.get("titles", {}).get("preferredTitle", "Unknown"),
        "onset": None,
        "onset_description": []
    }
    
    # Check clinical synopsis for onset section
    if "clinicalSynopsis" in entry:
        synopsis = entry["clinicalSynopsis"]
        if "inheritance" in synopsis:
            result["inheritance"] = synopsis["inheritance"]
        
        # Direct onset field in clinical synopsis
        if "onset" in synopsis:
            result["onset"] = synopsis["onset"]
    
    # If no direct onset field, search in textSections
    if not result["onset"] and "textSections" in entry:
        # Common onset-related terms
        onset_terms = [
            "onset", "begins", "beginning", "starts", "starting",
            "presents", "presenting", "presentation", "manifests",
            "age of", "childhood", "infantile", "juvenile", "adult", 
            "congenital", "neonatal", "perinatal"
        ]
        
        # Age patterns
        age_patterns = [
            r"\b(\d+)\s*-?\s*(\d*)\s*(years?|months?|weeks?|days?|yr)\b",
            r"\bage\s+of\s+(\d+)\b",
            r"\bin\s+(early|middle|late)\s+(childhood|infancy|adulthood)\b",
            r"\b(childhood|infantile|juvenile|adult|congenital|neonatal|perinatal)\s+onset\b"
        ]
        
        # Scan each text section
        for section in entry.get("textSections", []):
            section_content = section.get("content", "")
            
            # Skip empty content
            if not section_content:
                continue
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', section_content)
            
            for sentence in sentences:
                # Check if any onset term is in the sentence
                if any(term in sentence.lower() for term in onset_terms):
                    # Look for age patterns
                    for pattern in age_patterns:
                        matches = re.search(pattern, sentence, re.IGNORECASE)
                        if matches:
                            # Found a potential onset description
                            result["onset_description"].append(sentence.strip())
                            # If no direct onset yet, use the first match as onset
                            if not result["onset"]:
                                result["onset"] = matches.group(0)
                            break
    
    # Try to standardize onset information
    if result["onset"]:
        # Normalize onset to standard categories when possible
        onset_lower = result["onset"].lower()
        if any(term in onset_lower for term in ["birth", "congenital", "neonatal", "perinatal"]):
            result["onset_category"] = "Congenital"
        elif any(term in onset_lower for term in ["infant", "infancy"]):
            result["onset_category"] = "Infantile"
        elif any(term in onset_lower for term in ["child", "childhood"]):
            result["onset_category"] = "Childhood"
        elif any(term in onset_lower for term in ["juvenile", "adolescent", "adolescence", "teen"]):
            result["onset_category"] = "Juvenile"
        elif any(term in onset_lower for term in ["adult", "adulthood"]):
            result["onset_category"] = "Adult"
        else:
            # Try to extract age range
            age_match = re.search(r'(\d+)\s*-?\s*(\d*)\s*(years?|months?|weeks?|days?|yr)', onset_lower)
            if age_match:
                age_start = int(age_match.group(1))
                time_unit = age_match.group(3).lower()
                
                # Convert to years if needed
                if "month" in time_unit:
                    age_years = age_start / 12
                elif "week" in time_unit:
                    age_years = age_start / 52
                elif "day" in time_unit:
                    age_years = age_start / 365
                else:  # years
                    age_years = age_start
                
                # Categorize based on approximate age in years
                if age_years < 1:
                    result["onset_category"] = "Infantile"
                elif age_years < 12:
                    result["onset_category"] = "Childhood"
                elif age_years < 18:
                    result["onset_category"] = "Juvenile"
                else:
                    result["onset_category"] = "Adult"
    
    # Cache the result
    cache[cache_key] = result
    await save_cache(ctx)
    
    return result


async def get_omim_for_disease(
    ctx: RunContext[OMIMDependencies],
    disease_id: str,
    disease_name: str = None
) -> Dict[str, Any]:
    """
    Get comprehensive OMIM information for a disease.
    
    This function tries multiple approaches to find the correct OMIM entry:
    1. Directly using the OMIM ID if provided
    2. Searching by disease name
    3. Extracting OMIM ID from the provided disease ID (e.g., OMIM:123456)
    
    Args:
        ctx: The run context with OMIM configuration
        disease_id: The disease identifier (could be OMIM:123456 or other format)
        disease_name: Optional disease name to use for searching
        
    Returns:
        Dict[str, Any]: Comprehensive OMIM information including onset
    """
    # Check for OMIM ID in the disease_id
    omim_id_match = re.search(r'OMIM:?\s*(\d+)', disease_id)
    mim_number = None
    
    if omim_id_match:
        # Direct OMIM ID found
        mim_number = omim_id_match.group(1)
    elif disease_name:
        # Try to search by name
        search_results = await search_omim(ctx, disease_name)
        if search_results:
            # Use the first result
            entry = search_results[0].get("entry", {})
            mim_number = entry.get("mimNumber")
    
    # If MIM number found, get onset information
    if mim_number:
        onset_info = await extract_onset_information(ctx, mim_number)
        return onset_info
    else:
        # Return empty result if no MIM number found
        return {
            "disease_id": disease_id,
            "disease_name": disease_name,
            "onset": None,
            "onset_description": [],
            "message": "No OMIM entry found for this disease"
        }


async def batch_get_omim_for_diseases(
    ctx: RunContext[OMIMDependencies],
    diseases: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get OMIM information for multiple diseases in batch.
    
    Args:
        ctx: The run context with OMIM configuration
        diseases: List of disease dictionaries with 'disease_id' and optionally 'disease_name'
        
    Returns:
        List[Dict[str, Any]]: OMIM information for each disease
    """
    results = []
    
    for disease in diseases:
        disease_id = disease.get("disease_id") or disease.get("DISEASE_ID")
        disease_name = disease.get("disease_name") or disease.get("DISEASE_NAME")
        
        if not disease_id:
            continue
            
        try:
            # Get OMIM info for this disease
            omim_info = await get_omim_for_disease(ctx, disease_id, disease_name)
            
            # Add original disease info
            omim_info["original_disease"] = disease
            
            results.append(omim_info)
        except Exception as e:
            # Log error and continue with next disease
            print(f"Error retrieving OMIM info for {disease_id}: {str(e)}")
            results.append({
                "disease_id": disease_id,
                "disease_name": disease_name,
                "error": str(e)
            })
    
    return results


async def analyze_onset_compatibility(
    ctx: RunContext[OMIMDependencies],
    patient_onset: Dict[str, Any],
    diseases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze compatibility between patient onset and disease onset information.
    
    Args:
        ctx: The run context with OMIM configuration
        patient_onset: Dictionary with patient onset information (id, label)
        diseases: List of disease dictionaries with DISEASE_ID and DISEASE_NAME
        
    Returns:
        Dict[str, Any]: Compatibility analysis results
    """
    # Standardize patient onset
    patient_onset_category = None
    if patient_onset and "label" in patient_onset:
        onset_label = patient_onset["label"].lower()
        
        # Map to standard categories
        if any(term in onset_label for term in ["birth", "congenital", "neonatal", "perinatal"]):
            patient_onset_category = "Congenital"
        elif any(term in onset_label for term in ["infant", "infancy"]):
            patient_onset_category = "Infantile"
        elif any(term in onset_label for term in ["child", "childhood"]):
            patient_onset_category = "Childhood"
        elif any(term in onset_label for term in ["juvenile", "adolescent", "adolescence", "teen"]):
            patient_onset_category = "Juvenile"
        elif any(term in onset_label for term in ["adult", "adulthood"]):
            patient_onset_category = "Adult"
    
    # Get OMIM information for all diseases
    disease_onset_info = await batch_get_omim_for_diseases(ctx, diseases)
    
    # Analyze compatibility
    results = {
        "patient_onset": patient_onset,
        "patient_onset_category": patient_onset_category,
        "disease_onset_info": disease_onset_info,
        "compatibility_summary": {}
    }
    
    # Calculate compatibility scores if patient onset is available
    if patient_onset_category:
        # Define ordered categories for scoring
        onset_categories = ["Congenital", "Infantile", "Childhood", "Juvenile", "Adult"]
        patient_idx = onset_categories.index(patient_onset_category) if patient_onset_category in onset_categories else -1
        
        if patient_idx >= 0:
            for disease_info in disease_onset_info:
                disease_id = disease_info.get("disease_id") or disease_info.get("original_disease", {}).get("DISEASE_ID")
                disease_onset_category = disease_info.get("onset_category")
                
                if disease_onset_category and disease_onset_category in onset_categories:
                    disease_idx = onset_categories.index(disease_onset_category)
                    
                    # Calculate compatibility
                    if disease_idx == patient_idx:
                        compatibility = "Exact match"
                        score = 1.0
                    elif abs(disease_idx - patient_idx) == 1:
                        compatibility = "Close match"
                        score = 0.7
                    elif abs(disease_idx - patient_idx) == 2:
                        compatibility = "Partial match"
                        score = 0.4
                    else:
                        compatibility = "Poor match"
                        score = 0.1
                        
                    # Special case: if patient has congenital onset, any later onset is poor
                    if patient_onset_category == "Congenital" and disease_idx > 0:
                        compatibility = "Poor match - patient has congenital onset"
                        score = 0.1
                else:
                    compatibility = "Unknown"
                    score = 0.5
                    
                # Store compatibility result
                results["compatibility_summary"][disease_id] = {
                    "disease_onset": disease_info.get("onset"),
                    "disease_onset_category": disease_onset_category,
                    "compatibility": compatibility,
                    "score": score
                }
    
    return results
