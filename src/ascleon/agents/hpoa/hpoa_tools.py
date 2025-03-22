"""
Tools for the HPOA agent to analyze phenotype-disease frequencies.
"""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

from pydantic_ai import RunContext, ModelRetry

from ascleon.agents.hpoa.hpoa_config import HPOADependencies


async def load_hpoa_data(ctx: RunContext[HPOADependencies]) -> Dict[str, Any]:
    """
    Load and parse the HPO Annotation file.
    
    Args:
        ctx: The run context with HPOA configuration
        
    Returns:
        Dict: Structured HPOA data
    """
    # Check if data is already cached
    if ctx.deps.hpoa_data is not None:
        return ctx.deps.hpoa_data
    
    hpoa_path = ctx.deps.hpoa_path
    
    if not os.path.exists(hpoa_path):
        raise ModelRetry(f"HPOA file not found at {hpoa_path}")
    
    # Initialize data structures
    data = {
        "phenotype_to_disease": defaultdict(list),
        "disease_to_phenotype": defaultdict(list),
        "frequency_data": {},
        "onset_data": {}
    }
    
    # Parse HPOA file
    with open(hpoa_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
                
            # Extract key fields
            disease_id = fields[0]      # e.g., OMIM:123456
            disease_name = fields[1]
            hpo_id = fields[3]          # e.g., HP:0001250
            reference = fields[4]       # e.g., PMID:12345678
            evidence = fields[5]        # e.g., PCS
            onset = fields[6]           # e.g., HP:0003577
            frequency = fields[7]       # e.g., HP:0040281 or percentage
            aspect = fields[10] if len(fields) > 10 else ""  # P, I, or C
            
            # Store disease-phenotype associations
            data["phenotype_to_disease"][hpo_id].append({
                "disease_id": disease_id,
                "disease_name": disease_name,
                "onset": onset,
                "frequency": frequency,
                "evidence": evidence,
                "reference": reference,
                "aspect": aspect
            })
            
            # Store phenotype-disease associations
            data["disease_to_phenotype"][disease_id].append({
                "phenotype_id": hpo_id,
                "onset": onset,
                "frequency": frequency,
                "evidence": evidence,
                "reference": reference,
                "aspect": aspect
            })
            
            # Store frequency data for easy lookup
            data["frequency_data"][(disease_id, hpo_id)] = parse_frequency(frequency)
            
            # Store onset data if available
            if onset and onset != "":
                data["onset_data"][(disease_id, hpo_id)] = onset
    
    # Cache the data
    ctx.deps.hpoa_data = data
    
    return data


def parse_frequency(frequency_str: str) -> Dict[str, Any]:
    """
    Parse frequency information into structured data.
    
    Args:
        frequency_str: The frequency string from HPOA
        
    Returns:
        Dict: Parsed frequency data
    """
    # Map HP frequency terms to percentages
    frequency_map = {
        "HP:0040280": {"label": "Obligate", "min": 100, "max": 100},
        "HP:0040281": {"label": "Very frequent", "min": 80, "max": 99},
        "HP:0040282": {"label": "Frequent", "min": 30, "max": 79},
        "HP:0040283": {"label": "Occasional", "min": 5, "max": 29},
        "HP:0040284": {"label": "Very rare", "min": 1, "max": 4},
        "HP:0040285": {"label": "Excluded", "min": 0, "max": 0}
    }
    
    if frequency_str in frequency_map:
        return frequency_map[frequency_str]
    
    # Try to parse percentages like "12%" or "12% - 34%"
    if "%" in frequency_str:
        parts = frequency_str.replace("%", "").split("-")
        if len(parts) == 1:
            try:
                value = float(parts[0].strip())
                return {"min": value, "max": value, "label": f"{value}%"}
            except ValueError:
                pass
        elif len(parts) == 2:
            try:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return {"min": min_val, "max": max_val, "label": f"{min_val}%-{max_val}%"}
            except ValueError:
                pass
    
    return {"label": frequency_str, "min": None, "max": None}


async def get_phenotype_frequency(
    ctx: RunContext[HPOADependencies],
    phenotype_id: str
) -> Dict[str, Dict[str, Any]]:
    """
    Get frequency data for a phenotype across different diseases.
    
    Args:
        ctx: The run context with HPOA configuration
        phenotype_id: The HPO ID to look up
        
    Returns:
        Dict: Frequency data for the phenotype
    """
    # Load HPOA data
    hpoa_data = await load_hpoa_data(ctx)
    
    # Get diseases associated with this phenotype
    diseases = hpoa_data["phenotype_to_disease"].get(phenotype_id, [])
    
    # Collect frequency data
    frequency_data = {}
    for disease in diseases:
        disease_id = disease["disease_id"]
        frequency_str = disease["frequency"]
        
        frequency_data[disease_id] = parse_frequency(frequency_str)
        frequency_data[disease_id]["disease_name"] = disease["disease_name"]
        frequency_data[disease_id]["onset"] = disease["onset"]
        frequency_data[disease_id]["evidence"] = disease["evidence"]
        frequency_data[disease_id]["reference"] = disease["reference"]
    
    return frequency_data


async def get_disease_phenotypes(
    ctx: RunContext[HPOADependencies],
    disease_id: str
) -> List[Dict[str, Any]]:
    """
    Get phenotypes associated with a disease.
    
    Args:
        ctx: The run context with HPOA configuration
        disease_id: The disease ID to look up
        
    Returns:
        List[Dict]: Phenotypes associated with the disease
    """
    # Load HPOA data
    hpoa_data = await load_hpoa_data(ctx)
    
    # Get phenotypes for this disease
    phenotypes = hpoa_data["disease_to_phenotype"].get(disease_id, [])
    
    # Enhance with frequency data
    for pheno in phenotypes:
        pheno_id = pheno["phenotype_id"]
        freq_data = hpoa_data["frequency_data"].get((disease_id, pheno_id), 
                                                  {"label": "Unknown", "min": None, "max": None})
        pheno["frequency_data"] = freq_data
    
    return phenotypes


async def get_phenotype_onset(
    ctx: RunContext[HPOADependencies],
    disease_id: str,
    phenotype_id: str
) -> Optional[str]:
    """
    Get onset information for a phenotype in a disease.
    
    Args:
        ctx: The run context with HPOA configuration
        disease_id: The disease ID
        phenotype_id: The phenotype ID
        
    Returns:
        Optional[str]: The onset HP ID if available
    """
    # Load HPOA data
    hpoa_data = await load_hpoa_data(ctx)
    
    # Get onset information
    return hpoa_data["onset_data"].get((disease_id, phenotype_id))


async def compare_phenotypes_to_disease(
    ctx: RunContext[HPOADependencies],
    disease_id: str,
    phenotype_ids: List[str]
) -> Dict[str, Any]:
    """
    Compare patient phenotypes with disease-associated phenotypes.
    
    Args:
        ctx: The run context with HPOA configuration
        disease_id: The disease ID to check
        phenotype_ids: List of patient's phenotype IDs
        
    Returns:
        Dict: Analysis of phenotype overlap
    """
    # Load HPOA data
    hpoa_data = await load_hpoa_data(ctx)
    
    # Get phenotypes associated with this disease
    disease_phenotypes = hpoa_data["disease_to_phenotype"].get(disease_id, [])
    disease_phenotype_ids = [p["phenotype_id"] for p in disease_phenotypes]
    
    # Find overlapping and missing phenotypes
    overlapping = set(phenotype_ids).intersection(set(disease_phenotype_ids))
    missing_in_patient = set(disease_phenotype_ids) - set(phenotype_ids)
    unexpected_in_patient = set(phenotype_ids) - set(disease_phenotype_ids)
    
    # Get frequency data for overlapping phenotypes
    frequency_data = {}
    for phenotype_id in overlapping:
        frequency_data[phenotype_id] = hpoa_data["frequency_data"].get(
            (disease_id, phenotype_id),
            {"label": "Unknown", "min": None, "max": None}
        )
    
    # Calculate match score
    match_score = len(overlapping) / (len(phenotype_ids) + len(disease_phenotype_ids) - len(overlapping)) if (len(phenotype_ids) + len(disease_phenotype_ids) - len(overlapping)) > 0 else 0
    
    return {
        "disease_id": disease_id,
        "overlapping_phenotypes": list(overlapping),
        "missing_phenotypes": list(missing_in_patient),
        "unexpected_phenotypes": list(unexpected_in_patient),
        "frequency_data": frequency_data,
        "match_score": match_score
    }


async def compare_phenotypes_to_disease_with_exclusions(
    ctx: RunContext[HPOADependencies],
    disease_id: str,
    included_phenotype_ids: List[str],
    excluded_phenotype_ids: List[str]
) -> Dict[str, Any]:
    """
    Compare patient phenotypes (both included and excluded) with disease-associated phenotypes.
    
    Args:
        ctx: The run context with HPOA configuration
        disease_id: The disease ID to check
        included_phenotype_ids: List of patient's observed phenotype IDs
        excluded_phenotype_ids: List of patient's excluded phenotype IDs
        
    Returns:
        Dict: Analysis of phenotype overlap with exclusion data
    """
    # Load HPOA data
    hpoa_data = await load_hpoa_data(ctx)
    
    # Get phenotypes associated with this disease
    disease_phenotypes = hpoa_data["disease_to_phenotype"].get(disease_id, [])
    disease_phenotype_ids = [p["phenotype_id"] for p in disease_phenotypes]
    
    # Find overlapping and missing phenotypes for included ones
    overlapping = set(included_phenotype_ids).intersection(set(disease_phenotype_ids))
    missing_in_patient = set(disease_phenotype_ids) - set(included_phenotype_ids)
    unexpected_in_patient = set(included_phenotype_ids) - set(disease_phenotype_ids)
    
    # Check for conflicts with excluded phenotypes
    # If a disease is expected to show a phenotype that the patient has explicitly excluded,
    # this is a strong signal against that disease
    exclusion_conflicts = set(excluded_phenotype_ids).intersection(set(disease_phenotype_ids))
    
    # Get frequency data for overlapping phenotypes
    frequency_data = {}
    for phenotype_id in overlapping:
        frequency_data[phenotype_id] = hpoa_data["frequency_data"].get(
            (disease_id, phenotype_id),
            {"label": "Unknown", "min": None, "max": None}
        )
    
    # Calculate conflict score - how many excluded phenotypes conflict with disease
    conflict_score = len(exclusion_conflicts) / len(disease_phenotype_ids) if disease_phenotype_ids else 0
    
    # Calculate match score, penalizing for exclusion conflicts
    # More exclusion conflicts = lower score
    match_score = len(overlapping) / (len(included_phenotype_ids) + len(disease_phenotype_ids) - len(overlapping)) if (len(included_phenotype_ids) + len(disease_phenotype_ids) - len(overlapping)) > 0 else 0
    adjusted_score = match_score * (1 - conflict_score)
    
    return {
        "disease_id": disease_id,
        "overlapping_phenotypes": list(overlapping),
        "missing_phenotypes": list(missing_in_patient),
        "unexpected_phenotypes": list(unexpected_in_patient),
        "exclusion_conflicts": list(exclusion_conflicts),
        "frequency_data": frequency_data,
        "match_score": match_score,
        "exclusion_conflict_score": conflict_score,
        "adjusted_score": adjusted_score
    }