"""
Tools for the exomiser agent to re-rank results based on phenopacket data.
"""
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from pydantic_ai import RunContext, ModelRetry

from ascleon.agents.exomiser.exomiser_config import ExomiserDependencies


async def list_exomiser_results(ctx: RunContext[ExomiserDependencies]) -> List[str]:
    """
    List available Exomiser result files in the configured directory.
    
    Args:
        ctx: The run context with Exomiser configuration
        
    Returns:
        List[str]: List of available Exomiser result filenames
    """
    results_path = Path(ctx.deps.exomiser_results_path)
    
    # Check for pheval_disease_result or pheval_disease_results directories
    pheval_paths = [
        results_path / "pheval_disease_result",
        results_path / "pheval_disease_results"
    ]
    
    # Try each potential pheval directory
    for pheval_path in pheval_paths:
        if pheval_path.exists():
            results = [f.name for f in pheval_path.glob("**/*.tsv") if f.is_file()]
            if results:
                print(f"Found {len(results)} Exomiser result files in {pheval_path}")
                return results
    
    # If no pheval directories found or they're empty, look in base directory
    all_files = [f.name for f in results_path.glob("**/*.tsv") if f.is_file()]
    if all_files:
        print(f"Found {len(all_files)} Exomiser result files in {results_path}")
        return all_files
    
    # If still nothing found, raise error
    raise ModelRetry(f"No Exomiser result files found in {results_path} or any pheval_disease_result* subdirectories")


async def read_exomiser_result(
    ctx: RunContext[ExomiserDependencies],
    filename: str
) -> List[Dict[str, Any]]:
    """
    Read an Exomiser result file and return its contents as structured data.
    
    Args:
        ctx: The run context with Exomiser configuration
        filename: The name of the Exomiser result file to read
        
    Returns:
        List[Dict[str, Any]]: The parsed Exomiser results
    """
    results_path = Path(ctx.deps.exomiser_results_path)
    
    # Try different possible locations for the file
    possible_locations = [
        results_path / filename,  # Main directory
        results_path / "pheval_disease_result" / filename,  # pheval_disease_result subdirectory
        results_path / "pheval_disease_results" / filename,  # pheval_disease_results subdirectory
    ]
    
    # Also try searching recursively if not found in the common locations
    file_path = None
    for location in possible_locations:
        if location.exists():
            file_path = location
            break
    
    # If still not found, do a recursive search
    if not file_path:
        matching_files = list(results_path.glob(f"**/{filename}"))
        if matching_files:
            file_path = matching_files[0]
    
    if not file_path or not file_path.exists():
        raise ModelRetry(f"Could not find Exomiser result file: {filename}")
    
    try:
        results = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                results.append(dict(row))
        
        if not results:
            raise ModelRetry(f"Exomiser result file {filename} is empty")
        
        return results
    except Exception as e:
        raise ModelRetry(f"Error reading Exomiser result file {filename}: {str(e)}")


async def find_matching_phenopacket(
    ctx: RunContext[ExomiserDependencies], 
    exomiser_filename: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Find the phenopacket that corresponds to the given Exomiser result file.
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_filename: The name of the Exomiser result file
        
    Returns:
        Tuple[str, Dict[str, Any]]: The phenopacket filename and parsed content
    """
    phenopackets_path = Path(ctx.deps.phenopackets_path)
    
    # Check for potential subdirectories like 5084_phenopackets
    subdirs = [d for d in phenopackets_path.glob("*") if d.is_dir()]
    search_paths = [phenopackets_path] + subdirs  # Search in base dir and all subdirs
    
    # Extract the base name from the Exomiser filename (remove extension)
    base_name = os.path.splitext(exomiser_filename)[0]
    matching_files = []
    
    # Try multiple search strategies in all potential paths
    for search_path in search_paths:
        print(f"Searching for phenopackets in {search_path}")
        
        # Strategy 1: Direct name match
        matches = list(search_path.glob(f"*{base_name}*.json"))
        if matches:
            matching_files = matches
            break
            
        # Strategy 2: Partial name match with parts
        if not matching_files:
            base_parts = base_name.split('_')
            for part in base_parts:
                if len(part) > 3:  # Only use parts with meaningful length
                    matches = list(search_path.glob(f"*{part}*.json"))
                    if matches:
                        matching_files = matches
                        break
            
            if matching_files:
                break
    
    # If still not found, do a more aggressive recursive search
    if not matching_files:
        print(f"No matches found in common locations. Doing recursive search.")
        matching_files = list(phenopackets_path.glob("**/*.json"))
        if matching_files:
            # Sort by modification time (newest first) to prioritize
            matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not matching_files:
        raise ModelRetry(f"Could not find a phenopacket matching {exomiser_filename}")
    
    # Use the first matching file
    phenopacket_path = matching_files[0]
    print(f"Found matching phenopacket: {phenopacket_path}")
    
    try:
        with open(phenopacket_path, 'r') as f:
            phenopacket_data = json.load(f)
        
        return str(phenopacket_path.name), phenopacket_data
    except Exception as e:
        raise ModelRetry(f"Error reading phenopacket {phenopacket_path}: {str(e)}")


async def extract_pmid_from_phenopacket(
    phenopacket_data: Dict[str, Any],
    phenopacket_filename: str
) -> Optional[str]:
    """
    Extract a PMID from phenopacket data or filename.
    
    This function attempts to extract a PMID from a phenopacket through several methods:
    1. Look for PMIDs in the metadata (publications)
    2. Look for external references with PMID identifiers
    3. Look at the filename for patterns like "PMID_12345678" or "PMID-12345678"
    
    Args:
        phenopacket_data: The parsed phenopacket data
        phenopacket_filename: The filename of the phenopacket (for pattern matching)
        
    Returns:
        Optional[str]: The extracted PMID if found, None otherwise
    """
    # Method 1: Check metadata for publications
    if "meta_data" in phenopacket_data and "publications" in phenopacket_data["meta_data"]:
        publications = phenopacket_data["meta_data"]["publications"]
        for pub in publications:
            if "id" in pub and pub["id"].upper().startswith("PMID:"):
                pmid = pub["id"].replace("PMID:", "").strip()
                print(f"Found PMID in phenopacket metadata: {pmid}")
                return pmid
    
    # Method 2: Check external references
    if "external_references" in phenopacket_data:
        for ref in phenopacket_data["external_references"]:
            if "id" in ref and ref["id"].upper().startswith("PMID:"):
                pmid = ref["id"].replace("PMID:", "").strip()
                print(f"Found PMID in external references: {pmid}")
                return pmid
    
    # Method 3: Check filename for PMID patterns
    import re
    
    # Look for patterns like PMID_12345678 or PMID-12345678
    pmid_pattern = r"PMID[_-](\d+)"
    match = re.search(pmid_pattern, phenopacket_filename, re.IGNORECASE)
    
    if match:
        pmid = match.group(1)
        print(f"Found PMID in filename: {pmid}")
        return pmid
    
    # If no PMID was found with other methods, try more aggressive pattern matching
    # Look for any 8-digit number that might be a PMID
    digit_pattern = r"^.*?(\d{8}).*$"
    match = re.search(digit_pattern, phenopacket_filename)
    
    if match:
        pmid = match.group(1)
        print(f"Found potential PMID (8-digit number) in filename: {pmid}")
        return pmid
    
    return None

async def extract_phenotypes_from_phenopacket(
    phenopacket_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract comprehensive phenotype information from a phenopacket.
    
    This function parses a phenopacket to extract:
    - Included (observed) phenotypes
    - Excluded phenotypes
    - Onset information
    - Evidence for phenotypes
    - Modifiers on phenotypes
    - Any explanations for exclusions
    
    Args:
        phenopacket_data: The parsed phenopacket data with phenotypic features
        
    Returns:
        Dict[str, Any]: Structured phenotype information including:
          - id: Phenopacket ID
          - included_phenotypes: List of observed phenotypes with their details
          - excluded_phenotypes: List of explicitly excluded phenotypes
          - onset: Onset information if available
    """
    result = {
        "id": phenopacket_data.get("id", "Unknown"),
        "included_phenotypes": [],
        "excluded_phenotypes": [],
        "onset": None
    }
    
    # Extract phenotypic features
    phenotypic_features = phenopacket_data.get("phenotypicFeatures", [])
    
    for feature in phenotypic_features:
        # Extract basic phenotype information
        feature_data = {
            "id": feature.get("type", {}).get("id", "Unknown"),
            "label": feature.get("type", {}).get("label", "Unknown"),
        }
        
        # Check for onset information
        if "onset" in feature:
            onset_data = feature.get("onset", {})
            result["onset"] = {
                "id": onset_data.get("id", "Unknown"),
                "label": onset_data.get("label", "Unknown")
            }
        
        # Check for evidence and add it to the feature
        if "evidence" in feature:
            feature_data["evidence"] = feature["evidence"]
            
        # Check for modifiers and add them
        if "modifiers" in feature:
            feature_data["modifiers"] = feature["modifiers"]
        
        # Add any explanation for exclusion if available
        if "excluded" in feature and feature["excluded"]:
            if "description" in feature:
                feature_data["exclusion_reason"] = feature["description"]
        
        # Add to appropriate list based on excluded status
        if feature.get("excluded", False):
            result["excluded_phenotypes"].append(feature_data)
        else:
            result["included_phenotypes"].append(feature_data)
    
    return result


async def rerank_exomiser_results(
    ctx: RunContext[ExomiserDependencies],
    exomiser_results: List[Dict[str, Any]],
    phenotype_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Let the agent rerank Exomiser results based on phenopacket data.
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_results: The parsed Exomiser results
        phenotype_data: The structured phenotype data extracted from the phenopacket
        
    Returns:
        Dict[str, Any]: Reranked Exomiser results
    """
    # Prepare the data for processing
    top_results = exomiser_results[:10]  # Use top 10 results for reranking
    
    # Format phenotype information
    included_phenotypes_text = ", ".join([f"{p['label']} ({p['id']})" for p in phenotype_data['included_phenotypes']])
    
    # Format onset information
    if phenotype_data.get('onset'):
        onset_text = f"- Disease onset: {phenotype_data['onset']['label']} ({phenotype_data['onset']['id']})"
    else:
        onset_text = "- Disease onset: Not specified"
    
    # Format excluded phenotypes
    if phenotype_data.get('excluded_phenotypes'):
        excluded_phenotypes_list = []
        for p in phenotype_data.get('excluded_phenotypes', []):
            excluded_phenotypes_list.append(f"{p['label']} ({p['id']})")
        excluded_text = "- Excluded phenotypes: " + ", ".join(excluded_phenotypes_list)
    else:
        excluded_text = "- No explicitly excluded phenotypes"
    
    # Format disease candidates
    disease_candidates = [
        {
            "rank": i+1,
            "name": result.get('DISEASE_NAME', 'Unknown'),
            "id": result.get('DISEASE_ID', 'Unknown')
        } 
        for i, result in enumerate(top_results)
    ]
    
    # Create a structured input for the agent to process
    reranking_request = {
        "patient_info": {
            "id": phenotype_data['id'],
            "phenotypes": included_phenotypes_text,
            "onset": onset_text,
            "exclusions": excluded_text
        },
        "disease_candidates": disease_candidates
    }
    
    # Return the structured data for the agent to process
    return reranking_request


async def get_result_and_phenopacket(
    ctx: RunContext[ExomiserDependencies],
    exomiser_filename: str
) -> Dict[str, Any]:
    """
    Get both Exomiser results and matching phenopacket data for analysis.
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_filename: The name of the Exomiser result file
        
    Returns:
        Dict[str, Any]: Combined Exomiser and phenopacket data
    """
    # Read the Exomiser results
    exomiser_results = await read_exomiser_result(ctx, exomiser_filename)
    
    # Find matching phenopacket
    phenopacket_filename, phenopacket_data = await find_matching_phenopacket(ctx, exomiser_filename)
    
    # Extract phenotype data
    phenotype_data = await extract_phenotypes_from_phenopacket(phenopacket_data)
    
    return {
        "exomiser_filename": exomiser_filename,
        "exomiser_results": exomiser_results,
        "phenopacket_filename": phenopacket_filename,
        "phenopacket_data": phenopacket_data,
        "phenotype_data": phenotype_data
    }


async def perform_reranking(
    ctx: RunContext[ExomiserDependencies],
    exomiser_filename: str
) -> Dict[str, Any]:
    """
    Complete workflow to rerank Exomiser results based on phenopacket data.
    
    This function coordinates the basic reranking process:
    1. Gets Exomiser results and matching phenopacket
    2. Extracts phenotype data
    3. Prepares the data for reranking
    
    For more comprehensive analysis with HPOA data, frequency information,
    and excluded phenotypes, use the comprehensive_analysis function.
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_filename: The name of the Exomiser result file to analyze
        
    Returns:
        Dict[str, Any]: Results dictionary containing:
          - exomiser_filename: Original filename analyzed
          - phenopacket_filename: Matching phenopacket filename
          - original_results: Top 10 original Exomiser results
          - phenotype_data: Extracted phenotype information
          - reranking_request: Structured data for reranking
    """
    # Get combined data
    combined_data = await get_result_and_phenopacket(ctx, exomiser_filename)
    
    # Prepare the reranking request
    reranking_request = await rerank_exomiser_results(
        ctx, 
        combined_data["exomiser_results"],
        combined_data["phenotype_data"]
    )
    
    # Return the complete results
    return {
        "exomiser_filename": combined_data["exomiser_filename"],
        "phenopacket_filename": combined_data["phenopacket_filename"],
        "original_results": combined_data["exomiser_results"][:10],
        "phenotype_data": combined_data["phenotype_data"],
        "reranking_request": reranking_request
    }


async def analyze_with_hpoa(
    ctx: RunContext[ExomiserDependencies],
    phenotype_data: Dict[str, Any],
    exomiser_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze phenotype data using the HPOA agent.
    
    This function coordinates with the HPOA agent to:
    1. Get frequency data for included phenotypes
    2. Get frequency data for excluded phenotypes
    3. Analyze disease-phenotype compatibility with a focus on exclusions
    
    Args:
        ctx: The run context with Exomiser configuration
        phenotype_data: Extracted phenotype data from phenopacket
        exomiser_results: Exomiser result entries to analyze
        
    Returns:
        Dict[str, Any]: HPOA analysis results containing:
          - phenotype_frequencies: Frequency data for included phenotypes
          - excluded_phenotype_frequencies: Frequency data for excluded phenotypes
          - disease_analyses: Analysis of disease-phenotype compatibility for top candidates
    """
    # Extract phenotype IDs
    included_phenotype_ids = [p["id"] for p in phenotype_data["included_phenotypes"]]
    excluded_phenotype_ids = [p["id"] for p in phenotype_data["excluded_phenotypes"]]
    
    print(f"Analyzing {len(included_phenotype_ids)} included and {len(excluded_phenotype_ids)} excluded phenotypes with HPOA")
    
    # Get frequency data for included phenotypes
    phenotype_frequencies = {}
    for phenotype_id in included_phenotype_ids:
        try:
            from ascleon.agents.hpoa.hpoa_tools import get_phenotype_frequency
            freq_data = await get_phenotype_frequency(
                RunContext(deps=ctx.deps.hpoa), 
                phenotype_id
            )
            phenotype_frequencies[phenotype_id] = freq_data
        except Exception as e:
            print(f"Error getting frequency data for {phenotype_id}: {str(e)}")
    
    # Get frequency data for excluded phenotypes
    excluded_phenotype_frequencies = {}
    for phenotype_id in excluded_phenotype_ids:
        try:
            from ascleon.agents.hpoa.hpoa_tools import get_phenotype_frequency
            freq_data = await get_phenotype_frequency(
                RunContext(deps=ctx.deps.hpoa), 
                phenotype_id
            )
            excluded_phenotype_frequencies[phenotype_id] = freq_data
        except Exception as e:
            print(f"Error getting frequency data for excluded phenotype {phenotype_id}: {str(e)}")
    
    # Analyze disease-phenotype overlaps for top candidates
    disease_analyses = []
    for result in exomiser_results[:10]:  # Analyze top 10
        disease_id = result.get("DISEASE_ID")
        if disease_id:
            try:
                from ascleon.agents.hpoa.hpoa_tools import compare_phenotypes_to_disease_with_exclusions
                analysis = await compare_phenotypes_to_disease_with_exclusions(
                    RunContext(deps=ctx.deps.hpoa), 
                    disease_id, 
                    included_phenotype_ids,
                    excluded_phenotype_ids
                )
                disease_analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing disease {disease_id}: {str(e)}")
    
    return {
        "phenotype_frequencies": phenotype_frequencies,
        "excluded_phenotype_frequencies": excluded_phenotype_frequencies,
        "disease_analyses": disease_analyses
    }


async def get_enriched_phenotype_definitions(
    ctx: RunContext[ExomiserDependencies],
    phenotype_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Get enriched phenotype definitions using the HPO agent.
    
    This function coordinates with the HPO agent to:
    1. Get detailed definitions for phenotypes
    2. Get semantic relationships between phenotypes
    3. Analyze patterns in the phenotype set
    
    Args:
        ctx: The run context with Exomiser configuration
        phenotype_ids: List of phenotype IDs to get definitions for
        
    Returns:
        Dict[str, Dict[str, Any]]: Enriched phenotype information
    """
    # Skip if HPO agent is not configured
    if not ctx.deps.hpo or not ctx.deps.chromadb_path:
        print("HPO agent not configured, skipping enriched phenotype definitions")
        return {}
    
    enriched_data = {}
    
    # Get definitions for each phenotype
    for phenotype_id in phenotype_ids:
        try:
            from ascleon.agents.hpo.hpo_tools import get_phenotype_definition
            definition = await get_phenotype_definition(
                RunContext(deps=ctx.deps.hpo),
                phenotype_id
            )
            enriched_data[phenotype_id] = definition
        except Exception as e:
            print(f"Error getting enriched definition for {phenotype_id}: {str(e)}")
    
    # Try to analyze phenotype patterns if we have multiple phenotypes
    if len(phenotype_ids) > 1:
        try:
            from ascleon.agents.hpo.hpo_tools import analyze_phenotype_overlap
            overlap_analysis = await analyze_phenotype_overlap(
                RunContext(deps=ctx.deps.hpo),
                phenotype_ids
            )
            enriched_data["__analysis"] = overlap_analysis
        except Exception as e:
            print(f"Error analyzing phenotype patterns: {str(e)}")
    
    return enriched_data


async def analyze_with_omim(
    ctx: RunContext[ExomiserDependencies],
    patient_onset: Dict[str, Any],
    diseases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze disease onset information using the OMIM agent.
    
    This function coordinates with the OMIM agent to:
    1. Get detailed onset information for diseases from OMIM
    2. Analyze compatibility between patient onset and disease onset
    
    Args:
        ctx: The run context with Exomiser configuration
        patient_onset: Onset information from the patient's phenopacket
        diseases: Exomiser result entries to analyze
        
    Returns:
        Dict[str, Any]: OMIM analysis results containing:
          - disease_onset: Detailed onset information for each disease
          - onset_compatibility: Analysis of compatibility with patient onset
    """
    # Skip if OMIM API is not configured or not enabled
    if not ctx.deps.omim or not ctx.deps.omim.api_key or not ctx.deps.use_omim:
        print("OMIM API not configured or not enabled, skipping onset analysis with OMIM")
        return {"disease_onset": {}, "onset_compatibility": {}}
    
    print(f"Analyzing onset compatibility with OMIM for {len(diseases)} diseases")
    
    try:
        # Analyze compatibility between patient onset and diseases
        from ascleon.agents.omim.omim_tools import analyze_onset_compatibility
        compatibility_analysis = await analyze_onset_compatibility(
            RunContext(deps=ctx.deps.omim),
            patient_onset,
            diseases
        )
        
        return {
            "disease_onset": compatibility_analysis["disease_onset_info"],
            "onset_compatibility": compatibility_analysis["compatibility_summary"]
        }
    except Exception as e:
        print(f"Error during OMIM analysis: {str(e)}")
        return {"disease_onset": {}, "onset_compatibility": {}, "error": str(e)}


async def analyze_with_literature(
    ctx: RunContext[ExomiserDependencies],
    phenopacket_data: Dict[str, Any],
    phenopacket_filename: str,
    diseases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze diagnostic test recommendations from literature using the Literature agent.
    
    This function coordinates with the Literature agent to:
    1. Find PMIDs associated with the phenopacket
    2. Retrieve article abstracts or full text
    3. Extract diagnostic test recommendations relevant to the candidate diseases
    
    Args:
        ctx: The run context with Exomiser configuration
        phenopacket_data: The parsed phenopacket data
        phenopacket_filename: Filename of the phenopacket (for PMID extraction)
        diseases: Exomiser result entries to analyze
        
    Returns:
        Dict[str, Any]: Literature analysis results containing:
          - pmid: The PMID used for analysis (if found)
          - test_recommendations: Diagnostic test recommendations extracted from literature
          - disease_specific_tests: Tests especially relevant for top candidate diseases
    """
    # Skip if Literature agent is not enabled
    if not ctx.deps.literature or not ctx.deps.use_literature:
        print("Literature agent not enabled, skipping diagnostic test analysis")
        return {
            "pmid": None, 
            "test_recommendations": [], 
            "disease_specific_tests": {}
        }
    
    # Try to extract PMID from phenopacket
    pmid = await extract_pmid_from_phenopacket(phenopacket_data, phenopacket_filename)
    
    if not pmid:
        print("No PMID found in phenopacket, skipping literature analysis")
        return {
            "pmid": None, 
            "test_recommendations": [], 
            "disease_specific_tests": {}
        }
    
    print(f"Analyzing diagnostic test recommendations from literature using PMID: {pmid}")
    
    try:
        # Get abstract for the article
        from ascleon.agents.literature.literature_tools import get_article_abstract, lookup_pmid
        
        # Try to get full text first, but fall back to abstract if that fails
        try:
            article_text = await lookup_pmid(RunContext(deps=ctx.deps.literature), f"PMID:{pmid}")
            print(f"Retrieved full text for PMID:{pmid}")
        except Exception:
            # If full text fails, try just the abstract
            try:
                article_text = await get_article_abstract(RunContext(deps=ctx.deps.literature), f"PMID:{pmid}")
                print(f"Retrieved abstract for PMID:{pmid}")
            except Exception as e:
                print(f"Error retrieving article for PMID:{pmid}: {str(e)}")
                return {
                    "pmid": pmid,
                    "error": f"Failed to retrieve article: {str(e)}",
                    "test_recommendations": [],
                    "disease_specific_tests": {}
                }
        
        # Prepare disease information
        disease_info = []
        for i, disease in enumerate(diseases[:5]):  # Use top 5 diseases for analysis
            disease_info.append({
                "rank": i + 1,
                "id": disease.get("DISEASE_ID", "Unknown"),
                "name": disease.get("DISEASE_NAME", "Unknown")
            })
        
        # Return the article text and disease info for the agent to process
        return {
            "pmid": pmid,
            "article_text": article_text[:12000],  # Truncate if too long
            "disease_info": disease_info
        }
        
    except Exception as e:
        print(f"Error during literature analysis: {str(e)}")
        return {
            "pmid": pmid,
            "error": str(e),
            "test_recommendations": [],
            "disease_specific_tests": {}
        }


async def comprehensive_analysis(
    ctx: RunContext[ExomiserDependencies],
    exomiser_filename: str
) -> Dict[str, Any]:
    """
    Comprehensive multi-agent analysis of Exomiser results.
    
    This function coordinates multiple agents to perform a comprehensive analysis:
    1. Gets Exomiser results and matching phenopacket
    2. Uses HPOA agent to analyze phenotype frequencies and exclusion conflicts
    3. Uses HPO agent to get enriched phenotype definitions
    4. Uses OMIM agent to get enhanced onset information (if enabled)
    5. Uses Literature agent to extract diagnostic test recommendations (if enabled)
    6. Provides all data for reranking
    
    The analysis focuses on the four key pillars:
    - Age of onset from phenopackets and OMIM
    - Phenotype frequency data from HPOA
    - Excluded phenotypes analysis with HPO term enrichment
    - Literature evidence for diagnostic testing
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_filename: The name of the Exomiser result file to analyze
        
    Returns:
        Dict[str, Any]: Comprehensive analysis results containing:
          - original_data: Basic Exomiser and phenopacket data
          - hpoa_analysis: Frequency and disease compatibility data from HPOA
          - hpo_analysis: Enriched phenotype definitions from HPO agent
          - omim_analysis: OMIM onset data (if enabled)
          - literature_analysis: Literature-based diagnostic test recommendations (if enabled)
    """
    # Step 1: Get base Exomiser results and phenopacket
    base_data = await get_result_and_phenopacket(ctx, exomiser_filename)
    
    # Step 2: Get HPOA frequency and disease compatibility analysis
    hpoa_analysis = await analyze_with_hpoa(
        ctx,
        base_data["phenotype_data"],
        base_data["exomiser_results"]
    )
    
    # Step 3: Get enriched phenotype definitions from HPO agent
    all_phenotype_ids = []
    for p in base_data["phenotype_data"]["included_phenotypes"]:
        all_phenotype_ids.append(p["id"])
    for p in base_data["phenotype_data"]["excluded_phenotypes"]:
        all_phenotype_ids.append(p["id"])
    
    hpo_analysis = await get_enriched_phenotype_definitions(ctx, all_phenotype_ids)
    
    # Step 4: Get onset information from OMIM if enabled
    omim_analysis = {}
    if ctx.deps.use_omim:
        omim_analysis = await analyze_with_omim(
            ctx,
            base_data["phenotype_data"].get("onset"),
            base_data["exomiser_results"]
        )
    
    # Step 5: Get diagnostic test recommendations from literature if enabled
    literature_analysis = {}
    if ctx.deps.use_literature:
        literature_analysis = await analyze_with_literature(
            ctx,
            base_data["phenopacket_data"],
            base_data["phenopacket_filename"],
            base_data["exomiser_results"]
        )
    
    # Step 6: Prepare comprehensive reranking data
    comprehensive_context = {
        "exomiser_results": base_data["exomiser_results"][:10],  # Top 10 results
        "phenotype_data": base_data["phenotype_data"],
        "hpoa_analysis": hpoa_analysis,
        "hpo_analysis": hpo_analysis,
        "omim_analysis": omim_analysis,
        "literature_analysis": literature_analysis
    }
    
    # Return the complete results for the agent to process
    return {
        "original_data": base_data,
        "hpoa_analysis": hpoa_analysis,
        "hpo_analysis": hpo_analysis,
        "omim_analysis": omim_analysis if omim_analysis else None,
        "literature_analysis": literature_analysis if literature_analysis else None,
        "comprehensive_context": comprehensive_context
    }


async def comprehensive_reranking_with_exclusions(
    ctx: RunContext[ExomiserDependencies],
    full_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare comprehensive data for reranking by the agent.
    
    This function organizes all multi-agent data for the agent to perform reranking:
    1. Basic phenotype information with enriched definitions from HPO
    2. Excluded phenotype details with enhanced HPO definitions
    3. Disease-phenotype compatibility analysis including exclusion conflicts
    4. Frequency data for phenotypes across diseases
    5. Diagnostic test recommendations from literature
    
    The agent will then rerank diseases with special emphasis on:
    1. Exclusion conflicts - phenotypes expected in a disease but excluded in the patient
    2. Age of onset compatibility
    3. Phenotype frequency in diseases
    4. Available diagnostic tests that could confirm specific diagnoses
    
    Args:
        ctx: The run context with Exomiser configuration
        full_context: All collected data from various agents
        
    Returns:
        Dict[str, Any]: Structured data for reranking
    """
    # Extract key data
    exomiser_results = full_context["exomiser_results"]
    phenotype_data = full_context["phenotype_data"]
    hpo_analysis = full_context.get("hpo_analysis", {})
    
    # Format phenotypes with enriched definitions
    included_phenotypes = []
    for p in phenotype_data['included_phenotypes']:
        phenotype_id = p['id']
        phenotype_info = {
            "id": phenotype_id,
            "label": p['label']
        }
        # Add enriched definition if available
        if phenotype_id in hpo_analysis:
            definition = hpo_analysis[phenotype_id].get("definition", "").strip()
            if definition and definition != "No definition available":
                # Truncate long definitions
                if len(definition) > 100:
                    definition = definition[:100] + "..."
                phenotype_info["definition"] = definition
        included_phenotypes.append(phenotype_info)
    
    # Format excluded phenotypes with enriched definitions
    excluded_phenotypes = []
    for p in phenotype_data.get('excluded_phenotypes', []):
        phenotype_id = p['id']
        phenotype_info = {
            "id": phenotype_id,
            "label": p['label']
        }
        # Add enriched definition if available
        if phenotype_id in hpo_analysis:
            definition = hpo_analysis[phenotype_id].get("definition", "").strip()
            if definition and definition != "No definition available":
                # Truncate long definitions
                if len(definition) > 100:
                    definition = definition[:100] + "..."
                phenotype_info["definition"] = definition
        excluded_phenotypes.append(phenotype_info)
    
    # Format disease candidates
    disease_candidates = []
    for i, result in enumerate(exomiser_results[:10]):  # Top 10 results
        disease_info = {
            "rank": i + 1,
            "id": result.get("DISEASE_ID", "Unknown"),
            "name": result.get("DISEASE_NAME", "Unknown"),
            "score": result.get("COMBINED_SCORE", result.get("score", "Unknown"))
        }
        disease_candidates.append(disease_info)
    
    # Format phenotype analysis if available
    phenotype_analysis = None
    if "__analysis" in hpo_analysis:
        analysis = hpo_analysis["__analysis"]
        if analysis:
            phenotype_analysis = {
                "common_ancestors": analysis.get("common_ancestors", []),
                "phenotype_clusters": analysis.get("phenotype_clusters", {})
            }
    
    # Format disease-phenotype compatibility data
    disease_analysis = []
    if "disease_analyses" in full_context.get("hpoa_analysis", {}):
        for analysis in full_context["hpoa_analysis"]["disease_analyses"]:
            disease_id = analysis["disease_id"]
            disease_compatibility = {
                "id": disease_id,
                "matching_phenotypes": len(analysis['overlapping_phenotypes']),
                "total_patient_phenotypes": len(phenotype_data['included_phenotypes']),
                "missing_expected_phenotypes": len(analysis['missing_phenotypes']),
                "unexpected_phenotypes": len(analysis['unexpected_phenotypes']),
                "exclusion_conflicts": analysis.get('exclusion_conflicts', []),
                "adjusted_score": analysis.get('adjusted_score', 0)
            }
            disease_analysis.append(disease_compatibility)
    
    # Format onset information
    onset_info = {}
    if "omim_analysis" in full_context and full_context["omim_analysis"]:
        onset_compatibility = full_context["omim_analysis"].get("onset_compatibility", {})
        if onset_compatibility:
            for disease_id, info in onset_compatibility.items():
                onset_info[disease_id] = {
                    "disease_onset": info.get("disease_onset", "Unknown"),
                    "compatibility": info.get("compatibility", "Unknown"),
                    "score": info.get("score", 0)
                }
    
    # Format literature data
    literature_info = None
    if "literature_analysis" in full_context and full_context["literature_analysis"]:
        lit_data = full_context["literature_analysis"]
        if "pmid" in lit_data and lit_data["pmid"]:
            literature_info = {
                "pmid": lit_data["pmid"],
                "article_text": lit_data.get("article_text", ""),
                "disease_info": lit_data.get("disease_info", [])
            }
    
    # Prepare comprehensive reranking context
    reranking_context = {
        "patient": {
            "id": phenotype_data['id'],
            "included_phenotypes": included_phenotypes,
            "excluded_phenotypes": excluded_phenotypes,
            "onset": phenotype_data.get('onset')
        },
        "disease_candidates": disease_candidates,
        "phenotype_analysis": phenotype_analysis,
        "disease_analysis": disease_analysis,
        "onset_info": onset_info,
        "literature_info": literature_info
    }
    
    return reranking_context