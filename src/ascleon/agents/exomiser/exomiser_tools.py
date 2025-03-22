"""
Tools for the exomiser agent to re-rank results based on phenopacket data.
"""
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pydantic_ai
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
    Use an AI model to rerank Exomiser results based on phenopacket data.
    
    Args:
        ctx: The run context with Exomiser configuration
        exomiser_results: The parsed Exomiser results
        phenotype_data: The structured phenotype data extracted from the phenopacket
        
    Returns:
        Dict[str, Any]: Reranked Exomiser results
    """
    # Prepare the data for the model
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
    
    # Create a prompt for the model
    prompt = f"""
I need you to rerank these disease candidates based on their compatibility with the patient's phenotypes and onset information.

PATIENT INFORMATION:
- ID: {phenotype_data['id']}
- Phenotypes present: {included_phenotypes_text}
{onset_text}
{excluded_text}

DISEASE CANDIDATES (Original ranking):
{", ".join([f"{i+1}. {result.get('DISEASE_NAME', 'Unknown')} ({result.get('DISEASE_ID', 'Unknown')})" for i, result in enumerate(top_results)])}

Please rerank these disease candidates based on their compatibility with the patient's phenotypes AND especially the onset information.

Your response should only include the reranked list from 1 to 10 in this exact format:

1. [Disease Name] ([Disease ID])
2. [Disease Name] ([Disease ID])
...
10. [Disease Name] ([Disease ID])

You may also suggest other diseases not in the original list if appropriate, listing them after the reranked list.
"""
    
    # Use the model to rerank
    model_result = await pydantic_ai.chat.completions.create(
        model=ctx.deps.model,
        messages=[
            {"role": "system", "content": "You are a clinical geneticist expert in rare disease diagnosis. Your task is to rerank disease candidates based on phenotypic features and disease onset."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    
    response_text = model_result.choices[0].message.content
    
    # Parse the model's response to extract the reranked list
    lines = response_text.strip().split('\n')
    reranked_results = []
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            # This looks like a ranked result
            rank_text = line.split('. ', 1)[1]
            reranked_results.append({"rank": len(reranked_results) + 1, "description": rank_text})
    
    # Return the reranked results along with the model's full explanation
    return {
        "reranked_list": reranked_results[:10],  # Ensure we only return top 10
        "model_explanation": response_text
    }


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
    3. Performs basic reranking using onset information
    
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
          - reranking_results: Reranked disease list with explanation
    """
    # Get combined data
    combined_data = await get_result_and_phenopacket(ctx, exomiser_filename)
    
    # Perform the reranking
    reranking_results = await rerank_exomiser_results(
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
        "reranking_results": reranking_results
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
            article_text = await lookup_pmid(f"PMID:{pmid}")
            print(f"Retrieved full text for PMID:{pmid}")
        except Exception:
            # If full text fails, try just the abstract
            try:
                article_text = await get_article_abstract(f"PMID:{pmid}")
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
        
        # Create a prompt to extract diagnostic test recommendations
        import pydantic_ai
        
        # Add System Prompt
        prompt_system = "You are an expert in clinical genetics and diagnostic medicine. Your task is to extract diagnostic test recommendations from scientific literature that would help confirm or rule out specific genetic disease diagnoses."
        
        # Create User Prompt
        prompt_user = f"""
I have a patient with a suspected rare genetic disease, and I need to identify diagnostic tests that could help confirm or exclude the candidate diseases.
I need you to analyze the following article and extract explicit diagnostic test recommendations.

CANDIDATE DISEASES (in order of initial ranking):
{", ".join([f"{d['rank']}. {d['name']} ({d['id']})" for d in disease_info])}

ARTICLE TEXT:
{article_text[:12000]}  # Truncate text if too long

Please provide:
1. A list of all diagnostic tests mentioned in this article that could be useful for diagnosis
2. For each of the candidate diseases, identify which specific tests would be most informative
3. Indicate how each recommended test might help confirm or exclude a diagnosis

Format your response as structured information that can be easily parsed. Be specific about test names, methodologies, and how they relate to specific diseases.
"""

        # Use the model to extract test recommendations
        model_result = await pydantic_ai.chat.completions.create(
            model=ctx.deps.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            max_tokens=1500
        )
        
        response_text = model_result.choices[0].message.content
        
        # Process the extracted recommendations
        extracted_tests = []
        disease_specific_tests = {}
        
        # Simple logic to extract structured information from the model response
        # In a real implementation, this would use proper parsing logic or a structured schema
        
        # Try to parse the model's response into structured sections
        sections = response_text.split("\n\n")
        for section in sections:
            if "diagnostic test" in section.lower() or "recommended test" in section.lower():
                extracted_tests.append(section.strip())
            
            # Check for disease-specific test recommendations
            for disease in disease_info:
                disease_id = disease["id"]
                disease_name = disease["name"]
                
                if disease_id in section or disease_name in section:
                    if disease_id not in disease_specific_tests:
                        disease_specific_tests[disease_id] = []
                    disease_specific_tests[disease_id].append(section.strip())
        
        return {
            "pmid": pmid,
            "article_analysis": response_text,
            "test_recommendations": extracted_tests,
            "disease_specific_tests": disease_specific_tests
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
    6. Reranks diseases using all available information
    
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
          - reranking_results: Final reranking with comprehensive explanation
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
    
    # Step 6: Perform comprehensive reranking with all collected data
    full_context = {
        "exomiser_results": base_data["exomiser_results"],
        "phenotype_data": base_data["phenotype_data"],
        "phenotype_frequencies": hpoa_analysis["phenotype_frequencies"],
        "excluded_phenotype_frequencies": hpoa_analysis["excluded_phenotype_frequencies"],
        "disease_analyses": hpoa_analysis["disease_analyses"],
        "enriched_phenotypes": hpo_analysis,
        "omim_onset": omim_analysis,
        "literature_analysis": literature_analysis
    }
    
    reranking_results = await comprehensive_reranking_with_exclusions(ctx, full_context)
    
    # Return the complete results
    result = {
        "original_data": base_data,
        "hpoa_analysis": hpoa_analysis,
        "hpo_analysis": hpo_analysis,
        "reranking_results": reranking_results
    }
    
    # Add OMIM analysis if available
    if omim_analysis:
        result["omim_analysis"] = omim_analysis
    
    # Add literature analysis if available
    if literature_analysis:
        result["literature_analysis"] = literature_analysis
    
    return result


async def comprehensive_reranking_with_exclusions(
    ctx: RunContext[ExomiserDependencies],
    full_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhanced reranking using all available information from multiple agents.
    
    This function creates a comprehensive prompt for the LLM that includes:
    1. Basic phenotype information with enriched definitions from HPO
    2. Excluded phenotype details with enhanced HPO definitions
    3. Disease-phenotype compatibility analysis including exclusion conflicts
    4. Frequency data for phenotypes across diseases
    5. Diagnostic test recommendations from literature
    
    The LLM then reranks diseases with special emphasis on:
    1. Exclusion conflicts - phenotypes expected in a disease but excluded in the patient
    2. Age of onset compatibility
    3. Phenotype frequency in diseases
    4. Available diagnostic tests that could confirm specific diagnoses
    
    Args:
        ctx: The run context with Exomiser configuration
        full_context: All collected data from various agents
        
    Returns:
        Dict[str, Any]: Reranking results with detailed explanation of reasoning
    """
    # Extract key data
    exomiser_results = full_context["exomiser_results"]
    phenotype_data = full_context["phenotype_data"]
    top_results = exomiser_results[:10]
    enriched_phenotypes = full_context.get("enriched_phenotypes", {})
    
    # Prepare formatted strings for the prompt
    included_phenotypes_text_parts = []
    for p in phenotype_data['included_phenotypes']:
        phenotype_id = p['id']
        phenotype_text = f"{p['label']} ({phenotype_id})"
        # Add enriched definition if available
        if phenotype_id in enriched_phenotypes:
            definition = enriched_phenotypes[phenotype_id].get("definition", "").strip()
            if definition and definition != "No definition available":
                # Truncate long definitions
                if len(definition) > 100:
                    definition = definition[:100] + "..."
                phenotype_text += f" - {definition}"
        included_phenotypes_text_parts.append(phenotype_text)
    
    included_phenotypes_text = "\n  * " + "\n  * ".join(included_phenotypes_text_parts)
    
    # Format onset information
    if phenotype_data.get('onset'):
        onset_text = f"- Disease onset: {phenotype_data['onset']['label']} ({phenotype_data['onset']['id']})"
    else:
        onset_text = "- Disease onset: Not specified"
    
    # Format excluded phenotypes with enriched definitions
    excluded_text = "- EXCLUDED phenotypes: "
    if phenotype_data.get('excluded_phenotypes'):
        excluded_phenotypes_text_parts = []
        for p in phenotype_data.get('excluded_phenotypes', []):
            phenotype_id = p['id']
            phenotype_text = f"{p['label']} ({phenotype_id})"
            # Add enriched definition if available
            if phenotype_id in enriched_phenotypes:
                definition = enriched_phenotypes[phenotype_id].get("definition", "").strip()
                if definition and definition != "No definition available":
                    # Truncate long definitions
                    if len(definition) > 100:
                        definition = definition[:100] + "..."
                    phenotype_text += f" - {definition}"
            excluded_phenotypes_text_parts.append(phenotype_text)
        excluded_text += "\n  * " + "\n  * ".join(excluded_phenotypes_text_parts)
    else:
        excluded_text += "None"
    
    # Add phenotype relationship analysis if available
    phenotype_analysis_text = ""
    if "__analysis" in enriched_phenotypes:
        analysis = enriched_phenotypes["__analysis"]
        
        # Add information about common ancestors
        common_ancestors = analysis.get("common_ancestors", [])
        if common_ancestors:
            phenotype_analysis_text += "\nPHENOTYPE RELATIONSHIP ANALYSIS:\n"
            phenotype_analysis_text += "- Common ancestors/parent terms shared by multiple phenotypes:\n"
            for ancestor in common_ancestors[:3]:  # Limit to top 3
                phenotype_analysis_text += f"  * {ancestor.get('label', 'Unknown')} ({ancestor.get('id', 'Unknown')})\n"
        
        # Add information about phenotype clusters
        clusters = analysis.get("phenotype_clusters", {})
        if clusters:
            phenotype_analysis_text += "- Phenotype clusters (groups of related phenotypes):\n"
            for cluster_name, cluster_phenotypes in list(clusters.items())[:2]:  # Limit to top 2 clusters
                phenotype_analysis_text += f"  * {cluster_name}: "
                phenotype_analysis_text += ", ".join([f"{p.get('label', 'Unknown')} ({p.get('id', 'Unknown')})" 
                                                 for p in cluster_phenotypes])
                phenotype_analysis_text += "\n"
    
    # Build enhanced prompt incorporating all data with focus on excluded phenotypes
    prompt = f"""
I need you to rerank these disease candidates based on their compatibility with the patient's phenotypes and additional information, with special attention to EXCLUDED phenotypes.

PATIENT INFORMATION:
- ID: {phenotype_data['id']}
- Phenotypes present: {included_phenotypes_text}
{onset_text}
{excluded_text}
{phenotype_analysis_text}
"""
    
    # Add disease analysis data with focus on exclusion conflicts
    if "disease_analyses" in full_context:
        prompt += "\nDISEASE-PHENOTYPE ANALYSIS (including exclusion conflicts):\n"
        for analysis in full_context.get("disease_analyses", []):
            disease_id = analysis["disease_id"]
            prompt += f"\n{disease_id}:\n"
            prompt += f"- Matching phenotypes: {len(analysis['overlapping_phenotypes'])}/{len(phenotype_data['included_phenotypes'])}\n"
            prompt += f"- Missing expected phenotypes: {len(analysis['missing_phenotypes'])}\n"
            prompt += f"- Unexpected phenotypes: {len(analysis['unexpected_phenotypes'])}\n"
            
            # Highlight exclusion conflicts specifically
            exclusion_conflicts = analysis.get("exclusion_conflicts", [])
            if exclusion_conflicts:
                prompt += f"- EXCLUSION CONFLICTS: {len(exclusion_conflicts)} excluded phenotypes that typically occur in this disease\n"
                conflict_labels = []
                for pheno_id in exclusion_conflicts:
                    # Try to get the label for this phenotype
                    label = next((p["label"] for p in phenotype_data["excluded_phenotypes"] if p["id"] == pheno_id), pheno_id)
                    conflict_labels.append(f"{label} ({pheno_id})")
                prompt += f"  - Conflicts: {', '.join(conflict_labels)}\n"
            else:
                prompt += f"- EXCLUSION CONFLICTS: None (good!)\n"
            
            prompt += f"- Adjusted score (accounting for exclusions): {analysis.get('adjusted_score', 0):.2f}\n"
            
            # Add OMIM onset information if available
            if "omim_onset" in full_context and full_context["omim_onset"]:
                onset_compatibility = full_context["omim_onset"].get("onset_compatibility", {})
                if disease_id in onset_compatibility:
                    onset_info = onset_compatibility[disease_id]
                    prompt += f"- OMIM onset: {onset_info.get('disease_onset', 'Unknown')}\n"
                    prompt += f"- Onset compatibility: {onset_info.get('compatibility', 'Unknown')} (score: {onset_info.get('score', 0):.2f})\n"
    
    # Add phenotype frequency data
    if "phenotype_frequencies" in full_context:
        prompt += "\nPHENOTYPE FREQUENCY DATA:\n"
        for phenotype_id, frequencies in full_context.get("phenotype_frequencies", {}).items():
            phenotype_label = next((p["label"] for p in phenotype_data["included_phenotypes"] if p["id"] == phenotype_id), phenotype_id)
            prompt += f"\n{phenotype_label} ({phenotype_id}):\n"
            
            # Add frequencies for top diseases only to keep prompt manageable
            relevant_diseases = [r.get("DISEASE_ID") for r in top_results]
            relevant_freqs = {k: v for k, v in frequencies.items() if k in relevant_diseases}
            
            if relevant_freqs:
                for disease_id, freq in relevant_freqs.items():
                    prompt += f"- {disease_id}: {freq.get('label', 'Unknown')}\n"
            else:
                prompt += "- No frequency data available for top candidate diseases\n"
    
    # Add diagnostic test recommendations from literature
    literature_text = ""
    if "literature_analysis" in full_context and full_context.get("literature_analysis"):
        literature_data = full_context["literature_analysis"]
        
        if "pmid" in literature_data and literature_data["pmid"]:
            literature_text += f"\nDIAGNOSTIC TEST RECOMMENDATIONS FROM LITERATURE (PMID:{literature_data['pmid']}):\n"
            
            # General test recommendations
            if "test_recommendations" in literature_data and literature_data["test_recommendations"]:
                literature_text += "\nGeneral test recommendations:\n"
                for i, test in enumerate(literature_data["test_recommendations"][:5]):  # Limit to 5 tests
                    literature_text += f"- {test}\n"
            
            # Disease-specific test recommendations
            if "disease_specific_tests" in literature_data and literature_data["disease_specific_tests"]:
                literature_text += "\nDisease-specific test recommendations:\n"
                for disease_id, tests in literature_data["disease_specific_tests"].items():
                    disease_name = next((r.get("DISEASE_NAME", "Unknown") for r in top_results if r.get("DISEASE_ID") == disease_id), "Unknown")
                    literature_text += f"\n{disease_name} ({disease_id}):\n"
                    for test in tests[:2]:  # Limit to 2 tests per disease
                        literature_text += f"- {test}\n"
    
    prompt += literature_text
    
    # Add disease candidates for reranking
    prompt += f"""
DISEASE CANDIDATES (Original ranking):
{", ".join([f"{i+1}. {result.get('DISEASE_NAME', 'Unknown')} ({result.get('DISEASE_ID', 'Unknown')})" for i, result in enumerate(top_results)])}

Please rerank these disease candidates with special attention to these FIVE KEY FACTORS in order of importance:

1. EXCLUDED PHENOTYPES - Diseases that typically present with phenotypes explicitly excluded in this patient should be ranked lower or eliminated
2. AGE OF ONSET - Compatibility with the patient's disease onset information
3. PHENOTYPE FREQUENCY - How common each phenotype is in each disease
4. PHENOTYPE OVERLAP - How well the patient's phenotypes match the disease's typical presentation
5. DIAGNOSTIC TESTS - Available diagnostic tests that could confirm specific diagnoses

Your response should include the reranked list from 1 to 10 in this exact format:

1. [Disease Name] ([Disease ID])
2. [Disease Name] ([Disease ID])
...
10. [Disease Name] ([Disease ID])

After the list, provide detailed reasoning explaining your reranking decisions.
For each of the top 5 diseases, explicitly address:
- How the excluded phenotypes influenced your ranking (this is most important)
- How the onset information affected the ranking
- How frequency data informed your decision
- How the phenotype overlap impacted your assessment
- What diagnostic tests would be most helpful in confirming the diagnosis
"""
    
    # Use the model to rerank with all data
    model_result = await pydantic_ai.chat.completions.create(
        model=ctx.deps.model,
        messages=[
            {"role": "system", "content": "You are a clinical geneticist expert in rare disease diagnosis. Your task is to rerank disease candidates based on phenotypic features, with special emphasis on excluded phenotypes, disease onset, frequency data, and diagnostic test availability."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    
    response_text = model_result.choices[0].message.content
    
    # Parse the model's response to extract the reranked list
    lines = response_text.strip().split('\n')
    reranked_results = []
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            # This looks like a ranked result
            rank_text = line.split('. ', 1)[1]
            reranked_results.append({"rank": len(reranked_results) + 1, "description": rank_text})
    
    # Separate reasoning from results list
    reasoning_text = response_text
    for result in reranked_results:
        reasoning_text = reasoning_text.replace(f"{result['rank']}. {result['description']}", "")
    reasoning_text = reasoning_text.strip()
    
    return {
        "reranked_list": reranked_results[:10],  # Ensure we only return top 10
        "model_explanation": reasoning_text
    }