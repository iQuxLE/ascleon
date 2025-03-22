"""
Tools for the HPO agent to provide enriched phenotype information using ChromaDB.
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from pydantic_ai import RunContext, ModelRetry

from ascleon.agents.hpo.hpo_config import HPODependencies


async def initialize_chromadb(ctx: RunContext[HPODependencies]) -> None:
    """
    Initialize ChromaDB connection for HPO data.
    
    Args:
        ctx: The run context with HPO configuration
    """
    # Skip if already initialized
    if ctx.deps.chroma_client is not None and ctx.deps.collection is not None:
        return
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        chroma_path = ctx.deps.chromadb_path
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_path)
        
        ctx.deps.chroma_client = chromadb.Client(settings)
        
        # Get or create collection
        collection_name = ctx.deps.collection_name
        ctx.deps.collection = ctx.deps.chroma_client.get_collection(collection_name)
        
        print(f"ChromaDB initialized with collection: {collection_name}")
    except ImportError:
        raise ModelRetry("ChromaDB is not installed. Please install with 'pip install chromadb'")
    except Exception as e:
        raise ModelRetry(f"Failed to initialize ChromaDB: {str(e)}")


async def get_phenotype_definition(
    ctx: RunContext[HPODependencies],
    phenotype_id: str
) -> Dict[str, Any]:
    """
    Get detailed definition and information for a phenotype term.
    
    Args:
        ctx: The run context with HPO configuration
        phenotype_id: The HPO term ID to look up
        
    Returns:
        Dict[str, Any]: Detailed phenotype information including:
            - id: The HPO ID
            - label: The term label/name
            - definition: The full definition
            - synonyms: Alternative terms 
            - parent_terms: List of parent terms in the HPO hierarchy
            - child_terms: List of child terms in the HPO hierarchy
    """
    # Initialize ChromaDB if needed
    await initialize_chromadb(ctx)
    
    # Check if collection is available
    if ctx.deps.collection is None:
        raise ModelRetry("ChromaDB collection not available")
    
    # Search for the specific HPO ID
    results = ctx.deps.collection.query(
        query_texts=[phenotype_id],
        n_results=1,
        include=["metadatas", "documents"]
    )
    
    # If no exact match, try using it as a search term
    if not results["ids"] or not results["ids"][0]:
        results = ctx.deps.collection.query(
            query_texts=[phenotype_id.replace("HP:", "")],
            n_results=1,
            include=["metadatas", "documents"]
        )
    
    # Process results
    if results["ids"] and results["ids"][0] and results["metadatas"] and results["metadatas"][0]:
        metadata = results["metadatas"][0][0]
        document = results["documents"][0][0] if results["documents"] and results["documents"][0] else ""
        
        # Return structured information
        return {
            "id": metadata.get("id", phenotype_id),
            "label": metadata.get("label", "Unknown"),
            "definition": document or metadata.get("definition", "No definition available"),
            "synonyms": metadata.get("synonyms", []),
            "parent_terms": metadata.get("parents", []),
            "child_terms": metadata.get("children", [])
        }
    
    # If no results found, return minimal information
    return {
        "id": phenotype_id,
        "label": "Unknown",
        "definition": "No definition available in the HPO database",
        "synonyms": [],
        "parent_terms": [],
        "child_terms": []
    }


async def search_phenotype_terms(
    ctx: RunContext[HPODependencies],
    search_term: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for phenotype terms matching a description or keyword.
    
    Args:
        ctx: The run context with HPO configuration
        search_term: The text to search for in phenotype terms
        limit: Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of matching phenotype information
    """
    # Initialize ChromaDB if needed
    await initialize_chromadb(ctx)
    
    # Check if collection is available
    if ctx.deps.collection is None:
        raise ModelRetry("ChromaDB collection not available")
    
    # Search for matching terms
    results = ctx.deps.collection.query(
        query_texts=[search_term],
        n_results=limit,
        include=["metadatas", "documents"]
    )
    
    # Process results
    phenotype_results = []
    
    if results["ids"] and results["ids"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            document = results["documents"][0][i] if results["documents"] and len(results["documents"][0]) > i else ""
            
            phenotype_results.append({
                "id": metadata.get("id", "Unknown"),
                "label": metadata.get("label", "Unknown"),
                "definition": document or metadata.get("definition", "No definition available"),
                "synonyms": metadata.get("synonyms", []),
                "score": results["distances"][0][i] if "distances" in results and results["distances"] else None
            })
    
    return phenotype_results


async def get_phenotype_relationships(
    ctx: RunContext[HPODependencies],
    phenotype_id: str,
    relationship_type: str = "parent"
) -> List[Dict[str, Any]]:
    """
    Get related phenotype terms based on the specified relationship type.
    
    Args:
        ctx: The run context with HPO configuration
        phenotype_id: The HPO term ID to look up relationships for
        relationship_type: The type of relationship to retrieve ("parent", "child", "sibling")
        
    Returns:
        List[Dict[str, Any]]: List of related phenotype terms
    """
    # Get the base phenotype first
    phenotype_data = await get_phenotype_definition(ctx, phenotype_id)
    
    related_terms = []
    
    # Get related terms based on relationship type
    if relationship_type.lower() == "parent":
        # Get parent terms directly from the phenotype data
        parent_ids = phenotype_data.get("parent_terms", [])
        for parent_id in parent_ids:
            parent_data = await get_phenotype_definition(ctx, parent_id)
            related_terms.append(parent_data)
    
    elif relationship_type.lower() == "child":
        # Get child terms directly from the phenotype data
        child_ids = phenotype_data.get("child_terms", [])
        for child_id in child_ids:
            child_data = await get_phenotype_definition(ctx, child_id)
            related_terms.append(child_data)
    
    elif relationship_type.lower() == "sibling":
        # For siblings, first get parents, then get their children
        parent_ids = phenotype_data.get("parent_terms", [])
        sibling_ids = set()
        
        for parent_id in parent_ids:
            parent_data = await get_phenotype_definition(ctx, parent_id)
            for child_id in parent_data.get("child_terms", []):
                if child_id != phenotype_id:  # Exclude the original term
                    sibling_ids.add(child_id)
        
        for sibling_id in sibling_ids:
            sibling_data = await get_phenotype_definition(ctx, sibling_id)
            related_terms.append(sibling_data)
    
    return related_terms


async def analyze_phenotype_overlap(
    ctx: RunContext[HPODependencies],
    phenotype_ids: List[str]
) -> Dict[str, Any]:
    """
    Analyze a set of phenotypes to identify patterns and relationships.
    
    Args:
        ctx: The run context with HPO configuration
        phenotype_ids: List of HPO IDs to analyze
        
    Returns:
        Dict[str, Any]: Analysis results including:
            - common_ancestors: Common parent terms shared by multiple phenotypes
            - phenotype_clusters: Groups of phenotypes that are closely related
            - definitions: Detailed definitions for each phenotype
    """
    # Get definitions for all phenotypes
    definitions = {}
    all_parents = {}
    
    for pheno_id in phenotype_ids:
        pheno_data = await get_phenotype_definition(ctx, pheno_id)
        definitions[pheno_id] = pheno_data
        all_parents[pheno_id] = set(pheno_data.get("parent_terms", []))
    
    # Find common ancestors
    common_ancestors = set()
    if all_parents and len(all_parents) > 1:
        # Start with the parents of the first phenotype
        common_ancestors = set(list(all_parents.values())[0])
        
        # Intersect with parents of other phenotypes
        for parent_set in list(all_parents.values())[1:]:
            common_ancestors.intersection_update(parent_set)
    
    # Get details of common ancestors
    ancestor_details = []
    for ancestor_id in common_ancestors:
        ancestor_data = await get_phenotype_definition(ctx, ancestor_id)
        ancestor_details.append(ancestor_data)
    
    # Simple clustering by shared parents (can be improved)
    clusters = {}
    processed = set()
    
    for i, pheno_id in enumerate(phenotype_ids):
        if pheno_id in processed:
            continue
            
        # Start new cluster
        cluster = [pheno_id]
        processed.add(pheno_id)
        
        # Check remaining phenotypes for shared parents
        for j, other_id in enumerate(phenotype_ids[i+1:]):
            if other_id in processed:
                continue
                
            # If they share significant parents, add to cluster
            shared_parents = all_parents[pheno_id].intersection(all_parents[other_id])
            if len(shared_parents) >= 2:  # Arbitrary threshold
                cluster.append(other_id)
                processed.add(other_id)
        
        if len(cluster) > 1:  # Only keep non-singleton clusters
            clusters[f"cluster_{len(clusters)+1}"] = cluster
    
    # Format the clustering results
    cluster_details = {}
    for cluster_name, cluster_phenotypes in clusters.items():
        phenotype_details = []
        for pheno_id in cluster_phenotypes:
            phenotype_details.append({
                "id": pheno_id,
                "label": definitions[pheno_id].get("label")
            })
        cluster_details[cluster_name] = phenotype_details
    
    return {
        "common_ancestors": ancestor_details,
        "phenotype_clusters": cluster_details,
        "definitions": definitions
    }
