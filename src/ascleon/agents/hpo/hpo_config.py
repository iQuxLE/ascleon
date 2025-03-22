"""
Configuration classes for the HPO agent.
"""
from dataclasses import dataclass, field
import os
from typing import Optional, Dict, Any

from ascleon.dependencies.workdir import HasWorkdir, WorkDir

# Default paths for data
CHROMADB_PATH = "/db"


@dataclass
class HPODependencies(HasWorkdir):
    """
    Configuration for the HPO agent.
    """
    # Data paths
    chromadb_path: str = field(default=CHROMADB_PATH)
    
    # ChromaDB configuration
    collection_name: str = field(default="enhanced_lrd_hpo_large3")
    
    # Model configuration
    model: str = field(default="claude-3-opus-20240229")  # Default to o1
    
    # Cache for loaded data
    chroma_client: Optional[Any] = None
    collection: Optional[Any] = None
    hpo_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize the config with default values."""
        # Initialize workdir if not provided
        if self.workdir is None:
            self.workdir = WorkDir()


def get_config() -> HPODependencies:
    """
    Get the HPO configuration from environment variables or defaults.
    
    Returns:
        HPODependencies: The HPO dependencies
    """
    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    workdir = WorkDir(location=workdir_path) if workdir_path else None
    
    # Get any environment-specific settings
    chromadb_path = os.environ.get("CHROMADB_PATH", CHROMADB_PATH)
    collection_name = os.environ.get("CHROMADB_COLLECTION", "enhanced_lrd_hpo_large3")
    
    # Get model settings
    model = os.environ.get("HPO_MODEL", "claude-3-opus-20240229")
    
    return HPODependencies(
        workdir=workdir,
        chromadb_path=chromadb_path,
        collection_name=collection_name,
        model=model
    )
