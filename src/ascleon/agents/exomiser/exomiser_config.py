"""
Configuration classes for the exomiser agent.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

from ascleon.dependencies.workdir import HasWorkdir, WorkDir
from ascleon.agents.hpoa.hpoa_config import HPOADependencies
from ascleon.agents.hpo.hpo_config import HPODependencies
from ascleon.agents.omim.omim_config import OMIMDependencies
from ascleon.agents.literature.literature_config import LiteratureDependencies

# Find the repository root
def find_repo_root():
    """Find the repository root by looking for data directory."""
    # Start with the current file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Go up the directory tree looking for the repo root
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "data").exists() or (parent / "ascleon" / "data").exists():
            return parent
    
    # If not found, default to the current directory
    return current_dir

REPO_ROOT = find_repo_root()

# Default paths for data - prioritize environment variables, then repo structure
DEFAULT_EXOMISER_RESULTS_PATH = os.environ.get(
    "EXOMISER_RESULTS_PATH", 
    str(REPO_ROOT / "data" / "exomiser_results")
)
DEFAULT_PHENOPACKETS_PATH = os.environ.get(
    "PHENOPACKETS_PATH", 
    str(REPO_ROOT / "data" / "phenopackets")
)
DEFAULT_HPOA_PATH = os.environ.get(
    "HPOA_PATH", 
    str(REPO_ROOT / "data" / "hpoa" / "phenotype.hpoa")
)
DEFAULT_CHROMADB_PATH = os.environ.get(
    "CHROMADB_PATH", 
    str(REPO_ROOT / "data" / "chromadb")
)


@dataclass
class ExomiserDependencies(HasWorkdir):
    """
    Configuration for the exomiser agent.
    """
    # Data paths
    exomiser_results_path: str = field(default=DEFAULT_EXOMISER_RESULTS_PATH)
    phenopackets_path: str = field(default=DEFAULT_PHENOPACKETS_PATH)
    hpoa_path: str = field(default=DEFAULT_HPOA_PATH)
    chromadb_path: str = field(default=DEFAULT_CHROMADB_PATH)
    
    # ChromaDB configuration
    collection_name: str = field(default="enhanced_lrd_hpo_large3")
    
    # Model configuration
    model: str = field(default="gpt-4o")  # Default to o1
    multimodal_model: str = field(default="gemini-1.5-flash-latest")  # For image analysis
    
    # Agent dependencies
    hpoa: Optional[HPOADependencies] = None
    hpo: Optional[HPODependencies] = None
    omim: Optional[OMIMDependencies] = None
    literature: Optional[LiteratureDependencies] = None
    
    # Analysis options
    comprehensive: bool = field(default=False)  # Whether to run comprehensive analysis
    use_omim: bool = field(default=False)  # Whether to use OMIM data for onset analysis
    use_literature: bool = field(default=False)  # Whether to extract diagnostic tests from literature

    def __post_init__(self):
        """Initialize the config with default values."""
        # Initialize workdir if not provided
        if self.workdir is None:
            self.workdir = WorkDir()
            
        # Initialize connected agents
        if self.hpoa is None:
            self.hpoa = HPOADependencies(hpoa_path=self.hpoa_path)
            
        if self.hpo is None:
            self.hpo = HPODependencies(
                chromadb_path=self.chromadb_path,
                collection_name=self.collection_name
            )
            
        if self.omim is None:
            self.omim = OMIMDependencies()
            # OMIM API key is expected to be set via the environment variable
            
        if self.literature is None:
            self.literature = LiteratureDependencies()
            # Literature agent will use any available API keys from environment


def get_config() -> ExomiserDependencies:
    """
    Get the Exomiser configuration from environment variables or defaults.
    
    Returns:
        ExomiserDependencies: The exomiser dependencies
    """
    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    workdir = WorkDir(location=workdir_path) if workdir_path else None
    
    # Get any environment-specific settings
    exomiser_path = os.environ.get("EXOMISER_RESULTS_PATH", DEFAULT_EXOMISER_RESULTS_PATH)
    phenopackets_path = os.environ.get("PHENOPACKETS_PATH", DEFAULT_PHENOPACKETS_PATH)
    hpoa_path = os.environ.get("HPOA_PATH", DEFAULT_HPOA_PATH)
    chromadb_path = os.environ.get("CHROMADB_PATH", DEFAULT_CHROMADB_PATH)
    collection_name = os.environ.get("CHROMADB_COLLECTION", "enhanced_lrd_hpo_large3")
    
    # Get model settings
    # Explicitly support Gemini models
    model_name = os.environ.get("EXOMISER_MODEL", "claude-3-opus-20240229")
    if model_name.startswith("gemini"):
        model = f"google:{model_name}"
    else:
        model = model_name
    multimodal_model = os.environ.get("EXOMISER_MULTIMODAL_MODEL", "gemini-1.5-flash-latest")
    
    # Get analysis options
    comprehensive = os.environ.get("EXOMISER_COMPREHENSIVE", "false").lower() == "true"
    use_omim = os.environ.get("EXOMISER_USE_OMIM", "false").lower() == "true"
    use_literature = os.environ.get("EXOMISER_USE_LITERATURE", "false").lower() == "true"
    
    # Create agent dependencies
    hpoa_deps = HPOADependencies(
        workdir=workdir,
        hpoa_path=hpoa_path
    )
    
    hpo_deps = HPODependencies(
        workdir=workdir,
        chromadb_path=chromadb_path,
        collection_name=collection_name
    )
    
    omim_deps = OMIMDependencies(
        workdir=workdir,
        api_key=os.environ.get("OMIM_API_KEY", ""),
        cache_dir=os.environ.get("OMIM_CACHE_DIR", "omim_cache")
    )
    
    literature_deps = LiteratureDependencies(
        workdir=workdir
    )
    
    return ExomiserDependencies(
        workdir=workdir,
        exomiser_results_path=exomiser_path,
        phenopackets_path=phenopackets_path,
        hpoa_path=hpoa_path,
        chromadb_path=chromadb_path,
        collection_name=collection_name,
        model=model,
        multimodal_model=multimodal_model,
        hpoa=hpoa_deps,
        hpo=hpo_deps,
        omim=omim_deps,
        literature=literature_deps,
        comprehensive=comprehensive,
        use_omim=use_omim,
        use_literature=use_literature
    )
