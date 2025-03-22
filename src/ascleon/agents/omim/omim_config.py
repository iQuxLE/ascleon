"""
Configuration classes for the OMIM agent.
"""
from dataclasses import dataclass, field
import os
from typing import Optional, Dict, Any

from ascleon.dependencies.workdir import HasWorkdir, WorkDir


@dataclass
class OMIMDependencies(HasWorkdir):
    """
    Configuration for the OMIM agent.
    """
    # API configuration
    api_key: str = field(default="")
    cache_dir: str = field(default="omim_cache")
    
    # Rate limiting settings
    requests_per_second: float = field(default=0.5)  # Max 2 requests per second
    
    # Model configuration
    model: str = field(default="claude-3-opus-20240229")
    
    # Cache for loaded data
    cache: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize the config with default values."""
        # Initialize workdir if not provided
        if self.workdir is None:
            self.workdir = WorkDir()
            
        # Initialize cache
        if self.cache is None:
            self.cache = {}


def get_config() -> OMIMDependencies:
    """
    Get the OMIM configuration from environment variables or defaults.
    
    Returns:
        OMIMDependencies: The OMIM dependencies
    """
    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    workdir = WorkDir(location=workdir_path) if workdir_path else None
    
    # Get API key from environment variable (required for OMIM API)
    api_key = os.environ.get("OMIM_API_KEY", "")
    
    # Get cache directory
    cache_dir = os.environ.get("OMIM_CACHE_DIR", "omim_cache")
    
    # Get rate limiting settings
    requests_per_second = float(os.environ.get("OMIM_REQUESTS_PER_SECOND", "0.5"))
    
    # Get model settings
    model = os.environ.get("OMIM_MODEL", "claude-3-opus-20240229")
    
    return OMIMDependencies(
        workdir=workdir,
        api_key=api_key,
        cache_dir=cache_dir,
        requests_per_second=requests_per_second,
        model=model
    )
