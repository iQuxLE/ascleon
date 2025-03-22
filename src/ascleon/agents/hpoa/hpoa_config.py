"""
Configuration classes for the HPOA agent.
"""
from dataclasses import dataclass, field
import os
from typing import Optional, Dict, Any

from ascleon.dependencies.workdir import HasWorkdir, WorkDir

# Look for phenotype.hpoa in standard locations
from pathlib import Path

def find_hpoa_file():
    """Find the phenotype.hpoa file in standard locations."""
    # Start with the current file's directory
    current_dir = Path(__file__).resolve().parent
    repo_root = None
    
    # Go up the directory tree looking for the repo root
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "data").exists():
            repo_root = parent
            break
    
    if repo_root:
        # Check common locations
        possible_locations = [
            repo_root / "data" / "hpoa" / "phenotype.hpoa",
            repo_root / "data" / "phenotype.hpoa",
        ]
        
        for location in possible_locations:
            if location.exists():
                return str(location)
    
    # Default fallback
    return "/phenotype.hpoa"

# Default path for HPOA file
HPOA_PATH = find_hpoa_file()


@dataclass
class HPOADependencies(HasWorkdir):
    """
    Configuration for the HPOA agent.
    """
    hpoa_path: str = field(default=HPOA_PATH)
    _hpoa_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize the config with default values."""
        # Initialize workdir if not provided
        if self.workdir is None:
            self.workdir = WorkDir()
    
    @property
    def hpoa_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the cached HPOA data if available.
        
        Returns:
            Optional[Dict[str, Any]]: The cached HPOA data or None
        """
        return self._hpoa_data
    
    @hpoa_data.setter
    def hpoa_data(self, data: Dict[str, Any]):
        """
        Set the cached HPOA data.
        
        Args:
            data: The parsed HPOA data
        """
        self._hpoa_data = data


def get_config() -> HPOADependencies:
    """
    Get the HPOA configuration from environment variables or defaults.
    
    Returns:
        HPOADependencies: The HPOA dependencies
    """
    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    workdir = WorkDir(location=workdir_path) if workdir_path else None
    
    # Get any environment-specific settings
    hpoa_path = os.environ.get("HPOA_PATH", HPOA_PATH)
    
    return HPOADependencies(
        workdir=workdir,
        hpoa_path=hpoa_path
    )