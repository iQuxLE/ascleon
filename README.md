# Ascleon: Enhanced Exomiser Results Analysis

Ascleon is an AI-powered agent for re-ranking disease candidate lists from Exomiser based on phenopacket data, focusing on four key pillars:

1. **Excluded Phenotypes** - Leverages phenotypes explicitly excluded during patient evaluation
2. **Age of Onset** - Analyzes compatibility between patient onset and disease onset
3. **Phenotype Frequency** - Considers how common each phenotype is in candidate diseases
4. **Diagnostic Tests** - Extracts test recommendations from literature to guide diagnosis

## Key Features

- **Basic Analysis**: Simple reranking considering patient phenotypes and onset
- **Comprehensive Analysis**: In-depth analysis using multiple agents for:
  - Frequency data from Human Phenotype Ontology Annotations (HPOA)
  - Explicit excluded phenotype compatibility analysis
  - Disease onset information from OMIM
  - Diagnostic test recommendations from scientific literature
- **Interactive UI**: Gradio-based chat interface for exploring results
- **CLI Tools**: Command-line tools for batch processing

## Installation

```bash
# Install using pip
pip install ascleon

# Or install from source
git clone https://github.com/yourusername/ascleon.git
cd ascleon
pip install -e .
```

## Quick Start

```bash
# Run with UI - select from all available files
ascleon exomiser --ui --exomiser-path data/exomiser_results/pheval_disease_results --phenopackets-path data/phenopackets

# Analyze a specific file directly
ascleon exomiser --ui --file PMID_10571775_KSN-II-1-pheval_disease_result.tsv --exomiser-path data/exomiser_results/pheval_disease_results --phenopackets-path data/phenopackets

# List all available Exomiser result files
ascleon exomiser list --exomiser-path data/exomiser_results/pheval_disease_results

# Simple view of an Exomiser file without using the AI agent
ascleon exomiser simple-view --exomiser-path data/exomiser_results/pheval_disease_results PMID_10571775_KSN-II-1-pheval_disease_result.tsv

# Run comprehensive analysis with OMIM and Literature integration
ascleon exomiser --comprehensive --omim --literature --exomiser-path data/exomiser_results/pheval_disease_results --phenopackets-path data/phenopackets --ui
```

You can also use the `analyze` command for direct command-line analysis:

```bash
# Run basic analysis from the command line
ascleon exomiser analyze --exomiser-path data/exomiser_results/pheval_disease_results --phenopackets-path data/phenopackets PMID_10571775_KSN-II-1-pheval_disease_result.tsv

# Run comprehensive analysis from the command line
ascleon exomiser analyze --comprehensive --exomiser-path data/exomiser_results/pheval_disease_results --phenopackets-path data/phenopackets PMID_10571775_KSN-II-1-pheval_disease_result.tsv
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

MIT

## Data Setup

Ascleon requires several data files to function properly:

1. **Exomiser Results (TSV files)**: Output from Exomiser with ranked disease candidates
2. **Phenopackets (JSON files)**: Patient phenotype information
3. **HPO Annotations (phenotype.hpoa)**: Frequency data for phenotypes in diseases

See [Data Setup](docs/data-setup.md) for detailed instructions.

Quick setup:
```bash
# Download sample data (primarily the HPO annotations)
python download_sample_data.py

# Place your own data files
# - Exomiser TSV files → data/exomiser_results/
# - Phenopacket JSON files → data/phenopackets/
```

### File Matching

Ascleon intelligently matches Exomiser result files with their corresponding phenopackets using multiple strategies:

1. **PMID Matching**: Files with patterns like `PMID_12345678` will be matched with phenopackets containing the same PMID
2. **Direct Name Matching**: Searches for phenopackets with similar filenames
3. **Partial Matching**: Uses parts of the filename to find potential matches
4. **Recursive Search**: If needed, searches all subdirectories for possible matches

For best results, keep your phenopackets in the specified phenopackets directory with consistent naming patterns.
