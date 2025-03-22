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
# Run with UI
ascleon exomiser --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets --ui

# Run comprehensive analysis with OMIM and Literature integration
ascleon exomiser --comprehensive --omim --literature --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets --ui
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
