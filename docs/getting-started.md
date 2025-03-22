# Getting Started

This guide will help you get started with Ascleon for re-ranking Exomiser results.

## Prerequisites

1. Exomiser result files (.tsv format)
2. Corresponding phenopackets (.json format)
3. Optional: HPOA file for phenotype frequency analysis
4. Optional: ChromaDB with HPO terms for enriched definitions
5. Optional: OMIM API key for onset analysis

## Basic Setup

1. Install Ascleon:
   ```bash
   pip install ascleon
   ```

2. Identify your data paths:
   - Exomiser results directory
   - Phenopackets directory
   - HPOA file location (optional)
   - ChromaDB path (optional)

3. Start the UI:
   ```bash
   ascleon exomiser --ui --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
   ```

## Analyzing Results

1. Select an Exomiser result file from the dropdown
2. Ask questions about the results or request reranking
3. The agent will analyze the data and present reranked results

## Comprehensive Analysis

For comprehensive analysis that uses all four pillars:

```bash
ascleon exomiser --comprehensive --omim --literature --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets --ui
```

This enables:
- Excluded phenotype compatibility analysis
- Age of onset compatibility checking
- Phenotype frequency data from HPOA
- Diagnostic test recommendations from literature
