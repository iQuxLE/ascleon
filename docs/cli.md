# Command Line Interface

Ascleon provides a comprehensive command-line interface for analyzing Exomiser results.

## Basic Commands

### Running the UI

```bash
ascleon exomiser --ui --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```

### Analyzing a Specific File

```bash
ascleon analyze --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets sample_result.tsv
```

### Listing Available Files

```bash
ascleon list --exomiser-path /path/to/exomiser_results
```

## Analysis Modes

### Basic Mode

```bash
ascleon exomiser --basic --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```

### Comprehensive Mode

```bash
ascleon exomiser --comprehensive --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```

## Optional Integrations

### OMIM Integration (for enhanced onset analysis)

```bash
ascleon exomiser --comprehensive --omim --omim-api-key YOUR_API_KEY --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```

### Literature Integration (for diagnostic test recommendations)

```bash
ascleon exomiser --comprehensive --literature --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```

### Full Integration

```bash
ascleon exomiser --comprehensive --omim --literature --exomiser-path /path/to/exomiser_results --phenopackets-path /path/to/phenopackets
```
