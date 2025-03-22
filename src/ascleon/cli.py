#\!/usr/bin/env python3
"""
Command line interface for the standalone Exomiser agent.
"""
import os
import logging
import click

logger = logging.getLogger(__name__)

@click.group()
@click.option("-v", "--verbose", count=True)
@click.version_option("0.1.0")  # Set your version
def main(verbose: int):
    """
    Ascleon - Exomiser Agent CLI for re-ranking disease candidates using phenopacket data.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    import logfire
    logfire.configure()

@main.command()
@click.option("--model", "-m", help="The model to use for the agent.")
@click.option("--workdir", "-w", default="workdir", show_default=True, 
              help="The working directory for the agent.")
@click.option("--share/--no-share", default=False, show_default=True,
              help="Share the agent GradIO UI via URL.")
@click.option("--server-port", "-p", default=7860, show_default=True,
              help="The port to run the UI server on.")
@click.option("--ui/--no-ui", default=False, show_default=True,
              help="Start the agent in UI mode instead of direct query mode.")
@click.option("--exomiser-path", help="Path to the directory containing Exomiser results")
@click.option("--phenopackets-path", help="Path to the directory containing phenopackets")
@click.option("--hpoa-path", help="Path to the HPOA file for phenotype frequency analysis")
@click.option("--chromadb-path", help="Path to the ChromaDB directory", default="/db")
@click.option("--comprehensive/--basic", help="Enable comprehensive multi-agent analysis mode", default=False)
@click.option("--omim/--no-omim", help="Use OMIM API to fetch onset information", default=False)
@click.option("--literature/--no-literature", help="Use Literature agent to analyze publication-based diagnostic tests", default=False)
@click.option("--omim-api-key", help="OMIM API key for accessing the OMIM database", envvar="OMIM_API_KEY")
@click.option("--multimodal-model", help="Model to use for multimodal analysis", default="gemini-1.5-flash-latest")
@click.argument("query", nargs=-1, required=False)
def exomiser(ui, query, exomiser_path, phenopackets_path, hpoa_path, chromadb_path, 
             comprehensive, omim, literature, omim_api_key, multimodal_model, 
             model, workdir, share, server_port, **kwargs):
    """
    Exomiser Agent for re-ranking disease candidates from Exomiser results using phenopacket data.
    
    This agent analyzes phenopacket data and re-ranks disease candidates based on:
    
    1. EXCLUDED PHENOTYPES - Phenotypes ruled out during patient evaluation
    2. AGE OF ONSET - When symptoms first appeared
    3. PHENOTYPE FREQUENCY - How common phenotypes are in each disease
    4. DIAGNOSTIC TESTS - Literature-derived test recommendations
    
    Run with --ui flag to start the interactive interface.
    """
    # Set environment variables for configuration
    if workdir:
        os.environ["AURELIAN_WORKDIR"] = workdir
    if exomiser_path:
        os.environ["EXOMISER_RESULTS_PATH"] = exomiser_path
    if phenopackets_path:
        os.environ["PHENOPACKETS_PATH"] = phenopackets_path
    if hpoa_path:
        os.environ["HPOA_PATH"] = hpoa_path
    if chromadb_path:
        os.environ["CHROMADB_PATH"] = chromadb_path
    if model:
        os.environ["EXOMISER_MODEL"] = model
    if multimodal_model:
        os.environ["EXOMISER_MULTIMODAL_MODEL"] = multimodal_model
    
    # Set analysis options
    os.environ["EXOMISER_COMPREHENSIVE"] = "true" if comprehensive else "false"
    os.environ["EXOMISER_USE_OMIM"] = "true" if omim else "false"
    os.environ["EXOMISER_USE_LITERATURE"] = "true" if literature else "false"
    
    # Set OMIM API key if provided
    if omim_api_key:
        os.environ["OMIM_API_KEY"] = omim_api_key
    
    try:
        from ascleon.agents.exomiser.exomiser_config import get_config
        from ascleon.utils.async_utils import run_sync
        
        # Create agent dependencies
        deps = get_config()
        
        if ui:
            # Run in UI mode
            from ascleon.agents.exomiser.exomiser_gradio import chat
            ui = chat(
                exomiser_results_path=exomiser_path,
                phenopackets_path=phenopackets_path,
                hpoa_path=hpoa_path,
                chromadb_path=chromadb_path,
                model=model,
                multimodal_model=multimodal_model,
                comprehensive=comprehensive,
                use_omim=omim,
                use_literature=literature
            )
            ui.launch(server_port=server_port, share=share)
        else:
            # Run in direct query mode with given query
            if query:
                from ascleon.agents.exomiser.exomiser_agent import exomiser_agent
                
                if comprehensive:
                    query_text = f"Perform comprehensive analysis on Exomiser results: {' '.join(query)}"
                else:
                    query_text = f"Analyze Exomiser results: {' '.join(query)}"
                    
                result = run_sync(lambda: exomiser_agent.run_sync(query_text, deps=deps))
                print(result.data)
            else:
                click.echo("Error: Either provide a query or use --ui flag")
    except ImportError as e:
        click.echo(f"ERROR: Missing required modules. {str(e)}")
        click.echo("Make sure all dependencies are installed.")
    except Exception as e:
        click.echo(f"ERROR: {str(e)}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())

@main.command()
@click.option("--model", "-m", help="The model to use for the analysis.")
@click.option("--exomiser-path", help="Path to the directory containing Exomiser results", required=True)
@click.option("--phenopackets-path", help="Path to the directory containing phenopackets", required=True)
@click.option("--comprehensive/--basic", help="Enable comprehensive analysis", default=True)
@click.option("--omim/--no-omim", help="Use OMIM API for onset analysis", default=False)
@click.option("--literature/--no-literature", help="Use Literature agent for diagnostic tests", default=False)
@click.argument("filename", required=True)
def analyze(model, exomiser_path, phenopackets_path, comprehensive, omim, literature, filename):
    """
    Analyze a specific Exomiser result file and display the reranking.
    
    This command performs a direct analysis on the specified Exomiser result file
    without starting the UI. It's useful for batch processing or scripting.
    
    Example: ascleon analyze --exomiser-path /data/exomiser --phenopackets-path /data/phenopackets sample_result.tsv
    """
    # Set environment variables for configuration
    os.environ["EXOMISER_RESULTS_PATH"] = exomiser_path
    os.environ["PHENOPACKETS_PATH"] = phenopackets_path
    if model:
        os.environ["EXOMISER_MODEL"] = model
    
    # Set analysis options
    os.environ["EXOMISER_COMPREHENSIVE"] = "true" if comprehensive else "false"
    os.environ["EXOMISER_USE_OMIM"] = "true" if omim else "false"
    os.environ["EXOMISER_USE_LITERATURE"] = "true" if literature else "false"
    
    try:
        from ascleon.agents.exomiser.exomiser_config import get_config
        from ascleon.utils.async_utils import run_sync
        
        # Create agent dependencies
        deps = get_config()
        
        click.echo(f"Analyzing {filename}...")
        if comprehensive:
            click.echo("Running comprehensive analysis")
            from ascleon.agents.exomiser.exomiser_tools import comprehensive_analysis
            result = run_sync(lambda: comprehensive_analysis.run_sync(deps=deps, exomiser_filename=filename))
            
            # Print phenotype information
            click.echo("\n=== Patient Information ===")
            pheno_data = result["original_data"]["phenotype_data"]
            click.echo(f"Patient ID: {pheno_data['id']}")
            click.echo("\nIncluded Phenotypes:")
            for p in pheno_data["included_phenotypes"]:
                click.echo(f"- {p['label']} ({p['id']})")
            
            click.echo("\nExcluded Phenotypes:")
            if pheno_data["excluded_phenotypes"]:
                for p in pheno_data["excluded_phenotypes"]:
                    click.echo(f"- {p['label']} ({p['id']})")
            else:
                click.echo("- None")
            
            # Print reranking results
            click.echo("\n=== Reranking Results ===")
            for item in result["reranking_results"]["reranked_list"]:
                click.echo(f"{item['rank']}. {item['description']}")
            
            click.echo("\n=== Reranking Explanation ===")
            click.echo(result["reranking_results"]["model_explanation"])
            
        else:
            from ascleon.agents.exomiser.exomiser_tools import perform_reranking
            result = run_sync(lambda: perform_reranking.run_sync(deps=deps, exomiser_filename=filename))
            
            # Print reranking results
            click.echo("\n=== Reranking Results ===")
            for item in result["reranking_results"]["reranked_list"]:
                click.echo(f"{item['rank']}. {item['description']}")
            
            click.echo("\n=== Reranking Explanation ===")
            click.echo(result["reranking_results"]["model_explanation"])
        
    except ImportError as e:
        click.echo(f"ERROR: Missing required modules. {str(e)}")
        click.echo("Make sure all dependencies are installed.")
    except Exception as e:
        click.echo(f"ERROR: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())

@main.command()
@click.option("--exomiser-path", help="Path to the directory containing Exomiser results", required=True)
def list(exomiser_path):
    """
    List all available Exomiser result files.
    
    This command shows all the TSV files in the Exomiser results directory
    that can be analyzed with the 'analyze' command.
    """
    os.environ["EXOMISER_RESULTS_PATH"] = exomiser_path
    
    try:
        from ascleon.agents.exomiser.exomiser_config import get_config
        from ascleon.agents.exomiser.exomiser_tools import list_exomiser_results
        from ascleon.utils.async_utils import run_sync
        
        # Create agent dependencies
        deps = get_config()
        
        click.echo(f"Looking for Exomiser results in: {exomiser_path}")
        results = run_sync(lambda: list_exomiser_results.run_sync(deps=deps))
        
        if results:
            click.echo("\nAvailable Exomiser result files:")
            for i, filename in enumerate(results, 1):
                click.echo(f"{i}. {filename}")
        else:
            click.echo("No Exomiser result files found.")
            
    except Exception as e:
        click.echo(f"ERROR: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())

if __name__ == "__main__":
    main()
