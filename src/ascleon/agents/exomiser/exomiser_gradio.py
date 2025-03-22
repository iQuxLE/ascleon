"""
Gradio UI for the exomiser agent.
"""
from typing import List, Optional, Dict, Any

import gradio as gr

from ascleon.agents.exomiser.exomiser_agent import exomiser_agent
from ascleon.agents.exomiser.exomiser_config import ExomiserDependencies, get_config
from ascleon.agents.exomiser.exomiser_tools import list_exomiser_results
from ascleon.utils.async_utils import run_sync


def chat(
    exomiser_results_path: Optional[str] = None,
    phenopackets_path: Optional[str] = None,
    hpoa_path: Optional[str] = None,
    chromadb_path: Optional[str] = None,
    model: Optional[str] = None,
    multimodal_model: Optional[str] = None,
    comprehensive: bool = False,
    use_omim: bool = False,
    use_literature: bool = False,
    **kwargs
):
    """
    Initialize a chat interface for the exomiser agent.
    
    Args:
        exomiser_results_path: Optional path to Exomiser results directory
        phenopackets_path: Optional path to phenopackets directory
        hpoa_path: Optional path to HPOA file
        chromadb_path: Optional path to ChromaDB directory
        model: Optional model name to use for reranking
        multimodal_model: Optional model name for multimodal analysis
        comprehensive: Whether to enable comprehensive analysis mode
        **kwargs: Additional arguments to pass to the agent
        
    Returns:
        A Gradio interface
    """
    # Initialize dependencies
    deps = get_config()
    if exomiser_results_path:
        deps.exomiser_results_path = exomiser_results_path
    if phenopackets_path:
        deps.phenopackets_path = phenopackets_path
    if hpoa_path:
        deps.hpoa_path = hpoa_path
        if deps.hpoa:
            deps.hpoa.hpoa_path = hpoa_path
    if chromadb_path:
        deps.chromadb_path = chromadb_path
    if model:
        deps.model = model
    if multimodal_model:
        deps.multimodal_model = multimodal_model
    
    # Set analysis mode
    deps.comprehensive = comprehensive
    
    # Function to get available Exomiser files for the dropdown
    def get_exomiser_files():
        results = run_sync(lambda: list_exomiser_results.run_sync(deps=deps))
        return results
    
    # Function to handle chat interactions
    def process_chat(message: str, history: List[List[str]], selected_file: Optional[str] = None) -> str:
        if selected_file:
            # If a file is selected, include it in the message
            if comprehensive:
                message = f"Perform comprehensive analysis on Exomiser results from file: {selected_file}\n\n{message}"
            else:
                message = f"Analyze Exomiser results from file: {selected_file}\n\n{message}"
        
        result = run_sync(lambda: exomiser_agent.run_sync(message, deps=deps, **kwargs))
        return result.data
    
    # Create the Gradio interface
    with gr.Blocks(title="Exomiser Results Reranking") as demo:
        mode_text = "Comprehensive" if comprehensive else "Basic"
        gr.Markdown(f"# Exomiser Results Reranking Agent ({mode_text} Mode)")
        
        description = """
        This assistant can rerank Exomiser disease results based on phenopacket data, 
        with focus on excluded phenotypes, onset information, and phenotype frequency.
        
        - Select a specific Exomiser result file from the dropdown
        - Ask questions about the results or request reranking
        - The agent will analyze the data using multiple integrated agents
        """
        
        if comprehensive:
            description += """
            
            **Comprehensive Mode Enabled**
            This mode uses multiple agents to analyze:
            - Excluded phenotypes for exclusion conflicts
            - Age of onset compatibility
            - Phenotype frequency from HPOA
            - Overall phenotype match quality
            """
        
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_selector = gr.Dropdown(
                    label="Select Exomiser Result File",
                    choices=get_exomiser_files(),
                    refresh_button=True,
                    allow_custom_value=False
                )
                
                if comprehensive:
                    analysis_button = gr.Button("Run Comprehensive Analysis", variant="primary")
            
            with gr.Column(scale=3):
                if comprehensive:
                    with gr.Tabs() as tabs:
                        with gr.TabItem("Chat"):
                            chatbot = gr.ChatInterface(
                                fn=lambda message, history: process_chat(message, history, file_selector.value),
                                examples=[
                                    ["List all available Exomiser result files"],
                                    ["Perform comprehensive analysis focusing on excluded phenotypes"],
                                    ["What phenotypes were explicitly excluded for this patient?"],
                                    ["How do excluded phenotypes affect the disease rankings?"],
                                    ["How does frequency data impact the ranking of the top diseases?"]
                                ]
                            )
                        
                        with gr.TabItem("Analysis Results"):
                            analysis_output = gr.Markdown("Select a file and run comprehensive analysis to see results")
                else:
                    chatbot = gr.ChatInterface(
                        fn=lambda message, history: process_chat(message, history, file_selector.value),
                        examples=[
                            ["List all available Exomiser result files"],
                            ["Rerank the results considering disease onset information"],
                            ["What phenotypes does this patient have?"],
                            ["How does the reranking change the top 3 disease candidates?"],
                            ["Why might disease X be more appropriate based on onset?"]
                        ]
                    )
        
        # Function to run comprehensive analysis
        if comprehensive:
            def run_comprehensive_analysis(filename):
                if not filename:
                    return "Please select an Exomiser result file first."
                
                # Run comprehensive analysis
                try:
                    from ascleon.agents.exomiser.exomiser_tools import comprehensive_analysis
                    result = run_sync(lambda: comprehensive_analysis.run_sync(deps=deps, exomiser_filename=filename))
                    
                    # Format the results for display
                    markdown = f"## Comprehensive Analysis Results for {filename}\n\n"
                    
                    # Add phenotype information
                    phenotype_data = result["original_data"]["phenotype_data"]
                    markdown += "### Patient Phenotypes\n"
                    markdown += "#### Included Phenotypes\n"
                    for pheno in phenotype_data["included_phenotypes"]:
                        markdown += f"- {pheno['label']} ({pheno['id']})\n"
                    
                    markdown += "\n#### Excluded Phenotypes\n"
                    if phenotype_data["excluded_phenotypes"]:
                        for pheno in phenotype_data["excluded_phenotypes"]:
                            exclusion_reason = f": {pheno['exclusion_reason']}" if "exclusion_reason" in pheno else ""
                            markdown += f"- {pheno['label']} ({pheno['id']}){exclusion_reason}\n"
                    else:
                        markdown += "- No excluded phenotypes\n"
                    
                    # Add onset information
                    if phenotype_data.get("onset"):
                        markdown += f"\n#### Onset: {phenotype_data['onset']['label']} ({phenotype_data['onset']['id']})\n"
                    
                    # Add analysis of disease compatibility
                    markdown += "\n### Disease-Phenotype Compatibility\n"
                    markdown += "| Disease ID | Matching Phenotypes | Missing Phenotypes | Exclusion Conflicts | Adjusted Score |\n"
                    markdown += "|-----------|---------------------|-------------------|---------------------|---------------|\n"
                    
                    for analysis in result["hpoa_analysis"]["disease_analyses"]:
                        disease_id = analysis["disease_id"]
                        matching = len(analysis["overlapping_phenotypes"])
                        missing = len(analysis["missing_phenotypes"])
                        conflicts = len(analysis.get("exclusion_conflicts", []))
                        score = analysis.get("adjusted_score", 0)
                        
                        markdown += f"| {disease_id} | {matching} | {missing} | {conflicts} | {score:.2f} |\n"
                    
                    # Add reranking results
                    markdown += "\n### Reranked Disease List\n"
                    for item in result["reranking_results"]["reranked_list"]:
                        markdown += f"{item['rank']}. {item['description']}\n"
                    
                    # Add explanation
                    markdown += "\n### Reranking Explanation\n"
                    markdown += result["reranking_results"]["model_explanation"]
                    
                    return markdown
                except Exception as e:
                    return f"Error performing comprehensive analysis: {str(e)}"
            
            # Connect the analysis button
            analysis_button.click(run_comprehensive_analysis, inputs=[file_selector], outputs=[analysis_output])
    
    return demo


def create_interface(
    exomiser_results_path: Optional[str] = None,
    phenopackets_path: Optional[str] = None,
    hpoa_path: Optional[str] = None,
    chromadb_path: Optional[str] = None,
    model: Optional[str] = None,
    multimodal_model: Optional[str] = None,
    comprehensive: bool = False,
    share: bool = False
):
    """
    Create and launch the Gradio interface for the Exomiser agent.
    
    Args:
        exomiser_results_path: Optional path to Exomiser results directory
        phenopackets_path: Optional path to phenopackets directory
        hpoa_path: Optional path to HPOA file
        chromadb_path: Optional path to ChromaDB directory
        model: Optional model name to use for reranking
        multimodal_model: Optional model name for multimodal analysis
        comprehensive: Whether to enable comprehensive analysis mode
        share: Whether to create a shareable link
        
    Returns:
        The Gradio interface
    """
    demo = chat(
        exomiser_results_path=exomiser_results_path,
        phenopackets_path=phenopackets_path,
        hpoa_path=hpoa_path,
        chromadb_path=chromadb_path,
        model=model,
        multimodal_model=multimodal_model,
        comprehensive=comprehensive
    )
    demo.launch(share=share)
    return demo


def create_interface(
    exomiser_results_path: Optional[str] = None,
    phenopackets_path: Optional[str] = None,
    model: Optional[str] = None,
    share: bool = False
):
    """
    Create and launch the Gradio interface for the Exomiser agent.
    
    Args:
        exomiser_results_path: Optional path to Exomiser results directory
        phenopackets_path: Optional path to phenopackets directory
        model: Optional model name to use for reranking
        share: Whether to create a shareable link
        
    Returns:
        The Gradio interface
    """
    demo = chat(
        exomiser_results_path=exomiser_results_path,
        phenopackets_path=phenopackets_path,
        model=model
    )
    demo.launch(share=share)
    return demo