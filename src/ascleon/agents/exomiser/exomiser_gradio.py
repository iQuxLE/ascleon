"""
Gradio UI for the exomiser agent.
"""
from typing import List, Optional, Dict, Any

import gradio as gr

from ascleon.agents.exomiser.exomiser_agent import exomiser_agent
from ascleon.agents.exomiser.exomiser_config import ExomiserDependencies, get_config
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
    selected_file: Optional[str] = None,
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
        use_omim: Whether to enable OMIM integration for onset analysis
        use_literature: Whether to enable literature analysis
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
    deps.use_omim = use_omim
    deps.use_literature = use_literature
    
    # Function to get available Exomiser files for the dropdown
    def get_exomiser_files():
        # Skip API call if we already have a pre-selected file
        if selected_file:
            return [selected_file]
            
        try:
            # Use the agent to list the files
            result = run_sync(lambda: exomiser_agent.run_sync("List all available Exomiser result files", deps=deps))
            return result.data
        except Exception as e:
            print(f"Error listing Exomiser files: {str(e)}")
            # If there's an error, return a small list of common files
            return ["PMID_9312167_BII4-pheval_disease_result.tsv", "Please refresh when API connection is working"]
    
    # Function to handle chat interactions
    def process_chat(message: str, history: List[List[str]], selected_file: Optional[str] = None) -> str:
        if selected_file:
            # If a file is selected, include it in the message
            if comprehensive:
                message = f"Perform comprehensive analysis on Exomiser results from file: {selected_file}\n\n{message}"
            else:
                message = f"Analyze Exomiser results from file: {selected_file}\n\n{message}"
        
        # Use the agent without specifying a model (uses the default from exomiser_agent.py)
        result = run_sync(lambda: exomiser_agent.run_sync(message, deps=deps))
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
            
            if use_omim:
                description += "- OMIM onset information\n"
            if use_literature:
                description += "- Literature-based diagnostic test recommendations\n"
        
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                # If a pre-selected file is provided, use it, otherwise get all files
                initial_choice = selected_file if selected_file else None
                initial_choices = [selected_file] if selected_file else get_exomiser_files()
                
                # Create dropdown without refresh_button which might not be supported in this Gradio version
                file_selector = gr.Dropdown(
                    label="Select Exomiser Result File",
                    choices=initial_choices,
                    value=initial_choice,
                    allow_custom_value=False
                )
                
                if comprehensive:
                    analysis_button = gr.Button("Run Comprehensive Analysis", variant="primary")
            
            with gr.Column(scale=3):
                if comprehensive:
                    with gr.Tabs() as tabs:
                        with gr.TabItem("Chat"):
                            # Setup initial message that would automatically analyze the file if it's pre-selected
                            initial_message = ""
                            if selected_file:
                                if comprehensive:
                                    initial_message = f"Perform comprehensive analysis on Exomiser file: {selected_file}"
                                else:
                                    initial_message = f"Analyze Exomiser file: {selected_file}"
                            
                            # Create chat interface without value parameter which might not be supported
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
                            
                            # If a file was pre-selected, add a message to the interface prompting analysis
                            if initial_message:
                                gr.Markdown(f"**Auto-analysis will start when you click 'Submit'**: {initial_message}")
                        
                        with gr.TabItem("Analysis Results"):
                            analysis_output = gr.Markdown("Select a file and run comprehensive analysis to see results")
                else:
                    # Setup initial message that would automatically analyze the file if it's pre-selected
                    initial_message = ""
                    if selected_file:
                        initial_message = f"Analyze Exomiser file: {selected_file}"
                    
                    chatbot = gr.ChatInterface(
                        fn=lambda message, history: process_chat(message, history, file_selector.value),
                        examples=[
                            ["List all available Exomiser result files"],
                            ["Rerank the results considering disease onset information"],
                            ["What phenotypes does this patient have?"],
                            ["How does the reranking change the top 3 disease candidates?"],
                            ["Why might disease X be more appropriate based on onset?"]
                        ]
                        # Remove value parameter which is not supported in older Gradio versions
                    )
                    
                    # If a file was pre-selected, add guidance message
                    if initial_message:
                        gr.Markdown(f"**To analyze {selected_file}, click Submit with your first message or type your question and submit**")
        
        # Function to run comprehensive analysis
        if comprehensive:
            def run_comprehensive_analysis(filename):
                if not filename:
                    return "Please select an Exomiser result file first."
                
                # Run comprehensive analysis using the agent instead of direct function call
                try:
                    result = run_sync(lambda: exomiser_agent.run_sync(
                        f"Perform comprehensive analysis on Exomiser file: {filename}",
                        deps=deps
                    ))
                    
                    # The agent will return a structured result
                    analysis_data = result.data
                    
                    # Format the results for display
                    markdown = f"## Comprehensive Analysis Results for {filename}\n\n"
                    
                    # Check if we have original data with phenotype information
                    if "original_data" in analysis_data and "phenotype_data" in analysis_data["original_data"]:
                        phenotype_data = analysis_data["original_data"]["phenotype_data"]
                        
                        markdown += "### Patient Phenotypes\n"
                        markdown += "#### Included Phenotypes\n"
                        for pheno in phenotype_data["included_phenotypes"]:
                            markdown += f"- {pheno['label']} ({pheno['id']})\n"
                        
                        markdown += "\n#### Excluded Phenotypes\n"
                        if phenotype_data.get("excluded_phenotypes"):
                            for pheno in phenotype_data["excluded_phenotypes"]:
                                exclusion_reason = f": {pheno.get('exclusion_reason', '')}" if pheno.get("exclusion_reason") else ""
                                markdown += f"- {pheno['label']} ({pheno['id']}){exclusion_reason}\n"
                        else:
                            markdown += "- No excluded phenotypes\n"
                        
                        # Add onset information
                        if phenotype_data.get("onset"):
                            markdown += f"\n#### Onset: {phenotype_data['onset']['label']} ({phenotype_data['onset']['id']})\n"
                    
                    # Add analysis of disease compatibility if available
                    if "hpoa_analysis" in analysis_data and "disease_analyses" in analysis_data["hpoa_analysis"]:
                        markdown += "\n### Disease-Phenotype Compatibility\n"
                        markdown += "| Disease ID | Matching Phenotypes | Missing Phenotypes | Exclusion Conflicts | Adjusted Score |\n"
                        markdown += "|-----------|---------------------|-------------------|---------------------|---------------|\n"
                        
                        for analysis in analysis_data["hpoa_analysis"]["disease_analyses"]:
                            disease_id = analysis["disease_id"]
                            matching = len(analysis["overlapping_phenotypes"])
                            missing = len(analysis["missing_phenotypes"])
                            conflicts = len(analysis.get("exclusion_conflicts", []))
                            score = analysis.get("adjusted_score", 0)
                            
                            markdown += f"| {disease_id} | {matching} | {missing} | {conflicts} | {score:.2f} |\n"
                    
                    # Check if we have reranking results
                    if "reranking_results" in result.data:
                        reranking = result.data["reranking_results"]
                        
                        # Add reranking results
                        markdown += "\n### Reranked Disease List\n"
                        
                        if "reranked_list" in reranking:
                            for item in reranking["reranked_list"]:
                                markdown += f"{item['rank']}. {item['description']}\n"
                        
                        # Add explanation if available
                        if "model_explanation" in reranking:
                            markdown += "\n### Reranking Explanation\n"
                            markdown += reranking["model_explanation"]
                    
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
    use_omim: bool = False,
    use_literature: bool = False,
    selected_file: Optional[str] = None,
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
        use_omim: Whether to enable OMIM integration
        use_literature: Whether to enable literature analysis
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
        comprehensive=comprehensive,
        use_omim=use_omim,
        use_literature=use_literature,
        selected_file=selected_file
    )
    demo.launch(share=share)
    return demo