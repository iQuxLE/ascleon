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
    
    # Function to get available Exomiser files for the dropdown without using the agent
    def get_exomiser_files():
        # Skip file listing if we already have a pre-selected file
        if selected_file:
            return [selected_file]
            
        try:
            # Directly list files from the filesystem
            from pathlib import Path
            
            # Find in the exomiser path
            results_path = Path(deps.exomiser_results_path)
            
            # Check for pheval_disease_result or pheval_disease_results directories
            pheval_paths = [
                results_path / "pheval_disease_result",
                results_path / "pheval_disease_results"
            ]
            
            # Try each potential pheval directory
            tsv_files = []
            
            for pheval_path in pheval_paths:
                if pheval_path.exists():
                    files = [f.name for f in pheval_path.glob("**/*.tsv") if f.is_file()]
                    if files:
                        print(f"Found {len(files)} files in {pheval_path}")
                        tsv_files = files
                        break
            
            # If no pheval directories found or they're empty, look in base directory
            if not tsv_files:
                files = [f.name for f in results_path.glob("**/*.tsv") if f.is_file()]
                if files:
                    print(f"Found {len(files)} files in {results_path}")
                    tsv_files = files
            
            if tsv_files:
                # Limit to first 500 files if very large to prevent UI issues
                if len(tsv_files) > 500:
                    tsv_files = tsv_files[:500]
                    print(f"Limited display to first 500 files")
                return tsv_files
            else:
                return ["No Exomiser result files found"]
                
        except Exception as e:
            print(f"Error listing Exomiser files: {str(e)}")
            # If there's an error, return a small list of common files
            return ["PMID_9312167_BII4-pheval_disease_result.tsv", "Error listing files, please check paths"]
    
    # Function to handle chat interactions
    def process_chat(message: str, history: List[List[str]], selected_file: Optional[str] = None) -> str:
        # First message and a file is selected - automatically do reranking
        if not history and selected_file:
            # Check if the user's first message is empty or just asking for help
            if not message.strip() or message.strip().lower() in ["help", "hello", "hi"]:
                # Create an automatic reranking request for the selected file
                if comprehensive:
                    prompt = f"Perform comprehensive analysis on Exomiser results from file: {selected_file}" 
                else:
                    prompt = f"Please rerank the Exomiser results from file: {selected_file}"
                    
                print(f"Auto-generating analysis request for {selected_file}")
                result = run_sync(lambda: exomiser_agent.run_sync(prompt, deps=deps))
                return result.data
        
        # For subsequent messages or if user provided specific instructions
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
        This assistant can rerank Exomiser disease results to improve diagnostic accuracy.
        
        ### How to use:
        1. Select a specific Exomiser result file from the dropdown below
        2. Just click Submit with an empty message to automatically rerank the results
        3. Or ask specific questions about the results for more detailed analysis
        
        The analysis focuses on:
        - Disease-phenotype compatibility
        - Excluded phenotypes (when available)
        - Age of onset information
        - Phenotype frequency data
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
                            
                            # Add guidance for submitting analysis
                            if initial_message:
                                gr.Markdown(f"""
                                ### ✅ Auto-analysis ready
                                
                                File "{selected_file}" is selected. Just click **Submit** to analyze it, 
                                or type a specific question and then Submit.
                                """)
                        
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
                    
                    # Add guidance for submitting analysis
                    if initial_message:
                        gr.Markdown(f"""
                        ### ✅ Auto-analysis ready
                        
                        File "{selected_file}" is selected. Just click **Submit** to analyze it, 
                        or type a specific question and then Submit.
                        """)
        
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