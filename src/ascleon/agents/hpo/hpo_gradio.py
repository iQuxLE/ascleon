"""
Gradio interface for the HPO agent.
"""
import time
from pathlib import Path
from typing import Optional, Dict

import gradio as gr

from ascleon.agents.hpo.hpo_agent import hpo_agent, run_sync
from ascleon.agents.hpo.hpo_config import HPODependencies, get_config


def chat(
    model: Optional[str] = None,
    workdir: Optional[str] = None,
    chromadb_path: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> gr.Blocks:
    """Create a Gradio interface for the HPO agent.
    
    Args:
        model: The model to use for the agent
        workdir: The working directory for the agent
        chromadb_path: Path to the ChromaDB directory
        collection_name: The name of the ChromaDB collection to use
        
    Returns:
        gr.Blocks: A Gradio Blocks interface
    """
    # Get config
    deps = get_config()
    
    # Override with provided values
    if model:
        deps.model = model
    if workdir:
        deps.workdir.location = workdir
    if chromadb_path:
        deps.chromadb_path = chromadb_path
    if collection_name:
        deps.collection_name = collection_name
    
    # Create the chatbot
    with gr.Blocks(title="HPO Agent") as demo:
        gr.Markdown("# Human Phenotype Ontology (HPO) Agent")
        gr.Markdown("""
        This agent provides information about Human Phenotype Ontology (HPO) terms, 
        including definitions, relationships, and helps with finding the right terms 
        for phenotype descriptions.
        
        ## Example queries:
        - What is HP:0001250?
        - Search for phenotypes related to muscle weakness
        - What are the parent terms of HP:0001631?
        - Analyze these phenotypes: HP:0001250, HP:0001263, HP:0002360
        """)
        
        chatbot = gr.Chatbot(label="Chat", height=500)
        msg = gr.Textbox(label="Message", placeholder="Ask about HPO terms...")
        clear = gr.Button("Clear")
        
        def respond(message, history):
            """Respond to a message from the user."""
            # Add user message to history
            history.append((message, None))
            yield history
            
            # Run the agent
            start_time = time.time()
            try:
                result = run_sync(message, deps=deps)
                response = result.data
            except Exception as e:
                response = f"Error: {str(e)}"
            
            # Add timing information
            elapsed = time.time() - start_time
            response += f"\n\n_Response time: {elapsed:.2f} seconds using {deps.model}_"
            
            # Update history with response
            history[-1] = (message, response)
            yield history
        
        def clear_history():
            """Clear the chat history."""
            return None
        
        # Connect components
        msg.submit(respond, [msg, chatbot], [chatbot])
        clear.click(clear_history, None, chatbot)
        
        # Settings tab
        with gr.Accordion("Settings", open=False):
            gr.Markdown("## HPO Agent Settings")
            gr.Markdown(f"**Model:** {deps.model}")
            gr.Markdown(f"**ChromaDB Path:** {deps.chromadb_path}")
            gr.Markdown(f"**Collection:** {deps.collection_name}")
            
    return demo
