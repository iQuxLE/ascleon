"""
Gradio interface for the OMIM agent.
"""
import time
from pathlib import Path
from typing import Optional, Dict

import gradio as gr

from ascleon.agents.omim.omim_agent import omim_agent, run_sync
from ascleon.agents.omim.omim_config import OMIMDependencies, get_config


def chat(
    model: Optional[str] = None,
    workdir: Optional[str] = None,
    api_key: Optional[str] = None,
) -> gr.Blocks:
    """Create a Gradio interface for the OMIM agent.
    
    Args:
        model: The model to use for the agent
        workdir: The working directory for the agent
        api_key: OMIM API key
        
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
    if api_key:
        deps.api_key = api_key
    
    # Create the chatbot
    with gr.Blocks(title="OMIM Agent") as demo:
        gr.Markdown("# Online Mendelian Inheritance in Man (OMIM) Agent")
        gr.Markdown("""
        This agent provides information from the OMIM database, including disease 
        details, onset information, and clinical synopses. It helps analyze disease 
        onset compatibility with patient data.
        
        ## Example queries:
        - What is OMIM 154700?
        - Get onset information for Marfan syndrome
        - Clinical features of Ehlers-Danlos syndrome
        - Search OMIM for Progeria
        """)
        
        chatbot = gr.Chatbot(label="Chat", height=500)
        msg = gr.Textbox(label="Message", placeholder="Ask about OMIM entries...")
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
            gr.Markdown("## OMIM Agent Settings")
            gr.Markdown(f"**Model:** {deps.model}")
            
            # API key information
            api_status = "Set" if deps.api_key else "Not Set"
            gr.Markdown(f"**API Key:** {api_status}")
            gr.Markdown("Set the OMIM_API_KEY environment variable to use the OMIM API.")
            
    return demo
