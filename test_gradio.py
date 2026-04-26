import gradio as gr
import os

with gr.Blocks() as demo:
    gr.Markdown("Hello")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=True)
