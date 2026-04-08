import gradio as gr
import requests
import os

# ✅ Get HF Space URL automatically
BASE_URL = "http://localhost:8000"

def get_full_url(path):
    return f"https://{BASE_URL}{path}"

def reset_env(level):
    try:
        url = get_full_url(f"/reset?level={level}")
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_demo():
    try:
        url = get_full_url("/demo")
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks() as app:
    gr.Markdown("# 💰 Finance OpenEnv Demo")
    gr.Markdown("AI Agent for Financial Document Analysis")

    level = gr.Dropdown(
        choices=["easy", "medium", "hard"],
        value="easy",
        label="Select Task Difficulty"
    )

    reset_btn = gr.Button("🔄 Reset Environment")
    demo_btn = gr.Button("🚀 Run Demo")

    output = gr.JSON(label="Output")

    reset_btn.click(fn=reset_env, inputs=level, outputs=output)
    demo_btn.click(fn=run_demo, inputs=None, outputs=output)

app.launch(server_name="0.0.0.0", server_port=7860)