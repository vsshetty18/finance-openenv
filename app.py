from fastapi import FastAPI
from finance_env import FinanceEnv, Action
import gradio as gr
import re
import random

app = FastAPI()
env = FinanceEnv()

# ---------------- API ----------------
@app.get("/reset")
def reset(level: str = "easy"):
    global env
    env = FinanceEnv(level)
    return env.reset().dict()


@app.get("/demo")
def demo():
    env = FinanceEnv("hard")
    obs = env.reset()

    total_score = 0
    steps = []

    text = obs.raw_text.lower()

    # -------- Step 1: categorize --------
    if "laptop" in text:
        category = "electronics"
    elif "restaurant" in text:
        category = "food"
    elif "office" in text:
        category = "office"
    else:
        category = "other"

    obs, r, _, _ = env.step(Action(
        action_type="categorize",
        payload={"category": category}
    ))
    total_score += r
    steps.append({"step": "categorize", "reward": r})

    # -------- Step 2: extract amount --------
    match = re.search(r"\d+", obs.raw_text)
    amount = int(match.group()) if match else 0

    obs, r, _, _ = env.step(Action(
        action_type="extract_amount",
        payload={"amount": amount}
    ))
    total_score += r
    steps.append({"step": "extract_amount", "reward": r})

    # -------- Step 3: fraud detection --------
    actual_fraud = amount > 40000 or "cash" in text

    # Add slight randomness (realistic AI behavior)
    predicted_fraud = actual_fraud if random.random() > 0.2 else not actual_fraud

    obs, r, _, _ = env.step(Action(
        action_type="detect_fraud",
        payload={"fraud": predicted_fraud}
    ))
    total_score += r
    steps.append({"step": "detect_fraud", "reward": r})

    return {
        "document": obs.raw_text,
        "steps": steps,
        "final_score": total_score
    }


# ---------------- UI ----------------
def ui_reset(level):
    return reset(level)


def ui_demo():
    return demo()


with gr.Blocks() as ui:
    gr.Markdown("# 💰 Finance OpenEnv Demo")
    gr.Markdown("AI Agent for Financial Document Analysis")

    level = gr.Dropdown(["easy", "medium", "hard"], value="easy")

    btn1 = gr.Button("🔄 Reset Environment")
    btn2 = gr.Button("🚀 Run Demo")

    output = gr.JSON()

    btn1.click(ui_reset, inputs=level, outputs=output)
    btn2.click(ui_demo, inputs=None, outputs=output)


# Mount UI into FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = mount_gradio_app(app, ui, path="/")