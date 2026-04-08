from pydantic import BaseModel
import random
import re

# ---------------- MODELS ----------------
class Observation(BaseModel):
    document_id: str
    raw_text: str
    extracted_data: dict
    step_count: int
    history: list


class Action(BaseModel):
    action_type: str
    payload: dict


# ---------------- HELPER ----------------
def extract_info(text):
    text = text.lower()

    data = {
        "amount": 0,
        "category": "other",
        "suspicious": False
    }

    if "laptop" in text:
        data["category"] = "electronics"
    elif "restaurant" in text:
        data["category"] = "food"
    elif "office" in text:
        data["category"] = "office"

    match = re.search(r"\d+", text)
    if match:
        data["amount"] = int(match.group())

    if data["amount"] > 40000 or "cash" in text:
        data["suspicious"] = True

    return data


# ---------------- ENV ----------------
class FinanceEnv:

    def __init__(self, level="easy"):
        self.level = level
        self.step_count = 0
        self.raw_text = ""
        self.truth = {}

    def reset(self):
        self.step_count = 0

        samples = [
            "Invoice from ABC Electronics for laptops worth INR 50000",
            "Restaurant bill for INR 800",
            "Cash withdrawal of INR 70000",
            "Office supplies purchase INR 3000"
        ]

        self.raw_text = random.choice(samples)
        self.truth = extract_info(self.raw_text)

        return Observation(
            document_id="doc1",
            raw_text=self.raw_text,
            extracted_data={},
            step_count=0,
            history=[]
        )

    def step(self, action: Action):
        self.step_count += 1
        reward = 0

        if action.action_type == "categorize":
            if action.payload.get("category") == self.truth["category"]:
                reward += 0.3

        elif action.action_type == "extract_amount":
            if action.payload.get("amount") == self.truth["amount"]:
                reward += 0.3

        elif action.action_type == "detect_fraud":
            if action.payload.get("fraud") == self.truth["suspicious"]:
                reward += 0.4

        done = self.step_count >= 3

        return (
            Observation(
                document_id="doc1",
                raw_text=self.raw_text,
                extracted_data=action.payload,
                step_count=self.step_count,
                history=[]
            ),
            reward,
            done,
            {}
        )