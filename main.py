import subprocess

# Run FastAPI in background
subprocess.Popen([
    "uvicorn", "app:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

# Run UI
import ui