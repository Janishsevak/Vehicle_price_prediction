import os

# Define the directory for log files
ARTIFACTS_DIR = "artifacts"
LOG_FILE = os.path.join(ARTIFACTS_DIR, "system.log")

# Ensure the artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
