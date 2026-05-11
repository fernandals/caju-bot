import getpass
import os

# ============== LangSmith Variables ==============
if not os.environ.get("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

# ============== OpenAI Variables ==============
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

print("Environment variables set successfully!")

data_dir = "data"
data_dir.mkdir(exist_ok=True)

print("Data directory created successfully!")