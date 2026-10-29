from pathlib import Path

#current working directory
root = Path(".")
#Folder to create the project in

folders =[
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static",
    "uploads",
    "tests",
]
file = [
    "app/main.py",
    "requirements.txt",
    "ingest_sample_kb.py",
    "run.py",
    ".env",

]
# Create the folders
for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)
# Create the files
for file_path in file:
    (root / file_path).touch(exist_ok=True)
    print("project structure created successfully.")