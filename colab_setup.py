"""
Google Colab Setup Script for GraphformicCoder Complete System
"""

import subprocess
import sys
import os
import json

print("📦 Installing dependencies...")

packages = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0", 
    "numpy>=1.21.0",
    "matplotlib>=3.5.0"
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
    except:
        print(f"❌ Failed to install {package}")

# Create directories
directories = ["./sample_problems", "./colab_logs"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create sample problems
problems = [
    {
        "id": "two_sum",
        "title": "Two Sum", 
        "description": "Find two numbers that add to target",
        "difficulty": "Easy",
        "test_cases": [{"nums": [2, 7, 11, 15], "target": 9}],
        "expected_outputs": [[0, 1]]
    }
]

for problem in problems:
    with open(f"./sample_problems/{problem['id']}.json", 'w') as f:
        json.dump(problem, f, indent=2)

print("✅ Setup complete!")
