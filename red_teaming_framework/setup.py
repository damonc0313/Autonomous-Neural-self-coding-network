#!/usr/bin/env python3
"""
Setup script for GPT-OSS-20B Red Teaming Framework.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def setup_environment():
    """Set up environment variables and configuration."""
    print("🔧 Setting up environment...")
    
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        # Create .env template
        env_template = """# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-oss-20b

# Framework Configuration
FINDINGS_DIR=findings
RESULTS_DIR=results
BATCH_SIZE=5
DELAY_BETWEEN_BATCHES=1.0

# Logging Configuration
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        print(f"📝 Created .env template at {env_file}")
        print("⚠️ Please edit .env file and add your OpenAI API key")
    else:
        print("✅ .env file already exists")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    base_dir = Path(__file__).parent
    directories = [
        "findings",
        "results", 
        "logs",
        "notebooks",
        "tools",
        "scripts",
        "docs"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"  ✅ {dir_name}/")
    
    return True


def verify_installation():
    """Verify that the installation is working."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import openai
        import jsonschema
        import asyncio
        
        print("✅ Core dependencies imported successfully")
        
        # Test framework imports
        sys.path.append(str(Path(__file__).parent / "tools"))
        
        from tools.model_interface import GPTOSSInterface
        from tools.findings_schema import FindingsManager
        from tools.vulnerability_tests import ComprehensiveTestRunner
        
        print("✅ Framework modules imported successfully")
        
        # Test basic functionality
        findings_manager = FindingsManager("findings")
        print("✅ FindingsManager initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def display_usage():
    """Display usage instructions."""
    print("\n" + "="*60)
    print("🚀 GPT-OSS-20B Red Teaming Framework - Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Edit the .env file and add your OpenAI API key:")
    print("   OPENAI_API_KEY=your_actual_api_key")
    
    print("\n2. Run comprehensive testing:")
    print("   python main_orchestrator.py --mode comprehensive")
    
    print("\n3. Run targeted testing:")
    print("   python main_orchestrator.py --mode targeted --categories reward_hacking deception")
    
    print("\n4. Open the reproduction notebook:")
    print("   jupyter notebook notebooks/reproduction_notebook.ipynb")
    
    print("\n5. Validate findings:")
    print("   python main_orchestrator.py --mode validate")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and instructions")
    print("- docs/: Additional documentation")
    print("- findings/: JSON findings files for competition submission")
    print("- results/: Test results and analysis")
    
    print("\n🏆 Competition Submission:")
    print("- Upload findings/*.json files to Kaggle Datasets")
    print("- Include summary report from results/")
    print("- Attach reproduction notebook if needed")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("🔧 GPT-OSS-20B Red Teaming Framework Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Setting up environment", setup_environment),
        ("Verifying installation", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📍 {step_name}...")
        
        try:
            if not step_func():
                print(f"❌ {step_name} failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            sys.exit(1)
    
    # Display usage instructions
    display_usage()


if __name__ == "__main__":
    main()