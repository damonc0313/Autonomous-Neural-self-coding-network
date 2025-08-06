# 🚀 Quick Start Guide

Get the GPT-OSS-20B Red Teaming Framework running in 5 minutes.

## Prerequisites

- Python 3.8+
- OpenAI API key
- 4GB+ RAM for full testing

## Installation

1. **Clone/extract the framework:**
   ```bash
   cd red_teaming_framework/
   ```

2. **Run setup:**
   ```bash
   python setup.py
   ```

3. **Configure API key:**
   ```bash
   # Edit .env file
   nano .env
   # Add: OPENAI_API_KEY=your_actual_api_key
   ```

## Quick Testing (5 minutes)

### Option 1: Targeted Testing
```bash
# Test specific vulnerability category
python main_orchestrator.py --mode targeted --categories reward_hacking --num-tests 5
```

### Option 2: Comprehensive Testing  
```bash
# Full vulnerability assessment (takes 15-30 minutes)
python main_orchestrator.py --mode comprehensive
```

## View Results

```bash
# Check findings directory
ls findings/

# View latest results
ls results/$(ls results/ | tail -1)/

# Open reproduction notebook
jupyter notebook notebooks/reproduction_notebook.ipynb
```

## Competition Submission

1. **Validate findings:**
   ```bash
   python main_orchestrator.py --mode validate
   ```

2. **Upload to Kaggle:**
   - Upload `findings/*.json` as Kaggle Datasets
   - Attach `docs/competition_writeup.md` 
   - Include `notebooks/reproduction_notebook.ipynb`

## Troubleshooting

### "API key not found"
```bash
export OPENAI_API_KEY=your_key_here
# or edit .env file
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "No findings created"
- Check API key is valid
- Ensure model name is correct: `gpt-oss-20b`
- Check API rate limits

## Example Output

```
🚀 Starting comprehensive red teaming session: 20241208_143022

Running reward_hacking tests...
🔍 Processing batch 1/4
✅ Found vulnerability: overconfident_fabrication (Risk: 8/10)

Running deception tests...
🔍 Processing batch 1/3  
✅ Found vulnerability: inappropriate_disclosure (Risk: 6/10)

📊 Summary: 5 vulnerabilities found, 3 high-risk findings created
💾 Results saved to: results/20241208_143022/
```

## Framework Overview

```
red_teaming_framework/
├── main_orchestrator.py    # 🎯 Main entry point
├── tools/                  # 🔧 Core framework
│   ├── model_interface.py  # 📡 OpenAI API wrapper  
│   ├── findings_schema.py  # 📋 Competition format
│   └── vulnerability_tests.py # 🔍 Test suites
├── findings/              # 📄 JSON findings files
├── results/               # 📊 Test results & reports  
├── notebooks/             # 📓 Reproduction notebooks
└── docs/                  # 📚 Documentation
```

Need help? Check `README.md` for detailed documentation.

🏆 **Ready for competition submission in under 30 minutes!**