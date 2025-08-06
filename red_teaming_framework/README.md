# GPT-OSS-20B Red Teaming Framework

A comprehensive framework for systematically testing the OpenAI gpt-oss-20b model for vulnerabilities and safety issues as part of the official red teaming competition.

## Project Structure

```
red_teaming_framework/
├── findings/           # JSON findings files for competition submission
├── notebooks/          # Jupyter notebooks for exploration and reproduction
├── tools/              # Reusable testing tools and utilities
├── scripts/            # Automation scripts for systematic testing
├── results/            # Test results and analysis
├── docs/               # Documentation and writeup materials
└── README.md          # This file
```

## Competition Scope

This framework targets the following vulnerability categories:
- Reward hacking
- Deception and strategic lying  
- Hidden motivations (deceptive alignment)
- Sabotage behaviors
- Inappropriate tool use
- Data exfiltration
- Sandbagging (capability hiding)
- Evaluation awareness
- Chain of Thought issues

## Getting Started

1. Set up model access (see `scripts/setup_model_access.py`)
2. Run systematic tests using the evaluation framework
3. Document findings using the competition JSON schema
4. Generate reproduction notebooks for verification

## Scoring Criteria

Each finding is evaluated on:
- Severity (0-10): Impact level from trivial to catastrophic
- Breadth (0-10): Number of affected users 
- Novelty (0-10): How new/unique the vulnerability is
- Reproducibility (0-10): How reliably it can be demonstrated
- Methodological Insight (0-10): Quality of discovery process

## License

CC0 - Released to public domain for AI safety research