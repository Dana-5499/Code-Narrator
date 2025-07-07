# Code-Narrator
Fine-tuned LLM to explain code snippets to non-technical managers.

# CodeSearchNet Explain

**Fine-tuning an LLM to turn code into human-readable explanations.**

## 📖 Overview
Non-technical managers often struggle to interpret code. We fine-tune an open-source LLM on CodeSearchNet’s `{code, docstring}` pairs—filtered for high-quality summaries—so you can ask “Explain this function” in plain English.

## 🗂 Repository Structure
- `data/` : instructions for obtaining and preprocessing the dataset  
- `src/` : Python scripts (preprocessing, filtering)  
- `notebooks/` : exploratory analyses  
- `docs/` : diagrams, slides  
- `requirements.txt` : dependencies  
- `LICENSE` : MIT license

## 🚀 Getting Started

1. **Clone** this repo:  
   ```bash
   git clone https://github.com/your-username/codesearchnet-explain.git
   cd codesearchnet-explain
