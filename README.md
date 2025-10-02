# Code-Narrator
Fine-Tuning Open-Source LLMs for Non-Technical Monitoring of Software Development Progress.

## 📖 Introduction
Software development consumes a large portion of company budgets, often exceeding 50% of total costs. Yet, non-technical managers frequently struggle to monitor real progress, as they must rely heavily on developer reports. This lack of visibility can lead to misallocation of resources, over-hiring, and missed risks.

Code Narrator bridges this gap by fine-tuning open-source large language models to generate clear, natural-language explanations of software activity. The model can describe both code snippets and commit changes in plain English, enabling managers and stakeholders to track project progress without requiring deep technical expertise.

## 🎯 Project Goal
Our goal was to create a support tool that leverages AI not to write more code, but to help companies understand and monitor existing development work. We focused on:
- Code summarization (explaining what code does).
- Commit-diff explanation (describing changes between code versions).
- Multi-task fine-tuning to unify both tasks in a single model.
