# Code Narrator: Bridging non-technical managers and code with fine-tuned LLMs

This project aims to develop an AI-powered assistant for monitoring software development progress by fine-tuning open-source large language models. Instead of generating more code, the model is trained to summarize existing code snippets and explain commit differences in clear, plain language. Using multi-task fine-tuning on CodeSearchNet and CommitPackFT datasets, combined with LoRA adapters and 4-bit quantization, the system provides non-technical managers and stakeholders with concise, human-readable insights into developer activity and project progress.
![Presentation1](https://github.com/user-attachments/assets/a6a9b447-8958-4e4b-a2e7-c3ad1bda36a0)

---

## Table of Contents  

1. [Overview](#overview)  
2. [How To Run](#how-to-run)  
3. [Datasets](#datasets)  
   - 3.1 CodeSearchNet  
   - 3.2 CommitPackFT  
4. [Training Approach](#training-approach)  
5. [Model & Techniques](#model--techniques)  
6. [Experiments and Results](#experiments-and-results)  
   - 6.1 Baseline Comparisons  
   - 6.2 Hyperparameter Tuning
   - 6.3 Transfer to CommitPackFT 
   - 6.4 Cocktail Training (50/50)  
7. [Conclusions](#conclusions)  
8. [Future Work](#future-work)  
9. [References](#references)  

---

## Overview  
Software development consumes a large portion of company budgets, often exceeding 50% of total costs. Yet, non-technical managers frequently struggle to monitor real progress, as they must rely heavily on developer reports. This lack of visibility can lead to misallocation of resources, over-hiring, and missed risks.

Code Narrator bridges this gap by fine-tuning open-source large language models to generate clear, natural-language explanations of software activity. The model can describe both code snippets and commit changes in plain English, enabling managers and stakeholders to track project progress without requiring deep technical expertise.

Current LLM projects mostly aim to help developers write code faster. Our approach is different:  
we fine-tune open-source models to explain software development progress in plain language.  

This allows non-technical managers and stakeholders to:  
- Understand current project functionality and progress
- Track recent changes and commits
- Improve communication with development teams

---

## How To Run 

### Files Explanation

- **Data Sets**
  - `CodeSearchNet.py` – Preprocessing pipeline for the CodeSearchNet dataset (filtering, cleaning docstrings, preparing splits).
  - `CommitPackFT.py` – Preprocessing pipeline for the CommitPackFT dataset (commit message + subject cleaning, before/after code).
  - `Cocktail.py` – Builds the 50/50 mixed dataset (CodeSearchNet + CommitPackFT) for multi-task training.

- **Training**
  - `ComparingModelsAndDeepSeek.py` – Baseline comparison across multiple models.
  - `ComparingLanguagesAndOverFit.py` – Language-level baselines and overfitting check on small subset.
  - `CheckHyperParametersOnCommit.py` – Hyperparameter evaluation on CommitPackFT dataset.
  - `HyperParametersAndTestOnCodeSearchNet.py` – Hyperparameter tuning and validation on CodeSearchNet.
  - `FinalModelTrainingAndTest.py` – Full training on the cocktail dataset (CodeSearchNet + CommitPackFT), evaluation, and test reporting.

- **Figures**
  - Contains all generated plots and tables used for reporting and README visualizations (validation loss, baseline comparisons, cocktail results, etc.).

### Steps to Run

1. **Preprocess datasets**
   - Run `Data Sets/CodeSearchNet.py` to prepare CodeSearchNet.
   - Run `Data Sets/CommitPackFT.py` to prepare CommitPackFT.
   - Run `Data Sets/Cocktail.py` to generate the mixed dataset.

2. **Baseline evaluation**
   - Run `Training/ComparingModelsAndDeepSeek.py` for initial model comparison.
   - Run `Training/ComparingLanguagesAndOverFit.py` to check baselines by language and verify overfitting sanity checks.

3. **Hyperparameter tuning**
   - Run `Training/HyperParametersAndTestOnCodeSearchNet.py` for CodeSearchNet sweeps.
   - Run `Training/CheckHyperParametersOnCommit.py` to validate hyperparameters on CommitPackFT.

4. **Final training and evaluation**
   - Run `Training/FinalModelTrainingAndTest.py` for cocktail fine-tuning and final evaluation across all datasets.

---

## Datasets  

### 3.1 CodeSearchNet  
- Covers 6 languages: Python, Java, JavaScript, PHP, Ruby, Go  
- ~2M examples → compacted to 2,100 train / 600 test / 300 val per language  
- **Preprocessing**:  
  - Removed auto-generated docstrings (e.g., *Doxygen style* like `:param x:`)  
  - Filtered docstrings (≥30 chars, capitalized, valid punctuation, no generic "See/Refer")  
  - Deduplicated samples, enforced length ranges  

### 3.2 CommitPackFT  
- Derived from CommitPack (350+ languages)  
- Filtered to the same 6 languages for consistency  
- Fields kept: `old_code`, `new_code`, `language`, and combined `subject+message`  
- Stratified 70–10–20 splits per language  
- Balanced compact subsets - 2,100 train / 600 test / 300 val per language

---

## Training Approach  

Our method involved:  

- **Baseline comparison** across multiple models and languages  
- **Hyperparameter sweeps** (epochs, batch size, optimizers, schedulers)  
- **Transfer of tuned parameters** from CodeSearchNet → CommitPackFT  
- **Cocktail multi-task training** with a 50/50 mix of both datasets

<p align="center">
  <img src="https://github.com/user-attachments/assets/afdbf650-c1ca-44d0-b1cd-93ebc7d9962b" alt="Figure 1" width="400"/>
</p>

<p align="center">
  <em>Figure 1 – Overview of our training pipeline</em>
</p>
 
---

## Model & Techniques  

- **Base Model**: We selected DeepSeek-Coder 1.3B Instruct, a Transformer-based LLM specialized for code.
- **Quantization**: Applied NF4 4-bit quantization to reduce memory usage and support training on limited resources.
- **LoRA Fine-Tuning**: Used Low-Rank Adaptation (LoRA) applied to attention and projection layers, reducing trainable parameters while maintaining performance.
- **Loss Function**: Training used Cross-Entropy (CE) loss, the standard for next-token prediction in LLMs.  

---

## Experiments and Results  

### 6.1 Baseline Comparisons 
To establish a baseline, we evaluated **DeepSeek-Coder-1.3B** on three datasets:  
- **CodeSearchNet** (docstring-based summaries)  
- **CommitPackFT** (commit-diff explanations)  
- A **50/50 cocktail mix** of both  

Results show that CodeSearchNet achieved the lowest validation cross-entropy, reflecting its relative simplicity and consistency. CommitPackFT had substantially higher error, highlighting the difficulty of modeling natural commit messages. The cocktail dataset fell in between, demonstrating how combining both sources increases task diversity while providing more stable performance than CommitPackFT alone.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d6228f76-6d26-4200-82a5-13c4d800d005" alt="Figure 1" width="500"/>
</p>

<p align="center">
  <em>Figure 2 – Baseline validation cross-entropy across datasets</em>
</p>

### 6.2 Hyperparameter Tuning  
Before finalizing our training setup, we systematically explored a range of hyperparameters to identify the most effective configuration.  
We tested:  
- **Optimizers**: AdamW, Adagrad, SGD  
- **Learning rate schedulers**: linear, cosine, constant with warmup, polynomial  
- **Number of epochs**: 1, 2, 3  
- **Batch sizes**: 1, 2, 4  
- **Learning rates**: from 1e-4 to 5e-4  

The following figures illustrate the validation loss trends for each hyperparameter category.

<!-- Row 1: Optimizers + Schedulers -->
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/47dc6906-6691-425b-9b89-ee50d951aada" width="400" /><br/>
      <em>Figure 3 – Validation loss comparison across optimizers</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/20ecdc9c-bb29-4d05-aeea-998421fa54b3" width="400" /><br/>
      <em>Figure 4 – Validation loss across different learning-rate schedulers</em>
    </td>
  </tr>
</table>

<!-- Row 2: Epochs + Batch size -->
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/beb830b0-0435-4248-919b-3e6e8d22e22b" width="400" /><br/>
      <em>Figure 5 – Validation loss across training epochs</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6703e0e4-388e-4197-8add-8ecf4797e100" width="400" /><br/>
      <em>Figure 6 – Validation loss across batch sizes</em>
    </td>
  </tr>
</table>

<!-- Row 3: Learning rates (alone, centered) -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/2db58604-1a7a-4c0a-891d-c933708a18f6" width="500" /><br/>
  <em>Figure 7 – Validation loss across learning rates</em>
</p>

- **Best hyperparameters chosen:**  
  - Optimizer: Adagrad  
  - Scheduler: Constant w/ Warmup 
  - Epochs: 1 (due to resource constraints)  
  - Batch Size: 4  
  - Learning Rate: 2e-4
 
The selected combination of hyperparameters was then applied to the CodeSearchNet dataset, yielding the following training loss curve. This demonstrates how the tuned setup stabilized training and reduced loss efficiently.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e94d42f-2eef-487a-a391-246127dbc9fe" alt="Figure 11" width="700"/>
</p>

<p align="center">
  <em>Figure 8 – Training loss on CodeSearchNet with tuned hyperparameters</em>
</p>

### 6.3 Transfer to CommitPackFT 
We then evaluated the tuned hyperparameters on the second dataset, CommitPackFT, to verify their effectiveness for the code change explanation task.  

The graph below shows the training loss over steps when applying the tuned setup to CommitPackFT. The training curve demonstrates a clear decrease in loss, indicating that the model successfully adapted to the commit-diff explanation task. This confirms that the selected hyperparameters generalized well beyond the initial dataset, providing consistent convergence across both tasks.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/afd7cba5-e991-4770-993a-0a59d21f01bf" alt="Figure 12" width="700"/>
</p>

<p align="center">
  <em>Figure 9 – Training loss on CommitPackFT with transferred hyperparameters</em>
</p>

### 6.4 Cocktail Training (50/50)
In the final stage, we trained the model on a cocktail dataset composed of a 50/50 mix from both CodeSearchNet and CommitPackFT, using the tuned hyperparameters.  
This approach led to a clear reduction in test loss on the combined dataset as well as consistent improvements when evaluating each dataset individually, showing the benefits of multi-task fine-tuning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ed94c049-bdba-445a-833b-bcc54655f29e" alt="Figure 10" width="600"/>
</p>

<p align="center">
  <em>Figure 10 – Cocktail fine-tuning improved CE across the combined dataset</em>
</p>

<div align="center">

| Dataset           | Baseline (no fine-tuning) CE test loss | Model after Cocktail training CE test loss |
|-------------------|-----------------------------------------|--------------------------------------------|
| **CodeSearchNet** | 0.7850 | 0.0025 |
| **CommitPackFT**  | 2.7596 | 1.4980 |
| **Cocktail (50/50)** | 1.7609 | 0.7419 |

<em>Table 1 – Cocktail training outperformed baseline across all datasets.</em>

</div>

---

## Conclusions  

- Multi-task fine-tuning on both code summarization and commit-diff explanation proved effective
- Hyperparameter tuning on CodeSearchNet transferred successfully to CommitPackFT
- Cocktail training consistently outperformed baselines
- Generalization improved: performance gains extended across datasets 

---

## Future Work  

- Test different cocktail ratios (e.g., 40/60, 70/30) 
- Extend to more programming languages beyond Python 
- Enable cross-language transfer learning* 
- Handle multi-function snippets and repository-level tasks for large-scale projects  
- Add predictive tasks (e.g., estimating delivery timelines from commit activity)  

---

## References  

- Husain H., Wu H., Gazit T., Allamanis M. & Brockschmidt M. (2019). *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*. arXiv:1909.09436
- Muennighoff N., et al. (2023). *OctoPack: Instruction Tuning Code Large Language Models*. arXiv:2308.07124v2.
- Guo, D., et al. (2024). DeepSeek-Coder: *When the Large Language Model Meets Programming*. arXiv:2401.14196.
- Xu R., Li J., Sun J., Du H. & Li J. T. (2024). *Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning on LLM Performance*. arXiv:2410.01109v1.
- *MFTCoder*. GitHub repository: [https://github.com/codefuse-ai/MFTCoder](https://github.com/codefuse-ai/MFTCoder)
