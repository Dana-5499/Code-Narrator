# 📖 Code Narrator  

Fine-Tuning Open-Source LLMs for Non-Technical Monitoring of Software Development Progress  
![Presentation1](https://github.com/user-attachments/assets/5ed80d99-6ce8-4ac5-9743-1a6dbcaa52a0)

---

## 📑 Table of Contents  

1. [Overview](#overview)  
2. [How To Run](#how-to-run)  
3. [Datasets](#datasets)  
   - 3.1 CodeSearchNet  
   - 3.2 CommitPackFT  
4. [Training Approach](#training-approach)  
5. [Model & Techniques](#model--techniques)  
   - 5.1 Base Model (DeepSeek-Coder)  
   - 5.2 Quantization (NF4)  
   - 5.3 LoRA Fine-Tuning  
   - 5.4 Loss Function  
6. [Experiments and Results](#experiments-and-results)  
   - 6.1 Baseline Comparisons  
   - 6.2 Overfitting Check  
   - 6.3 Hyperparameter Tuning  
   - 6.4 Cocktail Training (50/50)  
7. [Conclusions](#conclusions)  
8. [Future Work](#future-work)  
9. [References](#references)  

---

## Overview  
Software development consumes a large portion of company budgets, often exceeding 50% of total costs. Yet, non-technical managers frequently struggle to monitor real progress, as they must rely heavily on developer reports. This lack of visibility can lead to misallocation of resources, over-hiring, and missed risks.

Code Narrator bridges this gap by fine-tuning open-source large language models (LLMs) to generate clear, natural-language explanations of software activity. The model can describe both code snippets and commit changes in plain English, enabling managers and stakeholders to track project progress without requiring deep technical expertise.

Current LLM projects mostly aim to help **developers write code faster**. Our approach is different:  
we fine-tune open-source models to **explain software development progress in plain language**.  

This allows **non-technical managers and stakeholders** to:  
- Understand current project functionality and progress
- Track recent changes and commits
- Improve communication with development teams

---

## How To Run  


## Datasets  

### 3.1 CodeSearchNet  
- Covers **6 languages**: Python, Java, JavaScript, PHP, Ruby, Go  
- ~2M examples → compacted to **2,100 train / 600 test / 300 val per language**  
- **Preprocessing**:  
  - Removed auto-generated docstrings (e.g., *Doxygen style* like `:param x:`)  
  - Filtered docstrings (≥30 chars, capitalized, valid punctuation, no generic "See/Refer")  
  - Deduplicated samples, enforced length ranges  

### 3.2 CommitPackFT  
- Derived from **CommitPack (350+ languages)**  
- Filtered to the same 6 languages for consistency  
- Fields kept: `old_code`, `new_code`, `language`, and combined `subject+message`  
- Stratified **70–10–20 splits** per language  
- Balanced compact subsets - **2,100 train / 600 test / 300 val per language** 

---

## Training Approach  

Our method involved:  

- **Baseline comparison** across multiple models and languages  
- **Hyperparameter sweeps** (epochs, batch size, optimizers, schedulers)  
- **Transfer of tuned parameters** from CodeSearchNet → CommitPackFT  
- **Cocktail multi-task training** with a **50/50 mix** of both datasets

 <img width="561" height="542" alt="image" src="https://github.com/user-attachments/assets/afdbf650-c1ca-44d0-b1cd-93ebc7d9962b" />
 
---

## Model & Techniques  

### 5.1 Base Model (DeepSeek-Coder)  
We selected **DeepSeek-Coder 1.3B Instruct**, a Transformer-based LLM specialized for code.  

### 5.2 Quantization (NF4)  
Applied **NF4 4-bit quantization** to reduce memory usage and support training on limited resources.  

### 5.3 LoRA Fine-Tuning  
Used **Low-Rank Adaptation (LoRA)** applied to attention and projection layers, reducing trainable parameters while maintaining performance.  

### 5.4 Loss Function  
Training used **Cross-Entropy (CE) loss**, the standard for next-token prediction in LLMs.  

---

## Experiments and Results  

### 6.1 Baseline Comparisons  
*Figure 1 – Baseline validation cross-entropy across datasets* 
<img width="1757" height="1064" alt="image" src="https://github.com/user-attachments/assets/d6228f76-6d26-4200-82a5-13c4d800d005" />

### 6.2 Hyperparameter Tuning  
- Tested **optimizers** (AdamW, Adagrad, SGD)
  <img width="1574" height="723" alt="image" src="https://github.com/user-attachments/assets/47dc6906-6691-425b-9b89-ee50d951aada" />
  *Figure 2 – Validation loss comparison across optimizers*
  
- Tested **schedulers** (linear, cosine, constant w/ warmup, polynomial)
  <img width="2428" height="938" alt="image" src="https://github.com/user-attachments/assets/20ecdc9c-bb29-4d05-aeea-998421fa54b3" />
  *Figure 3 – Validation loss across different learning rate schedulers*

- Tested **number of epochs** (1, 2, 3)
  <img width="1495" height="719" alt="image" src="https://github.com/user-attachments/assets/beb830b0-0435-4248-919b-3e6e8d22e22b" />
  *Figure 4 – Validation loss across training epochs*

- Tested **batch size** (1, 2, 4)
  <img width="1442" height="711" alt="image" src="https://github.com/user-attachments/assets/6703e0e4-388e-4197-8add-8ecf4797e100" />
  *Figure 5 – Validation loss across batch sizes*

- Tested **Learning rates** (1e^-4 -> 5e^-4)
  <img width="1331" height="692" alt="image" src="https://github.com/user-attachments/assets/2db58604-1a7a-4c0a-891d-c933708a18f6" />
  *Figure 6 – Validation loss across learning rates*

- **Best hyperparameters chosen:**  
  - Optimizer: **Adagrad**  
  - Scheduler: **Constant w/ Warmup**  
  - Epochs: **1** (due to resource constraints)  
  - Batch Size: **4**  
  - Learning Rate: **2e-4**  

### 6.4 Cocktail Training (50/50) |
<img width="2142" height="1046" alt="image" src="https://github.com/user-attachments/assets/c60b7bfb-5f55-4b44-8f87-2b8debdfbae7" />
*Figure 7 – Cocktail fine-tuning improved CE across the combined dataset* 

| Dataset       | Baseline CE | Cocktail CE | Baseline PPL | Cocktail PPL |
|---------------|-------------|-------------|--------------|--------------|
| CodeSearchNet | 2.08        | **1.79**    | 8.04         | **5.98**     |
| CommitPackFT  | 2.25        | **1.95**    | 9.49         | **7.02**     |

*Table 1 – Cocktail training outperformed baseline across all metrics.*  

---

## Conclusions  

- Multi-task fine-tuning on both **code summarization** and **commit-diff explanation** proved effective  
- **Hyperparameter tuning** on CodeSearchNet transferred successfully to CommitPackFT  
- **Cocktail training** consistently outperformed baselines  
- **Generalization improved**: performance gains extended across datasets  

---

## Future Work  

- Test different **cocktail ratios** (e.g., 40/60, 70/30)  
- Extend to **more programming languages** beyond Python  
- Enable **cross-language transfer learning**  
- Handle **multi-function snippets** and **repository-level tasks** for large-scale projects  
- Add predictive tasks (e.g., **estimating delivery timelines** from commit activity)  

---

## References  

- Husain H., Wu H., Gazit T., Allamanis M. & Brockschmidt M. (2019). *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*. arXiv:1909.09436
- Muennighoff N., et al. (2023). *OctoPack: Instruction Tuning Code Large Language Models*. arXiv:2308.07124v2.
- Guo, D., et al. (2024). DeepSeek-Coder: *When the Large Language Model Meets Programming*. arXiv:2401.14196.
- Xu R., Li J., Sun J., Du H. & Li J. T. (2024). *Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning on LLM Performance*. arXiv:2410.01109v1.
- *MFTCoder*. GitHub repository: [https://github.com/codefuse-ai/MFTCoder](https://github.com/codefuse-ai/MFTCoder)
