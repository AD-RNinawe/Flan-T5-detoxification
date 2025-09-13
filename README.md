# Flan-T5-detoxification
Fine-tuning FLAN-T5 with Reinforcement Learning (PPO) and PEFT to reduce toxicity in generated summaries. Includes training setup, evaluation pipeline, and integration with Meta AI’s hate speech reward model.
# Fine-Tuning FLAN-T5 for Detoxified Summarization

This repository contains a Jupyter Notebook that demonstrates how to fine-tune the **FLAN-T5** model using **Reinforcement Learning (PPO)** and **Parameter-Efficient Fine-Tuning (PEFT)** techniques.  
The objective is to guide the model toward generating **less-toxic summaries**, using Meta AI’s **hate speech reward model** as feedback.

---

## 📌 Project Overview
- **Model Base:** FLAN-T5 (Seq2Seq LM)
- **Techniques Used:**
  - Reinforcement Learning with PPO (from 🤗 TRL library)
  - PEFT with LoRA for efficient training
- **Reward Signal:** Meta AI’s hate speech binary classifier (`hate` vs. `not hate`)
- **Goal:** Reduce toxicity in abstractive text summarization

---

## 🛠️ Requirements
This project makes use of the following Python libraries:
- `transformers`
- `trl` (Hugging Face TRL library for RLHF)
- `peft`
- `datasets`
- `evaluate`
- `tqdm`
- `pandas`

Install them with:
```bash
pip install transformers trl peft datasets evaluate tqdm pandas


Acknowledgments

Hugging Face TRL for PPO training framework

Meta AI Hate Speech Classifier for reward modeling

FLAN-T5 from Google Research
