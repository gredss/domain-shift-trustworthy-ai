# Robustness Evaluation of Indonesian NLP Models under Domain Shift and Resource Constraints for Trustworthy AI

## Project Overview

This repository contains methodology for a comprehensive evaluation of Indonesian Natural Language Processing (NLP) models. The research focuses on the performance of Transformer based architectures when subjected to domain shift and resource constraints. 

A primary objective of this study is the rigorous assessment of **Robustness**, examining how models maintain performance across varied linguistic environments and under significant input noise.

## Core Research Focus: Robustness

The methodology prioritizes robustness through two specific dimensions:

1.  **Domain Robustness (RQ1):** Evaluating the generalization capabilities of models trained on formal source domains (News and Government) when deployed on informal target domains (Social Media and E-commerce).
2.  **Perturbation Robustness (RQ2):** Measuring model degradation across four noise levels (10% to 40%) using character swaps, deletions, insertions, and keyboard-adjacent errors.

## Research Methodology

The project is structured into seven distinct phases to ensure statistical validity and reproducibility.

### Phase 1: Research Foundation
The study begins with a synthesis of 24 research papers to identify the existing gap in joint evaluations of robustness, stability, and reliability. The scope is limited to Indonesian Sentiment Analysis and Text Classification tasks.

* **Primary Model:** IndoBERT-base-p1
* **Comparison Variants:** IndoBERT-large, NusaBERT, and IndoBERT-lite

### Phase 2: Data Preparation
Data is collected from multi-domain Indonesian corpora and partitioned into Source Domains (SD) and Target Domains (TD).
* **Preprocessing:** Includes normalization, tokenization, slang standardization, and Unicode cleaning.
* **Synthetic Noise:** Generation of perturbations at controlled levels to test the limits of model stability.
* **Quality Audit:** Validation through class balance checks and Out-of-Vocabulary (OOV) rate analysis.

### Phase 3: Model Fine-Tuning
All models are fine-tuned on the Source Domain using a consistent hyperparameter configuration. This controlled setup allows for a direct comparison of how model capacity and architecture influence resulting robustness and efficiency.

### Phase 4: Multi-Dimensional Evaluation
The evaluation addresses five core research questions:
* **RQ1 (Domain Robustness):** Cross-domain performance measured via F1 and Macro-F1 scores.
* **RQ2 (Perturbation Robustness):** Analysis of degradation curves based on performance deltas versus clean data.
* **RQ3 (Prediction Stability):** Assessment of label flip rates across semantically equivalent inputs.
* **RQ4 (Confidence Reliability):** Calibration analysis using Expected Calibration Error (ECE) and Reliability Diagrams.
* **RQ5 (Resource Trade-off):** Profiling inference latency, GPU memory usage, and parameter counts.

### Phase 5: Advanced Analysis and Statistical Grounding
The analysis phase utilizes IBM Bob (powered by Claude 3.7 Sonnet) to orchestrate error taxonomy and cross-dimensional reasoning.
* **Error Taxonomy:** Classification of failures into lexical, structural, noise-induced, or calibration-related categories.
* **Statistical Rigor:** Application of McNemar’s Test, Bootstrap Confidence Intervals, and Bonferroni Correction to validate all empirical claims.

### Phase 6: Results and Discussion
Findings are organized to provide a class-level flip analysis and detection of majority-class collapse under extreme domain shift. This phase establishes the "Performance vs. Cost" frontier for Indonesian NLP deployment.

### Phase 7: Conclusion
The final output provides empirically grounded guidelines for practitioners to deploy trustworthy and robust Indonesian NLP systems in real-world environments.

## Technical Specifications

| Component | Description |
| :--- | :--- |
| **Language** | Indonesian |
| **Architectures** | IndoBERT (Base, Large, Lite), NusaBERT |
| **Frameworks** | PyTorch, Hugging Face Transformers |
| **Statistical Tests** | McNemar, Cohen's h, Bootstrap CI |

---
*This projecrt provides the first joint evaluation of robustness, stability, and reliability for Indonesian NLP under domain shift and resource constraints.*
