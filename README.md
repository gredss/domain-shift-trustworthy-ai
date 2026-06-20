# IndoBERT Clickbait Detection: Robustness Evaluation System
By Grace Esther Simanjuntak, Angela Valerie Christy, Khansa Nabilah Awali.
## Overview

This repository contains the end-to-end robustness evaluation system for Indonesian clickbait detection using various IndoBERT model variants. The system systematically evaluates model performance against domain shifts and linguistic perturbations, providing a scientifically rigorous framework to assess model reliability in real-world scenarios.

---

## Key Features

* **Robust Fine-Tuning:** Supports multiple IndoBERT variants including base, large, and lite models with optimized hyperparameter configurations.
* **Cross-Domain Evaluation:** Evaluates models across five distinct text domains using a comprehensive 5x5 cross-domain performance matrix.
* **Behavioral Stress Testing:** Applies three levels of text perturbations (character typos, informal slang, and syntactic paraphrasing) to analyze model vulnerabilities.
* **Statistical Validation:** Features Bayesian significance testing and Region of Practical Equivalence (ROPE) analysis to mathematically validate model robustness.
* **Interactive Dashboard:** A Streamlit-powered web application providing real-time text prediction, domain shift heatmaps, and automated reliability reports.

---

## System Architecture

The project is structured into five modular components:

1. **Data Management Module:** Handles data cleaning, tokenization, and stratified train-validation-test splitting.
2. **Model Training Module:** Manages training loops, grid search optimization, and checkpoint saving for IndoBERT variants.
3. **Evaluation Engine:** Computes standard classification metrics and coordinates cross-domain testing scenarios.
4. **Statistical Analyzer:** Computes Source Drop and Target Drop metrics alongside Bayesian signed-rank tests.
5. **Dashboard Application:** Integrates all modules into an interactive user interface for visual analysis.

---

## Technology Stack

* **Frameworks:** PyTorch, Hugging Face Transformers, Streamlit
* **Data and Analysis:** NumPy, Pandas, SciPy, Scikit-learn, Baycomp
* **Visualization:** Plotly

---

## Installation and Usage

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Configure parameters in `config.py`.

### Running the System

To run the full evaluation pipeline:

```bash
python evaluation_engine.py

```

To launch the interactive visualization dashboard:

```bash
streamlit run dashboard_app.py

```

---

## Repository Structure

* `data_manager.py`: Data pipeline and preprocessing utilities.
* `model_trainer.py`: Model initialization and fine-tuning loops.
* `perturbation_engine.py`: Text noise and perturbation generators.
* `evaluation_engine.py`: Scenario testing and performance logging.
* `statistical_analyzer.py`: Bayesian verification logic.
* `dashboard_app.py`: Streamlit user interface implementation.
* `config.py`: Global hyperparameter and system settings.
