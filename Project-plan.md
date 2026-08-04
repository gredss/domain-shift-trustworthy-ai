# IndoBERT Clickbait Detection: Robustness Evaluation System
## Project Plan and Technical Specification

---

## 1. Executive Summary

This document outlines the complete implementation plan for an end-to-end robustness evaluation system for Indonesian clickbait detection using IndoBERT model variants. The system addresses four primary research questions through a comprehensive pipeline encompassing data preparation, model fine-tuning, perturbation testing, statistical validation, and interactive visualization.

**Primary Objectives:**
- Evaluate domain shift impact on model performance (RQ1)
- Assess vulnerability to linguistic perturbations (RQ2)
- Analyze interaction effects between domain shift and textual noise (RQ3)
- Deploy functional evaluation dashboard for practical use (RQ4)

**Technology Stack:** Python 3.8+, PyTorch, Transformers, Streamlit, NumPy, Pandas, SciPy

---

## 2. System Architecture Overview

The system follows a modular pipeline architecture with five core components:

**Component 1: Data Management Module**
- Dataset loading and validation
- Stratified train-validation-test splitting (70-15-15)
- Domain-specific data organization
- Reproducibility control (seed=42)

**Component 2: Model Training Module**
- IndoBERT variant initialization (base, large, lite)
- Grid search hyperparameter optimization
- Fine-tuning with optimal configuration
- Model checkpoint management

**Component 3: Evaluation Engine**
- In-domain performance assessment
- Cross-domain evaluation (5x5 matrix)
- Perturbation stress testing (3 intensity levels)
- Metric computation (Accuracy, Precision, Recall, F1-Score)

**Component 4: Statistical Analysis Module**
- Source Drop (SD) and Target Drop (TD) calculation
- Bayesian significance testing
- Region of Practical Equivalence (ROPE) analysis
- Robustness comparison across variants

**Component 5: Interactive Dashboard**
- Single-text prediction with real-time perturbation
- Domain shift visualization matrix
- Robustness saturation analysis charts
- Automated reliability summary generation

---

## 3. Data Pipeline Design

**3.1 Dataset Structure**
- Total entries: 5,000 samples
- Distribution: 1,000 samples per domain (5 domains)
- Labels: Binary (0=non-clickbait, 1=clickbait)
- Format: CSV with columns [text, label, domain]

**3.2 Preprocessing Workflow**
- Text normalization and cleaning
- Tokenization using IndoBERT tokenizer
- Maximum sequence length: 128 tokens
- Padding and truncation handling
- Attention mask generation

**3.3 Data Splitting Strategy**
- Stratified split maintaining label and domain distribution
- Training set: 3,500 samples (70%)
- Validation set: 750 samples (15%)
- Test set: 750 samples (15%)
- Fixed random seed for reproducibility

**3.4 Domain Organization**
- Each domain treated as separate evaluation unit
- Cross-domain test sets prepared for all combinations
- Domain metadata preserved for analysis

---

## 4. Model Training Framework

**4.1 Model Variants**
- IndoBERT-base-p1: Standard architecture
- IndoBERT-large-p1: Enhanced capacity variant
- IndoBERT-lite-base-p1: Lightweight variant

**4.2 Architecture Configuration**
- Feature extraction: CLS token from final layer
- Regularization: Dropout layer
- Classification head: Dense linear layer
- Activation: Softmax for binary probabilities

**4.3 Hyperparameter Configuration**
- Learning rate: 2e-5
- Batch size: 32
- Training epochs: 4
- Optimizer: AdamW
- Weight decay: 0.01
- Warmup steps: 10% of total steps

**4.4 Training Process**
- Grid search for optimal hyperparameters
- Validation-based early stopping
- Loss monitoring and logging
- Model checkpoint saving
- Performance tracking per epoch

---

## 5. Perturbation Testing Methodology

**5.1 Perturbation Levels**

**Low-Level Perturbations:**
- Character-level typos (insertion, deletion, substitution)
- Minor spelling errors
- Accidental key presses
- Intensity: 5-10% of characters affected

**Medium-Level Perturbations:**
- Informal language injection
- Slang and colloquial terms
- Abbreviations and acronyms
- Mixed formal-informal register
- Intensity: 15-25% of words affected

**High-Level Perturbations:**
- Complete sentence paraphrasing
- Synonym replacement
- Sentence structure modification
- Semantic preservation with syntactic variation
- Intensity: 40-60% of content altered

**5.2 Perturbation Application**
- Systematic application to test sets
- Preservation of original labels
- Domain-specific perturbation tracking
- Reproducible perturbation generation

**5.3 Evaluation Scenarios**
- In-domain clean baseline
- In-domain with perturbations (3 levels)
- Cross-domain clean
- Cross-domain with perturbations (3 levels)
- Total: 80 evaluation scenarios per model

---

## 6. Evaluation Metrics and Analysis

**6.1 Performance Metrics**
- Accuracy: Overall correctness
- Precision: Positive prediction reliability
- Recall: True positive detection rate
- F1-Score: Harmonic mean (primary metric)

**6.2 Robustness Metrics**
- Source Drop (SD): Performance degradation from source domain
- Target Drop (TD): Performance degradation in target domain
- Relative degradation percentages
- Absolute F1-Score differences

**6.3 Statistical Validation**
- Bayesian signed-rank test for paired comparisons
- Bayesian hierarchical modeling for variant comparison
- ROPE threshold: 0.01 (1% practical equivalence)
- Credible interval analysis
- Posterior probability distributions

**6.4 Comparative Analysis**
- Inter-variant robustness comparison
- Domain-specific vulnerability patterns
- Perturbation sensitivity profiles
- Interaction effect quantification

---

## 7. Dashboard Application Design

**7.1 Module 1: Single-Text Prediction**
- Text input interface
- Real-time prediction display
- Confidence score visualization
- Perturbation generator controls
- Side-by-side comparison (clean vs perturbed)

**7.2 Module 2: Domain Shift Matrix**
- 5x5 heatmap visualization
- Source domains (rows) vs target domains (columns)
- Color-coded performance indicators
- Interactive cell selection
- Detailed metrics on hover

**7.3 Module 3: Robustness Analysis**
- Line charts for perturbation levels
- Multi-model comparison plots
- Saturation point identification
- Performance degradation curves
- Domain-specific trend analysis

**7.4 Module 4: Reliability Summary**
- Automated report generation
- Statistical significance indicators
- Best-worst case scenarios
- Recommendation engine
- Exportable summary tables

**7.5 Technical Implementation**
- Streamlit framework
- Plotly for interactive charts
- Session state management
- Model caching for performance
- Responsive layout design

---

## 8. Module Structure and File Organization

**8.1 Core Modules**

**data_manager.py**
- Dataset loading and validation
- Stratified splitting implementation
- Data preprocessing pipeline
- Domain organization utilities

**model_trainer.py**
- Model initialization and configuration
- Training loop implementation
- Hyperparameter grid search
- Checkpoint management
- Evaluation utilities

**perturbation_engine.py**
- Low-level perturbation functions
- Medium-level perturbation functions
- High-level perturbation functions
- Perturbation intensity control
- Reproducible noise generation

**evaluation_engine.py**
- Metric calculation functions
- In-domain evaluation
- Cross-domain evaluation
- Perturbation testing orchestration
- Results aggregation

**statistical_analyzer.py**
- Bayesian test implementation
- ROPE analysis functions
- Significance testing utilities
- Comparative statistics
- Result interpretation

**dashboard_app.py**
- Streamlit application entry point
- UI component definitions
- Module integration
- State management
- Visualization functions

**8.2 Configuration Files**

**config.py**
- Model hyperparameters
- Training configuration
- Evaluation settings
- Perturbation parameters
- File paths and constants

**8.3 Utility Modules**

**utils.py**
- Common helper functions
- Logging utilities
- File I/O operations
- Reproducibility helpers

---

## 9. Implementation Workflow

**Phase 1: Environment Setup**
- Python environment configuration
- Dependency installation
- Directory structure creation
- Configuration file setup

**Phase 2: Data Pipeline Implementation**
- Dataset loading functionality
- Preprocessing pipeline
- Splitting mechanism
- Validation checks

**Phase 3: Model Training System**
- Model initialization
- Training loop
- Grid search implementation
- Checkpoint system

**Phase 4: Evaluation Framework**
- Metric calculation
- In-domain evaluation
- Cross-domain evaluation
- Results storage

**Phase 5: Perturbation System**
- Low-level perturbations
- Medium-level perturbations
- High-level perturbations
- Integration with evaluation

**Phase 6: Statistical Analysis**
- Bayesian test implementation
- ROPE analysis
- Significance testing
- Report generation

**Phase 7: Dashboard Development**
- UI layout design
- Module implementation
- Visualization integration
- Testing and refinement

**Phase 8: Integration and Testing**
- End-to-end pipeline testing
- Performance optimization
- Documentation completion
- Deployment preparation

---

## 10. Technical Specifications

**10.1 Dependencies**
- torch >= 1.10.0
- transformers >= 4.20.0
- streamlit >= 1.15.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- plotly >= 5.0.0
- baycomp >= 1.0.2

**10.2 Hardware Requirements**
- GPU: CUDA-compatible (recommended for training)
- RAM: Minimum 16GB
- Storage: 10GB for models and data
- CPU: Multi-core processor for parallel processing

**10.3 Performance Considerations**
- Model caching for dashboard responsiveness
- Batch processing for large-scale evaluation
- Efficient perturbation generation
- Optimized metric calculation
- Memory management for large models

---

## 11. Research Question Mapping

**RQ1: Domain Shift Impact**
- Addressed by: Cross-domain evaluation (Module 2)
- Metrics: SD, TD, F1-Score degradation
- Visualization: Domain shift matrix
- Analysis: Statistical significance testing

**RQ2: Linguistic Perturbation Vulnerability**
- Addressed by: Perturbation testing (Module 3)
- Metrics: Performance across 3 perturbation levels
- Visualization: Robustness curves
- Analysis: Degradation patterns

**RQ3: Interaction Effects**
- Addressed by: Combined domain-perturbation testing
- Metrics: Cumulative degradation analysis
- Visualization: Multi-dimensional comparison
- Analysis: Interaction effect quantification

**RQ4: System Deployment**
- Addressed by: Streamlit dashboard (Module 5)
- Features: All four dashboard modules
- Usability: Interactive and user-friendly
- Functionality: Complete evaluation pipeline

---

## 12. Quality Assurance

**12.1 Testing Strategy**
- Unit tests for individual functions
- Integration tests for module interactions
- End-to-end pipeline validation
- Dashboard functionality testing
- Performance benchmarking

**12.2 Validation Checks**
- Data integrity verification
- Model output validation
- Metric calculation accuracy
- Statistical test correctness
- Visualization accuracy

**12.3 Documentation Standards**
- Inline code documentation
- Function docstrings
- Module-level documentation
- User guide for dashboard
- Technical reference manual

---

## 13. Deliverables

**13.1 Code Artifacts**
- Complete Python implementation
- Configuration files
- Utility scripts
- Requirements specification

**13.2 Trained Models**
- Fine-tuned IndoBERT variants
- Model checkpoints
- Training logs
- Performance reports

**13.3 Evaluation Results**
- Performance metrics tables
- Statistical analysis reports
- Visualization outputs
- Domain shift matrices

**13.4 Dashboard Application**
- Functional Streamlit app
- Interactive visualizations
- User documentation
- Deployment guide

**13.5 Documentation**
- Project plan (this document)
- Technical documentation
- User manual
- API reference

---

## 14. Success Criteria

**14.1 Functional Requirements**
- All models successfully fine-tuned
- Complete evaluation across all scenarios
- Statistical validation implemented
- Dashboard fully operational
- All RQs empirically answered

**14.2 Performance Requirements**
- Training convergence achieved
- Evaluation metrics computed accurately
- Dashboard responsive (< 2s load time)
- Statistical tests properly executed
- Visualizations render correctly

**14.3 Quality Requirements**
- Code follows Python best practices
- Modular and maintainable structure
- Comprehensive documentation
- Reproducible results
- Professional presentation

---

## 15. Risk Mitigation

**15.1 Technical Risks**
- Model training instability: Use gradient clipping and learning rate scheduling
- Memory constraints: Implement batch processing and model caching
- Computational time: Optimize evaluation loops and use GPU acceleration
- Data quality issues: Implement validation checks and preprocessing

**15.2 Implementation Risks**
- Scope creep: Maintain focus on core requirements
- Integration challenges: Modular design with clear interfaces
- Performance bottlenecks: Profile and optimize critical paths
- Dependency conflicts: Use virtual environment and version pinning

---

## 16. Timeline Estimation

**Week 1-2: Foundation**
- Environment setup
- Data pipeline implementation
- Basic model training framework

**Week 3-4: Core Development**
- Complete training system
- Evaluation framework
- Perturbation engine

**Week 5-6: Analysis and Validation**
- Statistical analysis implementation
- Comprehensive evaluation execution
- Results validation

**Week 7-8: Dashboard and Integration**
- Dashboard development
- Module integration
- Testing and refinement

**Week 9: Finalization**
- Documentation completion
- Final testing
- Deployment preparation

---

## 17. Conclusion

This project plan provides a comprehensive roadmap for implementing a robust, modular, and scientifically rigorous evaluation system for IndoBERT clickbait detection models. The architecture balances modularity with practical file organization, ensuring maintainability without excessive fragmentation. The implementation will directly address all four research questions through empirical evaluation, statistical validation, and practical deployment, culminating in a functional dashboard that demonstrates the complete robustness evaluation pipeline.
