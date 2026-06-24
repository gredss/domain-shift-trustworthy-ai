"""
Evaluation Engine Module for IndoBERT Clickbait Detection System

This module handles metric calculation, in-domain evaluation, cross-domain evaluation,
perturbation testing orchestration, and results aggregation for robustness analysis.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates performance metrics for clickbait detection.
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Add macro and weighted averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_robustness_metrics(
        clean_metrics: Dict[str, float],
        perturbed_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate robustness degradation metrics.
        
        Args:
            clean_metrics: Metrics on clean data
            perturbed_metrics: Metrics on perturbed data
            
        Returns:
            Dictionary with degradation metrics
        """
        robustness = {}
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in clean_metrics and metric_name in perturbed_metrics:
                clean_val = clean_metrics[metric_name]
                perturbed_val = perturbed_metrics[metric_name]
                
                # Absolute drop
                robustness[f'{metric_name}_drop'] = clean_val - perturbed_val
                
                # Relative drop (percentage)
                if clean_val > 0:
                    robustness[f'{metric_name}_drop_pct'] = (
                        (clean_val - perturbed_val) / clean_val * 100
                    )
                else:
                    robustness[f'{metric_name}_drop_pct'] = 0.0
        
        return robustness
    
    @staticmethod
    def calculate_domain_shift_metrics(
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate Source Drop (SD) and Target Drop (TD) metrics.
        
        Args:
            source_metrics: Metrics on source domain
            target_metrics: Metrics on target domain
            
        Returns:
            Dictionary with SD and TD metrics
        """
        shift_metrics = {}
        
        # Source Drop (SD): Performance degradation from source domain
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in source_metrics and metric_name in target_metrics:
                source_val = source_metrics[metric_name]
                target_val = target_metrics[metric_name]
                
                # Absolute SD
                shift_metrics[f'sd_{metric_name}'] = source_val - target_val
                
                # Relative SD (percentage)
                if source_val > 0:
                    shift_metrics[f'sd_{metric_name}_pct'] = (
                        (source_val - target_val) / source_val * 100
                    )
                else:
                    shift_metrics[f'sd_{metric_name}_pct'] = 0.0
        
        return shift_metrics


class InDomainEvaluator:
    """
    Handles in-domain evaluation for trained models.
    """
    
    def __init__(self, model_trainer):
        """
        Initialize in-domain evaluator.
        
        Args:
            model_trainer: ModelTrainer instance with trained model
        """
        self.model_trainer = model_trainer
        self.metrics_calculator = MetricsCalculator()
        logger.info("InDomainEvaluator initialized")
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on in-domain test data.
        
        Args:
            test_df: Test DataFrame with 'text' and 'label' columns
            domain: Domain name (for logging)
            
        Returns:
            Dictionary with evaluation results
        """
        domain_str = f" ({domain})" if domain else ""
        logger.info(f"Evaluating in-domain{domain_str}: {len(test_df)} samples")
        
        # Get predictions
        y_true = test_df['label'].values
        y_pred, y_proba = self.model_trainer.predict(test_df['text'].tolist())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Add metadata
        results = {
            'domain': domain,
            'num_samples': len(test_df),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }
        
        logger.info(f"In-domain{domain_str} F1-Score: {metrics['f1']:.4f}")
        
        return results
    
    def evaluate_by_domain(
        self,
        test_df: pd.DataFrame,
        domains: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model separately for each domain.
        
        Args:
            test_df: Test DataFrame with 'domain' column
            domains: List of domain names
            
        Returns:
            Dictionary mapping domain names to evaluation results
        """
        logger.info(f"Evaluating across {len(domains)} domains")
        
        results = {}
        
        for domain in domains:
            domain_df = test_df[test_df['domain'] == domain].copy()
            if len(domain_df) > 0:
                results[domain] = self.evaluate(domain_df, domain)
            else:
                logger.warning(f"No samples found for domain: {domain}")
        
        return results


class CrossDomainEvaluator:
    """
    Handles cross-domain evaluation (5x5 matrix).
    """
    
    def __init__(self, model_trainers: Dict[str, Any]):
        """
        Initialize cross-domain evaluator.
        
        Args:
            model_trainers: Dictionary mapping domain names to ModelTrainer instances
        """
        self.model_trainers = model_trainers
        self.metrics_calculator = MetricsCalculator()
        logger.info("CrossDomainEvaluator initialized")
    
    def evaluate_cross_domain(
        self,
        source_domain: str,
        target_domain: str,
        target_test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate source domain model on target domain data.
        
        Args:
            source_domain: Domain the model was trained on
            target_domain: Domain to evaluate on
            target_test_df: Test data from target domain
            
        Returns:
            Dictionary with cross-domain evaluation results
        """
        logger.info(f"Cross-domain evaluation: {source_domain} -> {target_domain}")
        
        if source_domain not in self.model_trainers:
            raise ValueError(f"No model found for source domain: {source_domain}")
        
        model_trainer = self.model_trainers[source_domain]
        
        # Get predictions
        y_true = target_test_df['label'].values
        y_pred, y_proba = model_trainer.predict(target_test_df['text'].tolist())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        results = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'num_samples': len(target_test_df),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }
        
        logger.info(
            f"Cross-domain {source_domain} -> {target_domain} F1-Score: {metrics['f1']:.4f}"
        )
        
        return results
    
    def evaluate_all_combinations(
        self,
        test_data: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Evaluate all source-target domain combinations (5x5 matrix).
        
        Args:
            test_data: Dictionary mapping domain names to test DataFrames
            
        Returns:
            Dictionary mapping (source, target) tuples to evaluation results
        """
        domains = list(test_data.keys())
        logger.info(f"Evaluating {len(domains)}x{len(domains)} cross-domain matrix")
        
        results = {}
        
        for source_domain in domains:
            for target_domain in domains:
                if target_domain in test_data:
                    key = (source_domain, target_domain)
                    results[key] = self.evaluate_cross_domain(
                        source_domain,
                        target_domain,
                        test_data[target_domain]
                    )
        
        logger.info("Cross-domain evaluation matrix complete")
        return results
    
    def create_performance_matrix(
        self,
        cross_domain_results: Dict[Tuple[str, str], Dict[str, Any]],
        metric: str = 'f1'
    ) -> pd.DataFrame:
        """
        Create a performance matrix for visualization.
        
        Args:
            cross_domain_results: Results from evaluate_all_combinations
            metric: Metric to use for matrix values
            
        Returns:
            DataFrame with source domains as rows, target domains as columns
        """
        # Extract unique domains
        domains = sorted(set(
            [k[0] for k in cross_domain_results.keys()] +
            [k[1] for k in cross_domain_results.keys()]
        ))
        
        # Create matrix
        matrix = pd.DataFrame(index=domains, columns=domains, dtype=float)
        
        for (source, target), results in cross_domain_results.items():
            if metric in results['metrics']:
                matrix.loc[source, target] = results['metrics'][metric]
        
        return matrix


class PerturbationEvaluator:
    """
    Orchestrates perturbation testing across multiple levels.
    """
    
    def __init__(
        self,
        model_trainer,
        perturbation_engine
    ):
        """
        Initialize perturbation evaluator.
        
        Args:
            model_trainer: ModelTrainer instance
            perturbation_engine: PerturbationEngine instance
        """
        self.model_trainer = model_trainer
        self.perturbation_engine = perturbation_engine
        self.metrics_calculator = MetricsCalculator()
        logger.info("PerturbationEvaluator initialized")
    
    def evaluate_with_perturbation(
        self,
        test_df: pd.DataFrame,
        perturbation_level: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on perturbed test data.
        
        Args:
            test_df: Test DataFrame
            perturbation_level: 'low', 'medium', or 'high'
            domain: Domain name (for logging)
            
        Returns:
            Dictionary with evaluation results
        """
        domain_str = f" ({domain})" if domain else ""
        logger.info(
            f"Evaluating with {perturbation_level}-level perturbation{domain_str}"
        )
        
        # Apply perturbations
        perturbed_df = self.perturbation_engine.apply_to_dataframe(
            test_df.copy(),
            text_column='text',
            level=perturbation_level
        )
        
        # Get predictions
        y_true = perturbed_df['label'].values
        y_pred, y_proba = self.model_trainer.predict(perturbed_df['text'].tolist())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        results = {
            'domain': domain,
            'perturbation_level': perturbation_level,
            'num_samples': len(perturbed_df),
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'true_labels': y_true.tolist()
        }
        
        logger.info(
            f"{perturbation_level.capitalize()}-level perturbation{domain_str} "
            f"F1-Score: {metrics['f1']:.4f}"
        )
        
        return results
    
    def evaluate_all_levels(
        self,
        test_df: pd.DataFrame,
        clean_results: Dict[str, Any],
        domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model across all perturbation levels.
        
        Args:
            test_df: Test DataFrame
            clean_results: Results on clean (unperturbed) data
            domain: Domain name
            
        Returns:
            Dictionary mapping perturbation levels to results
        """
        levels = ['low', 'medium', 'high']
        results = {'clean': clean_results}
        
        for level in levels:
            results[level] = self.evaluate_with_perturbation(test_df, level, domain)
            
            # Calculate robustness metrics
            robustness = self.metrics_calculator.calculate_robustness_metrics(
                clean_results['metrics'],
                results[level]['metrics']
            )
            results[level]['robustness_metrics'] = robustness
        
        return results


class EvaluationEngine:
    """
    Main evaluation engine orchestrating all evaluation types.
    """
    
    def __init__(
        self,
        model_trainers: Dict[str, Any],
        perturbation_engine,
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize evaluation engine.
        
        Args:
            model_trainers: Dictionary mapping domain names to ModelTrainer instances
            perturbation_engine: PerturbationEngine instance
            output_dir: Directory to save evaluation results
        """
        self.model_trainers = model_trainers
        self.perturbation_engine = perturbation_engine
        self.output_dir = output_dir
        
        # Initialize evaluators
        self.metrics_calculator = MetricsCalculator()
        self.cross_domain_evaluator = CrossDomainEvaluator(model_trainers)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"EvaluationEngine initialized. Results will be saved to {output_dir}")
    
    def run_complete_evaluation(
        self,
        test_data: Dict[str, pd.DataFrame],
        include_perturbations: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline including in-domain, cross-domain,
        and perturbation testing.
        
        Args:
            test_data: Dictionary mapping domain names to test DataFrames
            include_perturbations: Whether to include perturbation testing
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting complete evaluation pipeline")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'domains': list(test_data.keys()),
            'in_domain': {},
            'cross_domain': {},
            'perturbation': {}
        }
        
        # 1. In-domain evaluation
        logger.info("Phase 1: In-domain evaluation")
        for domain, test_df in test_data.items():
            if domain in self.model_trainers:
                evaluator = InDomainEvaluator(self.model_trainers[domain])
                results['in_domain'][domain] = evaluator.evaluate(test_df, domain)
        
        # 2. Cross-domain evaluation
        logger.info("Phase 2: Cross-domain evaluation")
        results['cross_domain'] = self.cross_domain_evaluator.evaluate_all_combinations(
            test_data
        )
        
        # 3. Perturbation testing
        if include_perturbations:
            logger.info("Phase 3: Perturbation testing")
            for domain, test_df in test_data.items():
                if domain in self.model_trainers:
                    logger.info(f"Perturbation testing for domain: {domain}")
                    
                    # Get clean results
                    clean_results = results['in_domain'][domain]
                    
                    # Evaluate with perturbations
                    evaluator = PerturbationEvaluator(
                        self.model_trainers[domain],
                        self.perturbation_engine
                    )
                    results['perturbation'][domain] = evaluator.evaluate_all_levels(
                        test_df,
                        clean_results,
                        domain
                    )
        
        logger.info("Complete evaluation pipeline finished")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def run_cross_domain_with_perturbations(
        self,
        test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run cross-domain evaluation with perturbations (interaction effects).
        
        Args:
            test_data: Dictionary mapping domain names to test DataFrames
            
        Returns:
            Dictionary with cross-domain perturbation results
        """
        logger.info("Starting cross-domain evaluation with perturbations")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'cross_domain_perturbation': {}
        }
        
        domains = list(test_data.keys())
        perturbation_levels = ['clean', 'low', 'medium', 'high']
        
        for source_domain in domains:
            if source_domain not in self.model_trainers:
                continue
            
            model_trainer = self.model_trainers[source_domain]
            
            for target_domain in domains:
                if target_domain not in test_data:
                    continue
                
                key = f"{source_domain}_to_{target_domain}"
                results['cross_domain_perturbation'][key] = {}
                
                target_df = test_data[target_domain]
                
                # Evaluate on clean data
                y_true = target_df['label'].values
                y_pred, y_proba = model_trainer.predict(target_df['text'].tolist())
                clean_metrics = self.metrics_calculator.calculate_metrics(
                    y_true, y_pred, y_proba
                )
                results['cross_domain_perturbation'][key]['clean'] = {
                    'metrics': clean_metrics
                }
                
                # Evaluate with perturbations
                for level in ['low', 'medium', 'high']:
                    perturbed_df = self.perturbation_engine.apply_to_dataframe(
                        target_df.copy(),
                        text_column='text',
                        level=level
                    )
                    
                    y_true = perturbed_df['label'].values
                    y_pred, y_proba = model_trainer.predict(
                        perturbed_df['text'].tolist()
                    )
                    perturbed_metrics = self.metrics_calculator.calculate_metrics(
                        y_true, y_pred, y_proba
                    )
                    
                    # Calculate robustness metrics
                    robustness = self.metrics_calculator.calculate_robustness_metrics(
                        clean_metrics,
                        perturbed_metrics
                    )
                    
                    results['cross_domain_perturbation'][key][level] = {
                        'metrics': perturbed_metrics,
                        'robustness_metrics': robustness
                    }
                    
                    logger.info(
                        f"{source_domain} -> {target_domain} ({level}): "
                        f"F1={perturbed_metrics['f1']:.4f}"
                    )
        
        logger.info("Cross-domain perturbation evaluation complete")
        
        # Save results
        self.save_results(results, filename='cross_domain_perturbation_results.json')
        
        return results
    
    def aggregate_results(
        self,
        results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Aggregate evaluation results into a summary DataFrame.
        
        Args:
            results: Results dictionary from run_complete_evaluation
            
        Returns:
            DataFrame with aggregated results
        """
        logger.info("Aggregating evaluation results")
        
        rows = []
        
        # In-domain results
        if 'in_domain' in results:
            for domain, domain_results in results['in_domain'].items():
                row = {
                    'evaluation_type': 'in_domain',
                    'source_domain': domain,
                    'target_domain': domain,
                    'perturbation_level': 'clean',
                    **domain_results['metrics']
                }
                rows.append(row)
        
        # Cross-domain results
        if 'cross_domain' in results:
            for (source, target), domain_results in results['cross_domain'].items():
                row = {
                    'evaluation_type': 'cross_domain',
                    'source_domain': source,
                    'target_domain': target,
                    'perturbation_level': 'clean',
                    **domain_results['metrics']
                }
                rows.append(row)
        
        # Perturbation results
        if 'perturbation' in results:
            for domain, pert_results in results['perturbation'].items():
                for level, level_results in pert_results.items():
                    if level != 'clean':
                        row = {
                            'evaluation_type': 'perturbation',
                            'source_domain': domain,
                            'target_domain': domain,
                            'perturbation_level': level,
                            **level_results['metrics']
                        }
                        if 'robustness_metrics' in level_results:
                            row.update(level_results['robustness_metrics'])
                        rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save aggregated results
        output_path = os.path.join(self.output_dir, 'aggregated_results.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Aggregated results saved to {output_path}")
        
        return df
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = 'evaluation_results.json'
    ) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = self._make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj
    
    def generate_summary_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            results: Results dictionary from evaluation
            
        Returns:
            Summary report as string
        """
        report_lines = [
            "=" * 80,
            "EVALUATION SUMMARY REPORT",
            "=" * 80,
            f"Timestamp: {results.get('timestamp', 'N/A')}",
            f"Domains: {', '.join(results.get('domains', []))}",
            ""
        ]
        
        # In-domain summary
        if 'in_domain' in results:
            report_lines.append("IN-DOMAIN EVALUATION:")
            report_lines.append("-" * 80)
            for domain, domain_results in results['in_domain'].items():
                metrics = domain_results['metrics']
                report_lines.append(
                    f"  {domain}: F1={metrics['f1']:.4f}, "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"Prec={metrics['precision']:.4f}, "
                    f"Rec={metrics['recall']:.4f}"
                )
            report_lines.append("")
        
        # Cross-domain summary
        if 'cross_domain' in results:
            report_lines.append("CROSS-DOMAIN EVALUATION (F1-Scores):")
            report_lines.append("-" * 80)
            
            # Create performance matrix
            matrix = self.cross_domain_evaluator.create_performance_matrix(
                results['cross_domain'],
                metric='f1'
            )
            report_lines.append(matrix.to_string())
            report_lines.append("")
        
        # Perturbation summary
        if 'perturbation' in results:
            report_lines.append("PERTURBATION ROBUSTNESS:")
            report_lines.append("-" * 80)
            for domain, pert_results in results['perturbation'].items():
                report_lines.append(f"  {domain}:")
                for level in ['clean', 'low', 'medium', 'high']:
                    if level in pert_results:
                        metrics = pert_results[level]['metrics']
                        f1 = metrics['f1']
                        report_lines.append(f"    {level.capitalize()}: F1={f1:.4f}")
                        
                        if level != 'clean' and 'robustness_metrics' in pert_results[level]:
                            rob = pert_results[level]['robustness_metrics']
                            drop = rob.get('f1_drop', 0)
                            drop_pct = rob.get('f1_drop_pct', 0)
                            report_lines.append(
                                f"      Drop: {drop:.4f} ({drop_pct:.2f}%)"
                            )
                report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {report_path}")
        
        return report
