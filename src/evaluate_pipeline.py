"""
Evaluation Pipeline for IndoBERT Clickbait Detection System

This script orchestrates comprehensive evaluation:
1. In-domain evaluation (train and test on same domain)
2. Cross-domain evaluation (5×5 matrix)
3. Perturbation robustness testing (3 intensity levels)
4. Statistical significance analysis
5. Results aggregation and visualization

Usage:
    # Evaluate single model
    python evaluate_pipeline.py --model base
    
    # Evaluate all models
    python evaluate_pipeline.py --model all
    
    # Skip perturbation testing (faster)
    python evaluate_pipeline.py --model base --skip-perturbation
    
    # On Colab
    !python evaluate_pipeline.py --model base --device cuda
"""

import glob
import os
import re
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config import config
from data_manager import DataManager
from model_trainer import ModelTrainer
from evaluation_engine import EvaluationEngine
from perturbation_engine import PerturbationEngine
from statistical_analyzer import StatisticalAnalyzer
from utils import file_manager, reproducibility, Timer, get_timestamp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates comprehensive evaluation of trained IndoBERT models."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        dataset_dir: str = "dataset",
        output_dir: str = "evaluation_results",
        device: str = "auto",
        random_seed: int = 42
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            checkpoint_dir: Directory containing trained model checkpoints
            dataset_dir: Directory containing CSV files
            output_dir: Directory for evaluation results
            device: Device to use ('cuda', 'cpu', or 'auto')
            random_seed: Random seed for reproducibility
        """
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.device = device
        self.random_seed = random_seed
        
        reproducibility.set_seed(random_seed)
        file_manager.ensure_directory(output_dir)
        
        logger.info("Evaluation Pipeline initialized")
        logger.info(f"Checkpoints: {checkpoint_dir}, Dataset: {dataset_dir}, Output: {output_dir}")
        logger.info(f"Device: {device}, Random seed: {random_seed}")
    
    def load_data(self) -> Dict[str, Any]:
        """
        Load datasets and organize by domain.
        
        Returns:
            Dictionary with data_manager and domain-organized test data
        """
        logger.info("\n[STEP 1] Loading Data")
        
        with Timer() as timer:
            data_manager = DataManager.from_dataset_directory(
                dataset_dir=self.dataset_dir,
                tokenizer_name=config.model.DEFAULT_MODEL,
                max_length=config.model.MAX_SEQUENCE_LENGTH,
                random_seed=self.random_seed
            )
            
            summary = data_manager.get_summary()
            logger.info(f"Loaded {summary['total_samples']} samples from {len(summary['domains'])} domains")
            
            splits_dir = os.path.join('output', 'data_splits')
            if os.path.exists(splits_dir):
                logger.info(f"Loading existing splits from {splits_dir}")
                train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
                val_df = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
                test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
            else:
                logger.info("Creating new stratified splits")
                train_df, val_df, test_df = data_manager.stratified_split(
                    train_size=config.data.TRAIN_SIZE,
                    val_size=config.data.VAL_SIZE,
                    test_size=config.data.TEST_SIZE
                )
            
            test_data_by_domain = data_manager.organize_by_domain(test_df)
        
        logger.info(f"Data loading completed in {timer.elapsed():.2f}s")
        
        return {
            'data_manager': data_manager,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'test_data_by_domain': test_data_by_domain
        }
    
    def load_models(self, model_names: List[str]) -> Dict[str, Dict[str, ModelTrainer]]:
        logger.info("\n[STEP 2] Loading SPECIALIST Models")
        
        model_mapping = {
            'base': config.model.INDOBERT_BASE,
            'large': config.model.INDOBERT_LARGE,
            'lite': config.model.INDOBERT_LITE
        }
        
        all_models = {}
        
        for model_name in model_names:
            full_model_name = model_mapping[model_name]
            domain_trainers = {}
            
            # FIX: Loop through the 5 domains and load the SPECIFIC model for each
            for domain in config.data.DOMAINS:
                model_checkpoint_dir = os.path.join(self.checkpoint_dir, model_name, domain.capitalize())
                checkpoint_files = glob.glob(os.path.join(model_checkpoint_dir, 'best_model_f1_*.pt'))
                
                if not checkpoint_files:
                    logger.warning(f"No checkpoint found for {model_name} -> {domain}")
                    continue
                
                def get_f1(path):
                    m = re.search(r'best_model_f1_(\d+\.\d+)\.pt', os.path.basename(path))
                    return float(m.group(1)) if m else 0

                checkpoint_path = max(checkpoint_files, key=get_f1)
                logger.info(f"Loading {domain} specialist from {checkpoint_path}")
                
                trainer = ModelTrainer(
                    model_name=full_model_name,
                    max_length=config.model.MAX_SEQUENCE_LENGTH,
                    device=self.device,
                    random_seed=self.random_seed
                )
                trainer.load_checkpoint(checkpoint_path)
                
                # Assign the specific specialist trainer to its specific domain key
                domain_trainers[domain] = trainer
            
            all_models[model_name] = domain_trainers
        
        return all_models
    
    def evaluate_single_model(
        self,
        model_name: str,
        domain_trainers: Dict[str, ModelTrainer],
        test_data_by_domain: Dict[str, pd.DataFrame],
        skip_perturbation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name of model variant
            domain_trainers: Dictionary mapping domains to trainers
            test_data_by_domain: Test data organized by domain
            skip_perturbation: Whether to skip perturbation testing
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"\n[STEP 3] Evaluating {model_name.upper()} Model")
        
        total_timer = Timer()
        total_timer.start()
        
        perturbation_engine = PerturbationEngine(random_seed=self.random_seed)
        
        model_output_dir = os.path.join(self.output_dir, model_name)
        eval_engine = EvaluationEngine(
            model_trainers=domain_trainers,
            perturbation_engine=perturbation_engine,
            output_dir=model_output_dir
        )
        
        results = eval_engine.run_complete_evaluation(
            test_data=test_data_by_domain,
            include_perturbations=not skip_perturbation
        )
        print(type(results['cross_domain']))
        print(results['cross_domain'])

        # # Convert cross-domain tuple keys
        # if 'cross_domain' in results:
        #     cross_domain_json = {}

        #     for (source, target), result in results['cross_domain'].items():
        #         if source not in cross_domain_json:
        #             cross_domain_json[source] = {}

        #         cross_domain_json[source][target] = result

        #     results['cross_domain'] = cross_domain_json

        # Capture the output of save_results into the results_file variable
        results_file = eval_engine.save_results(results, filename='complete_evaluation.json')
        logger.info(f"Results saved to {results_file}")
        
        total_time = total_timer.stop()
        logger.info(f"Evaluation completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        return results
    
    def perform_statistical_analysis(
        self,
        all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis using sample-level performance scores.
        """
        logger.info("\n[STEP 4] Statistical Analysis (Sample-Level)")
        
        if len(all_results) < 2:
            logger.info("Skipping statistical analysis (need at least 2 models)")
            return {'status': 'skipped', 'reason': 'insufficient_models'}
        
        with Timer() as timer:
            analyzer = StatisticalAnalyzer()
            
            # Use a dictionary to store arrays of scores for ALL samples
            model_sample_scores = {}
            
            for model_name, results in all_results.items():
                all_model_scores = []
                
                # Iterate through every domain in this model
                for domain, domain_results in results['in_domain'].items():
                    # Extract raw probabilities: list of [prob_non_clickbait, prob_clickbait]
                    probs = np.array(domain_results['probabilities'])
                    
                    # We take the probability of the positive class (clickbait)
                    # This gives us N=75 (or more) data points per model instead of N=5
                    clickbait_scores = probs[:, 1] 
                    all_model_scores.extend(clickbait_scores.tolist())
                
                model_sample_scores[model_name] = np.array(all_model_scores)
            
            # Perform pairwise comparisons
            model_names = list(model_sample_scores.keys())
            comparisons = {}
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a, model_b = model_names[i], model_names[j]
                    
                    logger.info(f"Comparing {model_a} vs {model_b} (N={len(model_sample_scores[model_a])})")
                    
                    # The StatisticalAnalyzer will now receive N ~ 375 scores
                    comparison = analyzer.analyze_model_comparison(
                        model_a_scores=model_sample_scores[model_a],
                        model_b_scores=model_sample_scores[model_b],
                        model_a_name=model_a,
                        model_b_name=model_b
                    )
                    comparisons[f"{model_a}_vs_{model_b}"] = comparison
        
        return {
            'status': 'completed',
            'comparisons': comparisons,
            'n_samples_per_model': len(model_sample_scores[model_names[0]])
        }
    
    def generate_summary(
        self,
        all_results: Dict[str, Dict[str, Any]],
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evaluation summary.
        
        Args:
            all_results: Dictionary mapping model names to their results
            statistical_analysis: Statistical analysis results
            
        Returns:
            Summary dictionary
        """
        logger.info("\n[STEP 5] Generating Summary")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(all_results.keys()),
            'model_summaries': {}
        }
        
        for model_name, results in all_results.items():
            in_domain_f1 = []
            for domain, metrics in results['in_domain'].items():
                in_domain_f1.append(metrics['metrics']['f1'])
            
            avg_in_domain_f1 = sum(in_domain_f1) / len(in_domain_f1) if in_domain_f1 else 0
            
            cross_domain_f1 = []
            if 'cross_domain' in results:
                for (source, target), metrics_data in results['cross_domain'].items():
                    if source != target:
                        cross_domain_f1.append(metrics_data['metrics']['f1'])            
            avg_cross_domain_f1 = sum(cross_domain_f1) / len(cross_domain_f1) if cross_domain_f1 else 0
            
            model_summary = {
                'avg_in_domain_f1': avg_in_domain_f1,
                'avg_cross_domain_f1': avg_cross_domain_f1,
                'in_domain_f1_scores': in_domain_f1
            }
            
            if 'perturbation' in results and results['perturbation']:
                perturbation_f1 = []
                for domain, levels in results['perturbation'].items():
                    for level in ['low', 'medium', 'high']:
                        if level in levels:
                            perturbation_f1.append(levels[level]['metrics']['f1'])
                
                model_summary['avg_perturbation_f1'] = sum(perturbation_f1) / len(perturbation_f1) if perturbation_f1 else 0
            
            summary['model_summaries'][model_name] = model_summary
        
        summary['statistical_analysis'] = statistical_analysis
        
        summary_file = os.path.join(self.output_dir, 'evaluation_summary.json')
        file_manager.save_json(summary, summary_file)
        logger.info(f"Summary saved to {summary_file}")
        
        logger.info("\nEvaluation Summary:")
        for model_name, metrics in summary['model_summaries'].items():
            logger.info(f"  {model_name.upper()}:")
            logger.info(f"    Avg In-Domain F1: {metrics['avg_in_domain_f1']:.4f}")
            logger.info(f"    Avg Cross-Domain F1: {metrics['avg_cross_domain_f1']:.4f}")
            if 'avg_perturbation_f1' in metrics:
                logger.info(f"    Avg Perturbation F1: {metrics['avg_perturbation_f1']:.4f}")
        
        return summary
    
    def run(
        self,
        models: List[str] = ['base'],
        skip_perturbation: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            models: List of model names to evaluate
            skip_perturbation: Whether to skip perturbation testing
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("\nStarting Evaluation Pipeline")
        logger.info(f"Models: {models}, Skip perturbation: {skip_perturbation}")
        
        total_timer = Timer()
        total_timer.start()
        
        data = self.load_data()
        
        if 'all' in models:
            models = ['base', 'large', 'lite']
        
        all_model_trainers = self.load_models(models)
        
        if not all_model_trainers:
            logger.error("No models loaded. Please train models first using train_pipeline.py")
            return {'status': 'failed', 'reason': 'no_models_loaded'}
        
        all_results = {}
        for model_name, domain_trainers in all_model_trainers.items():
            try:
                results = self.evaluate_single_model(
                    model_name=model_name,
                    domain_trainers=domain_trainers,
                    test_data_by_domain=data['test_data_by_domain'],
                    skip_perturbation=skip_perturbation
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        statistical_analysis = self.perform_statistical_analysis(all_results)
        summary = self.generate_summary(all_results, statistical_analysis)
        
        total_time = total_timer.stop()
        logger.info(f"\nPipeline completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        return {
            'status': 'success',
            'results': all_results,
            'statistical_analysis': statistical_analysis,
            'summary': summary
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained IndoBERT models for clickbait detection'
    )
    
    parser.add_argument(
        '--model', type=str, default='base',
        choices=['base', 'large', 'lite', 'all'],
        help='Model variant to evaluate (default: base)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory containing trained model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--dataset-dir', type=str, default='dataset',
        help='Directory containing CSV files (default: dataset)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='evaluation_results',
        help='Directory for evaluation results (default: evaluation_results)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation (default: auto)'
    )
    parser.add_argument(
        '--skip-perturbation', action='store_true',
        help='Skip perturbation robustness testing (faster evaluation)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        pipeline = EvaluationPipeline(
            checkpoint_dir=args.checkpoint_dir,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            device=args.device,
            random_seed=args.seed
        )
        
        models = [args.model] if args.model != 'all' else ['base', 'large', 'lite']
        
        results = pipeline.run(
            models=models,
            skip_perturbation=args.skip_perturbation
        )
        
        if results['status'] == 'success':
            logger.info("Evaluation pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error(f"Evaluation pipeline failed: {results.get('reason', 'unknown')}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nEvaluation pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
