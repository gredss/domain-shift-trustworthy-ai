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

import os
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
        """
        Load trained models from checkpoints.
        
        Args:
            model_names: List of model variant names
            
        Returns:
            Dictionary mapping model names to domain-trainer dictionaries
        """
        logger.info("\n[STEP 2] Loading Models")
        
        model_mapping = {
            'base': config.model.INDOBERT_BASE,
            'large': config.model.INDOBERT_LARGE,
            'lite': config.model.INDOBERT_LITE
        }
        
        all_models = {}
        
        for model_name in model_names:
            if model_name not in model_mapping:
                logger.warning(f"Invalid model name: {model_name}, skipping")
                continue
            
            full_model_name = model_mapping[model_name]
            model_checkpoint_dir = os.path.join(self.checkpoint_dir, model_name)
            checkpoint_path = os.path.join(model_checkpoint_dir, 'best_model.pt')
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping {model_name}")
                continue
            
            logger.info(f"Loading {model_name} model from {checkpoint_path}")
            
            trainer = ModelTrainer(
                model_name=full_model_name,
                max_length=config.model.MAX_SEQUENCE_LENGTH,
                device=self.device,
                random_seed=self.random_seed
            )
            
            trainer.load_checkpoint(checkpoint_path)
            
            domain_trainers = {
                domain: trainer for domain in config.data.DOMAINS
            }
            
            all_models[model_name] = domain_trainers
        
        logger.info(f"Loaded {len(all_models)} model(s)")
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
        
        results_file = os.path.join(model_output_dir, 'complete_evaluation.json')
        file_manager.save_json(results, results_file)
        logger.info(f"Results saved to {results_file}")
        
        total_time = total_timer.stop()
        logger.info(f"Evaluation completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        return results
    
    def perform_statistical_analysis(
        self,
        all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis across models.
        
        Args:
            all_results: Dictionary mapping model names to their results
            
        Returns:
            Dictionary with statistical analysis results
        """
        logger.info("\n[STEP 4] Statistical Analysis")
        
        if len(all_results) < 2:
            logger.info("Skipping statistical analysis (need at least 2 models)")
            return {'status': 'skipped', 'reason': 'insufficient_models'}
        
        with Timer() as timer:
            analyzer = StatisticalAnalyzer()
            
            model_f1_scores = {}
            for model_name, results in all_results.items():
                f1_scores = []
                for domain, metrics in results['in_domain'].items():
                    f1_scores.append(metrics['metrics']['f1'])
                model_f1_scores[model_name] = f1_scores
            
            model_names = list(model_f1_scores.keys())
            comparisons = {}
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a = model_names[i]
                    model_b = model_names[j]
                    
                    logger.info(f"Comparing {model_a} vs {model_b}")
                    
                    comparison = analyzer.analyze_model_comparison(
                        model_a_scores=np.array(model_f1_scores[model_a]),
                        model_b_scores=np.array(model_f1_scores[model_b]),
                        model_a_name=model_a,
                        model_b_name=model_b
                    )
                    
                    comparisons[f"{model_a}_vs_{model_b}"] = comparison
        
        logger.info(f"Statistical analysis completed in {timer.elapsed():.2f}s")
        
        return {
            'status': 'completed',
            'comparisons': comparisons,
            'model_f1_scores': model_f1_scores
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
            if 'cross_domain' in results and 'transfer_matrix' in results['cross_domain']:
                for source in results['cross_domain']['transfer_matrix']:
                    for target, metrics in results['cross_domain']['transfer_matrix'][source].items():
                        if source != target:
                            cross_domain_f1.append(metrics['metrics']['f1'])
            
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
            logger.info("✅ Evaluation pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error(f"❌ Evaluation pipeline failed: {results.get('reason', 'unknown')}")
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