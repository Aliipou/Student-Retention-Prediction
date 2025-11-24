"""
Evaluation Module for Student Retention Prediction
Comprehensive model evaluation with metrics, visualizations, and interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import shap
import os
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""

    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: list = None, output_dir: str = 'assets'):
        """
        Initialize evaluator.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            output_dir: Directory to save plots
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_test.shape[1])]
        self.output_dir = output_dir

        # Make predictions
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
            'pr_auc': average_precision_score(self.y_test, self.y_pred_proba),
            'log_loss': log_loss(self.y_test, self.y_pred_proba),
            'brier_score': brier_score_loss(self.y_test, self.y_pred_proba)
        }

        return metrics

    def print_metrics(self):
        """Print evaluation metrics in formatted way."""
        metrics = self.compute_metrics()

        print("\n=== Model Evaluation Metrics ===")
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1 Score:        {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
        print(f"Log Loss:        {metrics['log_loss']:.4f}")
        print(f"Brier Score:     {metrics['brier_score']:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(self.y_test, self.y_pred,
                                   target_names=['Retained', 'Dropout']))

    def plot_confusion_matrix(self, save: bool = True):
        """
        Plot confusion matrix.

        Args:
            save: Whether to save plot to file
        """
        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Retained', 'Dropout'],
                   yticklabels=['Retained', 'Dropout'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {self.output_dir}/confusion_matrix.png")

        plt.close()

    def plot_roc_curve(self, save: bool = True):
        """
        Plot ROC curve.

        Args:
            save: Whether to save plot to file
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {self.output_dir}/roc_curve.png")

        plt.close()

    def plot_precision_recall_curve(self, save: bool = True):
        """
        Plot Precision-Recall curve.

        Args:
            save: Whether to save plot to file
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        ap = average_precision_score(self.y_test, self.y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {self.output_dir}/precision_recall_curve.png")

        plt.close()

    def plot_calibration_curve(self, n_bins: int = 10, save: bool = True):
        """
        Plot calibration curve to assess probability calibration.

        Args:
            n_bins: Number of bins for calibration
            save: Whether to save plot to file
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=n_bins, strategy='uniform'
        )

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', linewidth=2,
                label='Model')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/calibration_curve.png', dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {self.output_dir}/calibration_curve.png")

        plt.close()

    def plot_prediction_distribution(self, save: bool = True):
        """
        Plot distribution of predicted probabilities by true class.

        Args:
            save: Whether to save plot to file
        """
        plt.figure(figsize=(10, 6))

        plt.hist(self.y_pred_proba[self.y_test == 0], bins=50, alpha=0.6,
                label='Retained Students', color='green', edgecolor='black')
        plt.hist(self.y_pred_proba[self.y_test == 1], bins=50, alpha=0.6,
                label='Dropout Students', color='red', edgecolor='black')

        plt.xlabel('Predicted Dropout Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/prediction_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Prediction distribution saved to {self.output_dir}/prediction_distribution.png")

        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save: bool = True):
        """
        Plot feature importance from model.

        Args:
            top_n: Number of top features to display
            save: Whether to save plot to file
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return

        importance = self.model.feature_importances_
        indices = np.argsort(importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance saved to {self.output_dir}/feature_importance.png")

        plt.close()

    def compute_shap_values(self, sample_size: int = 1000) -> Tuple[Any, np.ndarray]:
        """
        Compute SHAP values for model interpretability.

        Args:
            sample_size: Number of samples to use for SHAP (for performance)

        Returns:
            Tuple of (explainer, shap_values)
        """
        print("\n=== Computing SHAP Values ===")

        # Sample data for performance
        if len(self.X_test) > sample_size:
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[indices]
        else:
            X_sample = self.X_test

        # Create appropriate explainer based on model type
        try:
            # Try TreeExplainer first (for tree-based models)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, handle different return formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

        except Exception as e:
            # Fallback to KernelExplainer
            print(f"TreeExplainer failed ({e}), using KernelExplainer (slower)")
            background = shap.sample(self.X_test, 100)
            explainer = shap.KernelExplainer(self.model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        print(f"SHAP values computed for {X_sample.shape[0]} samples")
        return explainer, shap_values

    def plot_shap_summary(self, sample_size: int = 1000, save: bool = True):
        """
        Plot SHAP summary plot.

        Args:
            sample_size: Number of samples to use
            save: Whether to save plot to file
        """
        _, shap_values = self.compute_shap_values(sample_size)

        # Sample data for plotting
        if len(self.X_test) > sample_size:
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[indices]
        else:
            X_sample = self.X_test

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
            print(f"SHAP summary saved to {self.output_dir}/shap_summary.png")

        plt.close()

    def plot_shap_bar(self, sample_size: int = 1000, save: bool = True):
        """
        Plot SHAP bar plot showing mean absolute SHAP values.

        Args:
            sample_size: Number of samples to use
            save: Whether to save plot to file
        """
        _, shap_values = self.compute_shap_values(sample_size)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.output_dir}/shap_bar.png', dpi=300, bbox_inches='tight')
            print(f"SHAP bar plot saved to {self.output_dir}/shap_bar.png")

        plt.close()

    def generate_full_report(self):
        """Generate complete evaluation report with all metrics and plots."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60)

        # Print metrics
        self.print_metrics()

        # Generate all plots
        print("\n=== Generating Visualizations ===")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_calibration_curve()
        self.plot_prediction_distribution()
        self.plot_feature_importance()

        # SHAP analysis
        try:
            self.plot_shap_summary()
            self.plot_shap_bar()
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")

        print("\n" + "="*60)
        print("EVALUATION REPORT COMPLETE")
        print(f"All plots saved to: {self.output_dir}/")
        print("="*60)


def compare_models(evaluators: Dict[str, ModelEvaluator], output_dir: str = 'assets'):
    """
    Compare multiple models side by side.

    Args:
        evaluators: Dictionary of model name -> ModelEvaluator
        output_dir: Directory to save comparison plots
    """
    print("\n=== Comparing Models ===")

    # Collect metrics from all models
    metrics_df = pd.DataFrame({
        name: evaluator.compute_metrics()
        for name, evaluator in evaluators.items()
    }).T

    print("\n=== Model Comparison ===")
    print(metrics_df.round(4))

    # Plot ROC curves
    plt.figure(figsize=(10, 7))
    for name, evaluator in evaluators.items():
        fpr, tpr, _ = roc_curve(evaluator.y_test, evaluator.y_pred_proba)
        auc = roc_auc_score(evaluator.y_test, evaluator.y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_roc.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {output_dir}/model_comparison_roc.png")
    plt.close()

    # Save metrics comparison
    metrics_df.to_csv(f'{output_dir}/model_comparison_metrics.csv')
    print(f"Metrics comparison saved to {output_dir}/model_comparison_metrics.csv")


def main():
    """Evaluation module demonstration."""
    print("=== Evaluation Module ===")
    print("This module provides comprehensive model evaluation including:")
    print("- Classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)")
    print("- Confusion matrix")
    print("- ROC and Precision-Recall curves")
    print("- Calibration curves")
    print("- Prediction distribution plots")
    print("- Feature importance visualization")
    print("- SHAP analysis for model interpretability")


if __name__ == "__main__":
    main()
