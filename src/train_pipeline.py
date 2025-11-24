"""
Complete Training Pipeline for Student Retention Prediction
Orchestrates data generation, preprocessing, training, and evaluation.
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime

from data_generator import StudentDataGenerator
from preprocessing import DataPreprocessor
from models import StudentRetentionModel, ModelEnsemble
from evaluation import ModelEvaluator, compare_models


def main(args):
    """Run complete training pipeline."""

    print("\n" + "="*70)
    print("STUDENT RETENTION PREDICTION - TRAINING PIPELINE")
    print("="*70)

    start_time = datetime.now()

    # Step 1: Generate Data (if needed)
    print("\n" + "="*70)
    print("STEP 1: DATA GENERATION")
    print("="*70)

    data_path = 'data/student_data.csv'

    if args.generate_data or not os.path.exists(data_path):
        print("Generating synthetic student data...")
        generator = StudentDataGenerator(n_samples=args.n_samples, random_state=42)
        df = generator.generate()
        generator.save(df, data_path)
        df.to_parquet('data/student_data.parquet', index=False)
        print("Data generation complete!")
    else:
        print(f"Using existing data from {data_path}")

    # Step 2: Load and Preprocess Data
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)

    print(f"\nDataset statistics:")
    print(f"  Total students: {len(df):,}")
    print(f"  Dropout rate: {df['dropout_risk'].mean():.2%}")
    print(f"  Features: {df.shape[1]}")

    # Prepare features
    X, y = preprocessor.prepare_features(df, fit=True)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )

    # Save preprocessor
    preprocessor.save()

    # Step 3: Train Models
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)

    models_to_train = args.models.split(',') if args.models != 'all' else ['random_forest', 'xgboost', 'lightgbm']

    trained_models = {}
    evaluators = {}

    for model_type in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*70}")

        model = StudentRetentionModel(model_type=model_type, random_state=42)

        if args.tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            model.tune_hyperparameters(X_train, y_train, cv=3)
        else:
            print("Training with default parameters...")
            model.train(X_train, y_train, X_val, y_val)

        # Save model
        model.save(f'models/{model_type}_model.joblib')

        trained_models[model_type] = model

    # Step 4: Evaluate Models
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)

    best_model = None
    best_auc = 0
    best_model_name = None

    for model_name, model in trained_models.items():
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name.upper()} Model")
        print(f"{'='*70}")

        evaluator = ModelEvaluator(
            model=model.model,
            X_test=X_test,
            y_test=y_test,
            feature_names=preprocessor.feature_columns,
            output_dir=f'assets/{model_name}'
        )

        evaluator.generate_full_report()
        evaluators[model_name] = evaluator

        # Track best model
        metrics = evaluator.compute_metrics()
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = model
            best_model_name = model_name

    # Step 5: Compare Models
    if len(evaluators) > 1:
        print("\n" + "="*70)
        print("STEP 5: MODEL COMPARISON")
        print("="*70)

        compare_models(evaluators, output_dir='assets')

    # Save best model
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model_name.upper()} (ROC-AUC: {best_auc:.4f})")
    print("="*70)

    best_model.save('models/best_model.joblib')
    print(f"Best model saved to models/best_model.joblib")

    # Step 6: Generate Predictions on Full Dataset
    print("\n" + "="*70)
    print("STEP 6: GENERATING PREDICTIONS")
    print("="*70)

    # Load original data and add predictions
    df_full = pd.read_csv(data_path)
    df_for_pred = df_full.drop(['student_id', 'dropout_risk'], axis=1, errors='ignore')

    preprocessor_new = DataPreprocessor()
    preprocessor_new.load()

    X_full, _ = preprocessor_new.prepare_features(df_for_pred, fit=False)
    predictions = best_model.predict_proba(X_full)[:, 1]

    df_full['predicted_risk'] = predictions
    df_full['risk_category'] = pd.cut(
        predictions,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Save predictions
    df_full.to_csv('data/student_data_with_predictions.csv', index=False)
    print(f"Predictions saved to data/student_data_with_predictions.csv")

    # Summary statistics
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total students: {len(df_full):,}")
    print(f"\nRisk Distribution:")
    print(df_full['risk_category'].value_counts())
    print(f"\nAverage predicted risk: {predictions.mean():.2%}")
    print(f"Actual dropout rate: {df_full['dropout_risk'].mean():.2%}")

    # Pipeline completion
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total time: {duration:.1f} seconds")
    print(f"\nNext steps:")
    print(f"  1. Review evaluation plots in: assets/")
    print(f"  2. Launch dashboard: streamlit run src/dashboard.py")
    print(f"  3. Review predictions: data/student_data_with_predictions.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Student Retention Prediction Models')

    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new synthetic data')
    parser.add_argument('--n-samples', type=int, default=20000,
                       help='Number of samples to generate (default: 20000)')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models or "all" (default: all)')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation set size (default: 0.1)')

    args = parser.parse_args()

    main(args)
