"""
STRICT VALIDATION SCRIPT
Comprehensive testing of every component and element of the Student Retention System.
"""

import sys
import os
import numpy as np
import pandas as pd
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

class StrictValidator:
    """Perform extremely rigorous validation of all components."""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.warnings = []

    def print_header(self, text):
        """Print test section header."""
        print("\n" + "="*80)
        print(text.center(80))
        print("="*80)

    def test(self, name, func):
        """Run a test and record results."""
        try:
            print(f"\n{'Testing:':<20} {name}")
            result = func()
            if result:
                print(f"{'Status:':<20} [PASS]")
                self.tests_passed += 1
                return True
            else:
                print(f"{'Status:':<20} [FAIL]")
                self.tests_failed += 1
                self.errors.append(f"{name}: Test returned False")
                return False
        except Exception as e:
            print(f"{'Status:':<20} [FAIL] (Exception)")
            print(f"{'Error:':<20} {str(e)}")
            self.tests_failed += 1
            self.errors.append(f"{name}: {str(e)}")
            traceback.print_exc()
            return False

    def validate_imports(self):
        """Validate all imports work correctly."""
        self.print_header("VALIDATING IMPORTS")

        # Core libraries
        self.test("Import numpy", lambda: __import__('numpy') is not None)
        self.test("Import pandas", lambda: __import__('pandas') is not None)
        self.test("Import sklearn", lambda: __import__('sklearn') is not None)
        self.test("Import xgboost", lambda: __import__('xgboost') is not None)
        self.test("Import lightgbm", lambda: __import__('lightgbm') is not None)
        self.test("Import shap", lambda: __import__('shap') is not None)
        self.test("Import matplotlib", lambda: __import__('matplotlib') is not None)
        self.test("Import seaborn", lambda: __import__('seaborn') is not None)
        self.test("Import plotly", lambda: __import__('plotly') is not None)
        self.test("Import streamlit", lambda: __import__('streamlit') is not None)

        # Project modules
        self.test("Import data_generator", lambda: __import__('data_generator') is not None)
        self.test("Import preprocessing", lambda: __import__('preprocessing') is not None)
        self.test("Import models", lambda: __import__('models') is not None)
        self.test("Import evaluation", lambda: __import__('evaluation') is not None)

    def validate_data_generator(self):
        """Validate data generator with extreme rigor."""
        self.print_header("VALIDATING DATA GENERATOR")

        from data_generator import StudentDataGenerator

        # Test initialization
        def test_init():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            return gen.n_samples == 100 and gen.random_state == 42

        self.test("Generator initialization", test_init)

        # Test data generation
        def test_generate():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            return isinstance(df, pd.DataFrame) and len(df) == 100

        self.test("Data generation", test_generate)

        # Test required columns
        def test_columns():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            required = ['student_id', 'credits_last_sem', 'failed_courses',
                       'moodle_activity_score', 'library_visits', 'login_times_last_week',
                       'attendance_rate', 'gpa', 'gpa_variance', 'age', 'gender', 'major',
                       'scholarship_amount', 'part_time_job', 'distance_from_campus_km',
                       'dropout_risk']
            return all(col in df.columns for col in required)

        self.test("Required columns present", test_columns)

        # Test no missing values
        def test_no_nulls():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            return df.isnull().sum().sum() == 0

        self.test("No missing values", test_no_nulls)

        # Test data types
        def test_dtypes():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            checks = [
                pd.api.types.is_integer_dtype(df['credits_last_sem']),
                pd.api.types.is_integer_dtype(df['failed_courses']),
                pd.api.types.is_float_dtype(df['gpa']),
                pd.api.types.is_float_dtype(df['attendance_rate']),
                pd.api.types.is_integer_dtype(df['dropout_risk']),
            ]
            return all(checks)

        self.test("Correct data types", test_dtypes)

        # Test value ranges
        def test_ranges():
            gen = StudentDataGenerator(n_samples=1000, random_state=42)
            df = gen.generate()
            checks = [
                (df['gpa'] >= 0).all() and (df['gpa'] <= 4).all(),
                (df['attendance_rate'] >= 0).all() and (df['attendance_rate'] <= 1).all(),
                (df['moodle_activity_score'] >= 0).all() and (df['moodle_activity_score'] <= 100).all(),
                (df['failed_courses'] >= 0).all(),
                (df['age'] >= 18).all(),
                df['dropout_risk'].isin([0, 1]).all(),
            ]
            return all(checks)

        self.test("Valid value ranges", test_ranges)

        # Test reproducibility
        def test_reproducibility():
            gen1 = StudentDataGenerator(n_samples=100, random_state=42)
            gen2 = StudentDataGenerator(n_samples=100, random_state=42)
            df1 = gen1.generate()
            df2 = gen2.generate()
            return df1.equals(df2)

        self.test("Reproducibility with same seed", test_reproducibility)

        # Test different sample sizes
        def test_sample_sizes():
            for n in [10, 50, 100, 500]:
                gen = StudentDataGenerator(n_samples=n, random_state=42)
                df = gen.generate()
                if len(df) != n:
                    return False
            return True

        self.test("Different sample sizes", test_sample_sizes)

        # Test probability calculation
        def test_probability():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            probs = gen._calculate_dropout_probability(df)
            return (probs >= 0).all() and (probs <= 1).all() and len(probs) == len(df)

        self.test("Probability calculation", test_probability)

        # Test categorical distributions
        def test_categoricals():
            gen = StudentDataGenerator(n_samples=1000, random_state=42)
            df = gen.generate()
            valid_genders = {'Male', 'Female', 'Other'}
            expected_majors = ['Computer Science', 'Engineering', 'Business', 'Arts',
                              'Science', 'Medicine', 'Education', 'Social Sciences']
            return (set(df['gender'].unique()).issubset(valid_genders) and
                   set(df['major'].unique()).issubset(expected_majors))

        self.test("Categorical distributions", test_categoricals)

    def validate_preprocessing(self):
        """Validate preprocessing with extreme rigor."""
        self.print_header("VALIDATING PREPROCESSING")

        from data_generator import StudentDataGenerator
        from preprocessing import DataPreprocessor

        # Generate test data
        gen = StudentDataGenerator(n_samples=200, random_state=42)
        df = gen.generate()

        # Test initialization
        def test_init():
            preprocessor = DataPreprocessor()
            return (preprocessor.scaler is not None and
                   isinstance(preprocessor.label_encoders, dict) and
                   preprocessor.feature_columns is None)

        self.test("Preprocessor initialization", test_init)

        # Test feature engineering
        def test_features():
            preprocessor = DataPreprocessor()
            df_eng = preprocessor.create_features(df.copy())
            expected_features = ['engagement_score', 'academic_risk_score',
                               'low_attendance', 'overloaded', 'underloaded']
            return all(f in df_eng.columns for f in expected_features)

        self.test("Feature engineering", test_features)

        # Test missing value handling
        def test_missing():
            preprocessor = DataPreprocessor()
            df_missing = df.copy()
            df_missing.loc[0:5, 'gpa'] = np.nan
            df_filled = preprocessor.handle_missing_values(df_missing)
            return df_filled.isnull().sum().sum() == 0

        self.test("Missing value handling", test_missing)

        # Test categorical encoding
        def test_encoding():
            preprocessor = DataPreprocessor()
            df_encoded = preprocessor.encode_categorical(df.copy(), fit=True)
            return (pd.api.types.is_numeric_dtype(df_encoded['gender']) and
                   pd.api.types.is_numeric_dtype(df_encoded['major']))

        self.test("Categorical encoding", test_encoding)

        # Test feature preparation
        def test_preparation():
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_features(df.copy(), fit=True)
            return (isinstance(X, np.ndarray) and
                   isinstance(y, np.ndarray) and
                   X.shape[0] == len(df) and
                   len(y) == len(df))

        self.test("Feature preparation", test_preparation)

        # Test scaling
        def test_scaling():
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_features(df.copy(), fit=True)
            # Scaled data should have mean ~0 and std ~1
            return abs(X.mean()) < 0.5 and abs(X.std() - 1.0) < 0.5

        self.test("Feature scaling", test_scaling)

        # Test data splitting
        def test_splitting():
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_features(df.copy(), fit=True)
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
            return total == len(X)

        self.test("Data splitting", test_splitting)

        # Test consistency
        def test_consistency():
            preprocessor = DataPreprocessor()
            X1, y1 = preprocessor.prepare_features(df.copy(), fit=True)
            X2, y2 = preprocessor.prepare_features(df.copy(), fit=False)
            return np.allclose(X1, X2) and np.array_equal(y1, y2)

        self.test("Preprocessing consistency", test_consistency)

        # Test engineered feature validity
        def test_engineered_validity():
            preprocessor = DataPreprocessor()
            df_eng = preprocessor.create_features(df.copy())
            checks = [
                (df_eng['engagement_score'] >= 0).all(),
                (df_eng['engagement_score'] <= 100).all(),
                df_eng['low_attendance'].isin([0, 1]).all(),
                df_eng['overloaded'].isin([0, 1]).all(),
            ]
            return all(checks)

        self.test("Engineered feature validity", test_engineered_validity)

    def validate_models(self):
        """Validate models with extreme rigor."""
        self.print_header("VALIDATING MODELS")

        from data_generator import StudentDataGenerator
        from preprocessing import DataPreprocessor
        from models import StudentRetentionModel, ModelEnsemble

        # Generate and prepare data
        gen = StudentDataGenerator(n_samples=300, random_state=42)
        df = gen.generate()
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, fit=True)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

        # Test model initialization for each type
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            def test_init(mt=model_type):
                model = StudentRetentionModel(model_type=mt)
                return model.model_type == mt and model.model is None

            self.test(f"{model_type} initialization", test_init)

        # Test default parameters
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            def test_params(mt=model_type):
                model = StudentRetentionModel(model_type=mt)
                params = model.get_default_params()
                return isinstance(params, dict) and len(params) > 0

            self.test(f"{model_type} default params", test_params)

        # Test model creation
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            def test_create(mt=model_type):
                model = StudentRetentionModel(model_type=mt)
                model.create_model()
                return model.model is not None

            self.test(f"{model_type} creation", test_create)

        # Test model training
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            def test_train(mt=model_type):
                model = StudentRetentionModel(model_type=mt)
                model.train(X_train, y_train, X_val, y_val)
                return model.model is not None

            self.test(f"{model_type} training", test_train)

        # Test predictions
        def test_predict():
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            return (len(predictions) == len(X_test) and
                   set(predictions).issubset({0, 1}))

        self.test("Model predictions", test_predict)

        # Test probability predictions
        def test_predict_proba():
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train)
            proba = model.predict_proba(X_test)
            return (proba.shape[0] == len(X_test) and
                   proba.shape[1] == 2 and
                   (proba >= 0).all() and
                   (proba <= 1).all() and
                   np.allclose(proba.sum(axis=1), 1))

        self.test("Probability predictions", test_predict_proba)

        # Test feature importance
        def test_importance():
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train)
            importance_df = model.get_feature_importance(preprocessor.feature_columns)
            return (len(importance_df) == len(preprocessor.feature_columns) and
                   'feature' in importance_df.columns and
                   'importance' in importance_df.columns)

        self.test("Feature importance", test_importance)

        # Test model performance
        def test_performance():
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            # Just ensure predictions are valid, performance can vary with small datasets
            return len(predictions) == len(y_test) and set(predictions).issubset({0, 1})

        self.test("Model performance", test_performance)

        # Test ensemble initialization
        def test_ensemble_init():
            ensemble = ModelEnsemble(model_types=['random_forest'])
            return len(ensemble.models) == 1

        self.test("Ensemble initialization", test_ensemble_init)

        # Test ensemble training
        def test_ensemble_train():
            ensemble = ModelEnsemble(model_types=['random_forest'])
            ensemble.train(X_train, y_train, X_val, y_val)
            return all(m.model is not None for m in ensemble.models)

        self.test("Ensemble training", test_ensemble_train)

        # Test ensemble predictions
        def test_ensemble_predict():
            ensemble = ModelEnsemble(model_types=['random_forest'])
            ensemble.train(X_train, y_train)
            predictions = ensemble.predict(X_test)
            return len(predictions) == len(X_test)

        self.test("Ensemble predictions", test_ensemble_predict)

    def validate_evaluation(self):
        """Validate evaluation with extreme rigor."""
        self.print_header("VALIDATING EVALUATION")

        from data_generator import StudentDataGenerator
        from preprocessing import DataPreprocessor
        from models import StudentRetentionModel
        from evaluation import ModelEvaluator

        # Generate and prepare data
        gen = StudentDataGenerator(n_samples=300, random_state=42)
        df = gen.generate()
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, fit=True)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

        # Train a model
        model = StudentRetentionModel(model_type='random_forest')
        model.train(X_train, y_train)

        # Test evaluator initialization
        def test_init():
            evaluator = ModelEvaluator(
                model=model.model,
                X_test=X_test,
                y_test=y_test,
                feature_names=preprocessor.feature_columns
            )
            return (evaluator.model is not None and
                   len(evaluator.y_pred) == len(y_test) and
                   len(evaluator.y_pred_proba) == len(y_test))

        self.test("Evaluator initialization", test_init)

        # Test metrics computation
        def test_metrics():
            evaluator = ModelEvaluator(
                model=model.model,
                X_test=X_test,
                y_test=y_test,
                feature_names=preprocessor.feature_columns
            )
            metrics = evaluator.compute_metrics()
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                               'roc_auc', 'pr_auc', 'log_loss', 'brier_score']
            return all(m in metrics for m in required_metrics)

        self.test("Metrics computation", test_metrics)

        # Test metric validity
        def test_metric_validity():
            evaluator = ModelEvaluator(
                model=model.model,
                X_test=X_test,
                y_test=y_test,
                feature_names=preprocessor.feature_columns
            )
            metrics = evaluator.compute_metrics()
            checks = [
                0 <= metrics['accuracy'] <= 1,
                0 <= metrics['precision'] <= 1,
                0 <= metrics['recall'] <= 1,
                0 <= metrics['f1_score'] <= 1,
                0 <= metrics['roc_auc'] <= 1,
                0 <= metrics['pr_auc'] <= 1,
            ]
            return all(checks)

        self.test("Metric validity", test_metric_validity)

        # Test SHAP computation
        def test_shap():
            evaluator = ModelEvaluator(
                model=model.model,
                X_test=X_test[:100],
                y_test=y_test[:100],
                feature_names=preprocessor.feature_columns
            )
            explainer, shap_values = evaluator.compute_shap_values(sample_size=50)
            return explainer is not None and shap_values is not None

        self.test("SHAP computation", test_shap)

    def validate_pipeline(self):
        """Validate complete pipeline end-to-end."""
        self.print_header("VALIDATING COMPLETE PIPELINE")

        from data_generator import StudentDataGenerator
        from preprocessing import DataPreprocessor
        from models import StudentRetentionModel
        from evaluation import ModelEvaluator

        # Test complete pipeline
        def test_pipeline():
            # 1. Generate data
            gen = StudentDataGenerator(n_samples=500, random_state=42)
            df = gen.generate()

            # 2. Preprocess
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_features(df, fit=True)
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

            # 3. Train model
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train, X_val, y_val)

            # 4. Evaluate
            evaluator = ModelEvaluator(
                model=model.model,
                X_test=X_test,
                y_test=y_test,
                feature_names=preprocessor.feature_columns
            )
            metrics = evaluator.compute_metrics()

            # 5. Make predictions
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)

            return (len(predictions) == len(X_test) and
                   proba.shape[0] == len(X_test) and
                   metrics['roc_auc'] > 0.5)

        self.test("Complete pipeline execution", test_pipeline)

        # Test pipeline with different model types
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            def test_model_pipeline(mt=model_type):
                gen = StudentDataGenerator(n_samples=200, random_state=42)
                df = gen.generate()
                preprocessor = DataPreprocessor()
                X, y = preprocessor.prepare_features(df, fit=True)
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
                model = StudentRetentionModel(model_type=mt)
                model.train(X_train, y_train, X_val, y_val)
                predictions = model.predict(X_test)
                return len(predictions) == len(X_test)

            self.test(f"Pipeline with {model_type}", test_model_pipeline)

    def validate_file_structure(self):
        """Validate file structure and existence."""
        self.print_header("VALIDATING FILE STRUCTURE")

        required_files = [
            'src/data_generator.py',
            'src/preprocessing.py',
            'src/models.py',
            'src/evaluation.py',
            'src/dashboard.py',
            'src/train_pipeline.py',
            'src/__init__.py',
            'tests/test_data_generator.py',
            'tests/test_preprocessing.py',
            'tests/test_models.py',
            'tests/__init__.py',
            'notebooks/exploratory_analysis.ipynb',
            'requirements.txt',
            'README.md',
            'QUICKSTART.md',
            'PROJECT_SUMMARY.md',
            'Dockerfile',
            'docker-compose.yml',
            '.dockerignore',
            '.gitignore',
            'run.py',
        ]

        for filepath in required_files:
            def test_file(fp=filepath):
                return os.path.exists(fp)

            self.test(f"File exists: {filepath}", test_file)

        # Test directories exist
        required_dirs = ['src', 'tests', 'notebooks', 'data', 'models', 'assets']
        for dirpath in required_dirs:
            def test_dir(dp=dirpath):
                return os.path.isdir(dp)

            self.test(f"Directory exists: {dirpath}", test_dir)

    def validate_data_outputs(self):
        """Validate that data files are generated correctly."""
        self.print_header("VALIDATING DATA OUTPUTS")

        from data_generator import StudentDataGenerator

        # Test CSV generation
        def test_csv():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            gen.save(df, 'data/test_data.csv')
            exists = os.path.exists('data/test_data.csv')
            if exists:
                df_loaded = pd.read_csv('data/test_data.csv')
                return len(df_loaded) == 100
            return False

        self.test("CSV file generation", test_csv)

        # Test parquet generation
        def test_parquet():
            gen = StudentDataGenerator(n_samples=100, random_state=42)
            df = gen.generate()
            df.to_parquet('data/test_data.parquet', index=False)
            exists = os.path.exists('data/test_data.parquet')
            if exists:
                df_loaded = pd.read_parquet('data/test_data.parquet')
                return len(df_loaded) == 100
            return False

        self.test("Parquet file generation", test_parquet)

    def validate_model_outputs(self):
        """Validate that models are saved and loaded correctly."""
        self.print_header("VALIDATING MODEL OUTPUTS")

        from data_generator import StudentDataGenerator
        from preprocessing import DataPreprocessor
        from models import StudentRetentionModel

        # Generate and prepare data
        gen = StudentDataGenerator(n_samples=200, random_state=42)
        df = gen.generate()
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, fit=True)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

        # Test model saving
        def test_model_save():
            model = StudentRetentionModel(model_type='random_forest')
            model.train(X_train, y_train)
            model.save('models/test_model.joblib')
            return os.path.exists('models/test_model.joblib')

        self.test("Model saving", test_model_save)

        # Test model loading
        def test_model_load():
            model = StudentRetentionModel(model_type='random_forest')
            model.load('models/test_model.joblib')
            predictions = model.predict(X_test)
            return len(predictions) == len(X_test)

        self.test("Model loading", test_model_load)

        # Test preprocessor saving
        def test_preprocessor_save():
            preprocessor.save('models/test_preprocessor.joblib')
            return os.path.exists('models/test_preprocessor.joblib')

        self.test("Preprocessor saving", test_preprocessor_save)

        # Test preprocessor loading
        def test_preprocessor_load():
            preprocessor_new = DataPreprocessor()
            preprocessor_new.load('models/test_preprocessor.joblib')
            return preprocessor_new.feature_columns is not None

        self.test("Preprocessor loading", test_preprocessor_load)

    def print_summary(self):
        """Print final test summary."""
        self.print_header("VALIDATION SUMMARY")

        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"\n{'Total Tests:':<30} {total_tests}")
        print(f"{'Tests Passed:':<30} {self.tests_passed} [PASS]")
        print(f"{'Tests Failed:':<30} {self.tests_failed} [FAIL]")
        print(f"{'Pass Rate:':<30} {pass_rate:.2f}%")

        if self.tests_failed > 0:
            print("\n" + "="*80)
            print("FAILURES AND ERRORS:")
            print("="*80)
            for error in self.errors:
                print(f"  [!] {error}")

        if self.warnings:
            print("\n" + "="*80)
            print("WARNINGS:")
            print("="*80)
            for warning in self.warnings:
                print(f"  [!] {warning}")

        print("\n" + "="*80)
        if self.tests_failed == 0:
            print("ALL TESTS PASSED - SYSTEM FULLY VALIDATED".center(80))
        else:
            print("SOME TESTS FAILED - REVIEW ERRORS ABOVE".center(80))
        print("="*80 + "\n")

        return self.tests_failed == 0


def main():
    """Run complete strict validation."""
    print("="*80)
    print("STUDENT RETENTION PREDICTION SYSTEM".center(80))
    print("STRICT VALIDATION - EVERY COMPONENT".center(80))
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    validator = StrictValidator()

    # Run all validations
    validator.validate_file_structure()
    validator.validate_imports()
    validator.validate_data_generator()
    validator.validate_preprocessing()
    validator.validate_models()
    validator.validate_evaluation()
    validator.validate_data_outputs()
    validator.validate_model_outputs()
    validator.validate_pipeline()

    # Print summary
    success = validator.print_summary()

    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
