"""
Quick setup and run script for Student Retention Prediction System.
"""

import os
import sys
import subprocess
import argparse


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def check_dependencies():
    """Check if required packages are installed."""
    print_header("CHECKING DEPENDENCIES")

    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm',
        'shap', 'matplotlib', 'seaborn', 'plotly', 'streamlit', 'pytest'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed!")
    return True


def generate_data():
    """Generate synthetic student data."""
    print_header("GENERATING DATA")

    if os.path.exists('data/student_data.csv'):
        response = input("Data already exists. Regenerate? (y/N): ")
        if response.lower() != 'y':
            print("Using existing data.")
            return True

    try:
        from src.data_generator import StudentDataGenerator

        generator = StudentDataGenerator(n_samples=20000, random_state=42)
        df = generator.generate()
        generator.save(df)
        df.to_parquet('data/student_data.parquet', index=False)

        print("✓ Data generation complete!")
        return True
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return False


def train_models():
    """Train machine learning models."""
    print_header("TRAINING MODELS")

    try:
        subprocess.run([sys.executable, 'src/train_pipeline.py'], check=True)
        print("✓ Model training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error training models: {e}")
        return False


def run_tests():
    """Run unit tests."""
    print_header("RUNNING TESTS")

    try:
        result = subprocess.run(['pytest', 'tests/', '-v'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            return False
        print("✓ All tests passed!")
        return True
    except FileNotFoundError:
        print("✗ pytest not found. Install with: pip install pytest")
        return False


def launch_dashboard():
    """Launch Streamlit dashboard."""
    print_header("LAUNCHING DASHBOARD")

    print("Starting Streamlit dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(['streamlit', 'run', 'src/dashboard.py'])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except FileNotFoundError:
        print("✗ streamlit not found. Install with: pip install streamlit")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Student Retention Prediction - Quick Setup')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--dashboard-only', action='store_true', help='Only launch dashboard')

    args = parser.parse_args()

    print_header("STUDENT RETENTION PREDICTION SYSTEM")
    print("Quick Setup and Launch Script\n")

    if args.dashboard_only:
        launch_dashboard()
        return

    # Step 1: Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("\n⚠️  Please install missing dependencies first.")
            return

    # Step 2: Generate data
    if not args.skip_data:
        if not generate_data():
            print("\n⚠️  Data generation failed. Cannot continue.")
            return

    # Step 3: Train models
    if not args.skip_train:
        if not train_models():
            print("\n⚠️  Model training failed. You can still view data in dashboard.")

    # Step 4: Run tests
    if not args.skip_tests:
        run_tests()

    # Step 5: Launch dashboard
    print_header("SETUP COMPLETE")
    print("All components are ready!")
    print("\nWhat would you like to do?")
    print("  1. Launch dashboard")
    print("  2. Run Jupyter notebook")
    print("  3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        launch_dashboard()
    elif choice == '2':
        print("\nStarting Jupyter notebook...")
        try:
            subprocess.run(['jupyter', 'notebook', 'notebooks/exploratory_analysis.ipynb'])
        except FileNotFoundError:
            print("✗ jupyter not found. Install with: pip install jupyter")
    else:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
