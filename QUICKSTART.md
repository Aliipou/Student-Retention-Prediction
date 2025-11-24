# Quick Start Guide

Get up and running with the Student Retention Prediction System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space

## Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Clone or navigate to the project directory
cd student-retention

# Run the automated setup script
python run.py
```

The script will:
1. Check dependencies
2. Generate synthetic data
3. Train ML models
4. Run tests
5. Launch the dashboard

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python src/data_generator.py

# 3. Train models
python src/train_pipeline.py

# 4. Launch dashboard
streamlit run src/dashboard.py
```

## First Steps

### 1. Explore the Dashboard

Once launched, the dashboard opens at `http://localhost:8501`

Navigate through 5 pages:
- **Overview**: System metrics and risk distribution
- **Student Search**: Individual student analysis
- **Risk Analysis**: Demographic risk factors
- **Model Insights**: Performance metrics and SHAP
- **Bulk Predictions**: Export filtered students

### 2. Analyze Individual Students

1. Go to "Student Search" page
2. Select a student ID from dropdown
3. View their complete profile and risk assessment
4. Compare with peer averages

### 3. Identify High-Risk Students

1. Go to "Overview" page
2. See the "High-Risk Students" table
3. These students need immediate intervention
4. Note their risk factors (low GPA, failed courses, etc.)

### 4. Export Data for Interventions

1. Go to "Bulk Predictions" page
2. Filter by risk level (High Risk)
3. Select specific majors if needed
4. Download CSV with all high-risk students
5. Share with advisors for outreach

## Common Tasks

### Regenerate Data with Different Size

```bash
python src/data_generator.py
# Edit the file to change n_samples parameter

# Or use the pipeline
python src/train_pipeline.py --generate-data --n-samples 50000
```

### Train with Hyperparameter Tuning

```bash
python src/train_pipeline.py --tune-hyperparameters
```

This takes longer but improves model performance.

### Train Specific Models

```bash
# Train only Random Forest and XGBoost
python src/train_pipeline.py --models random_forest,xgboost
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Explore with Jupyter

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook includes:
- Data exploration with visualizations
- Feature correlation analysis
- Model training walkthrough
- SHAP interpretability examples

## Docker Deployment

### Using Docker Compose (Easiest)

```bash
# Build and run
docker-compose up

# Dashboard: http://localhost:8501
# Jupyter: http://localhost:8888
```

### Using Dockerfile

```bash
# Build image
docker build -t student-retention .

# Run container
docker run -p 8501:8501 student-retention
```

## Troubleshooting

### Dashboard shows "Model not found"

**Solution**: Train models first
```bash
python src/train_pipeline.py
```

### Import errors

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Tests failing

**Solution**: Ensure working directory is project root
```bash
cd student-retention
pytest tests/ -v
```

### SHAP plots taking too long

**Solution**: Reduce sample size in evaluation.py
```python
# In evaluation.py, line ~389
shap_values = evaluator.compute_shap_values(sample_size=500)  # Reduce from 1000
```

### Dashboard is slow

**Solution**:
1. Use fewer samples for SHAP analysis
2. Filter data to recent records only
3. Cache data loading is already enabled

## Next Steps

### 1. Understand Your Data

Explore the Jupyter notebook to understand:
- What features drive dropout risk
- Patterns in your student population
- Model decision-making process

### 2. Customize for Your Institution

Modify `data_generator.py` to reflect:
- Your institution's actual features
- Realistic dropout rates
- Specific risk factors

### 3. Integrate with Real Data

Replace synthetic data with real student data:
1. Export your student database
2. Map columns to expected format
3. Run preprocessing pipeline
4. Retrain models on real data

### 4. Deploy to Production

Options:
- **Streamlit Cloud**: Free hosting for dashboard
- **AWS/GCP**: Scalable cloud deployment
- **On-Premise**: Docker container on local server

### 5. Track Interventions

Extend the system to:
- Record advisor outreach
- Track intervention effectiveness
- Measure retention improvements
- A/B test different strategies

## Performance Tips

### For Large Datasets (>100k students)

1. Use Parquet format instead of CSV
2. Reduce SHAP sample size to 500
3. Use LightGBM (fastest training)
4. Enable pagination in dashboard

### For Real-Time Predictions

1. Load model once at startup
2. Use REST API wrapper (FastAPI)
3. Cache preprocessor transformations
4. Batch predictions when possible

## Support

### Documentation
- Full README: [README.md](README.md)
- Development plan: [development-plan.md](development-plan.md)

### Getting Help
- Check the README FAQ section
- Review code comments and docstrings
- Run tests to verify functionality
- Open GitHub issues for bugs

## Key Metrics to Monitor

1. **ROC-AUC Score**: Should be >0.85 for good performance
2. **Precision**: Minimize false positives (low-risk marked as high-risk)
3. **Recall**: Minimize false negatives (high-risk marked as low-risk)
4. **Calibration**: Predicted probabilities match actual rates

## Success Criteria

A successful deployment should:
- ✅ Identify 80%+ of at-risk students
- ✅ Maintain <20% false positive rate
- ✅ Provide actionable insights via SHAP
- ✅ Update predictions weekly/monthly
- ✅ Track intervention effectiveness

---

**You're ready to go! Launch the dashboard and start exploring:**

```bash
streamlit run src/dashboard.py
```

Happy predicting! 🎓
