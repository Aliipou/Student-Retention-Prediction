"""
Interactive Dashboard for Student Retention Prediction
Streamlit-based dashboard for visualizing predictions and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataPreprocessor
from models import StudentRetentionModel

# Page configuration
st.set_page_config(
    page_title="Student Retention Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load student data."""
    data_path = 'data/student_data.csv'
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.info("Please run: python src/data_generator.py")
        return None
    return pd.read_csv(data_path)


@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        model_path = 'models/best_model.joblib'
        preprocessor_path = 'models/preprocessor.joblib'

        if not os.path.exists(model_path):
            return None, None, "Model not found. Please train the model first."

        model_state = joblib.load(model_path)
        model = StudentRetentionModel()
        model.model = model_state['model']
        model.model_type = model_state['model_type']

        preprocessor = DataPreprocessor()
        preprocessor.load(preprocessor_path)

        return model, preprocessor, None

    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def get_risk_category(probability):
    """Categorize risk level based on probability."""
    if probability < 0.3:
        return "Low Risk", "low-risk"
    elif probability < 0.6:
        return "Medium Risk", "medium-risk"
    else:
        return "High Risk", "high-risk"


def plot_risk_distribution(df, risk_col='risk_probability'):
    """Plot distribution of risk scores."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[risk_col],
        nbinsx=50,
        name='Risk Distribution',
        marker_color='steelblue',
        opacity=0.7
    ))

    fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                  annotation_text="Low Risk Threshold")
    fig.add_vline(x=0.6, line_dash="dash", line_color="orange",
                  annotation_text="High Risk Threshold")

    fig.update_layout(
        title="Distribution of Dropout Risk Scores",
        xaxis_title="Dropout Risk Probability",
        yaxis_title="Number of Students",
        showlegend=False,
        height=400
    )

    return fig


def plot_feature_impact(df, feature, risk_col='risk_probability'):
    """Plot relationship between feature and risk."""
    fig = px.scatter(
        df, x=feature, y=risk_col,
        color=risk_col,
        color_continuous_scale='RdYlGn_r',
        title=f'Impact of {feature} on Dropout Risk',
        labels={risk_col: 'Dropout Risk', feature: feature.replace('_', ' ').title()}
    )

    fig.update_layout(height=400)
    return fig


def plot_risk_by_category(df, category_col, risk_col='risk_probability'):
    """Plot average risk by categorical variable."""
    avg_risk = df.groupby(category_col)[risk_col].mean().sort_values(ascending=False)

    fig = go.Figure(go.Bar(
        x=avg_risk.values,
        y=avg_risk.index,
        orientation='h',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title=f'Average Dropout Risk by {category_col.replace("_", " ").title()}',
        xaxis_title='Average Dropout Risk',
        yaxis_title=category_col.replace('_', ' ').title(),
        height=400
    )

    return fig


def show_student_details(student_row, risk_probability):
    """Display detailed information about a student."""
    risk_category, risk_class = get_risk_category(risk_probability)

    st.markdown(f"### Student ID: {student_row['student_id']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<h4>Risk Level: <span class="{risk_class}">{risk_category}</span></h4>',
                   unsafe_allow_html=True)
        st.markdown(f'<h3 class="{risk_class}">{risk_probability:.1%}</h3>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("GPA", f"{student_row['gpa']:.2f}")
        st.metric("Attendance Rate", f"{student_row['attendance_rate']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Failed Courses", int(student_row['failed_courses']))
        st.metric("Credits Last Semester", int(student_row['credits_last_sem']))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Complete Student Profile")
    profile_df = pd.DataFrame({
        'Attribute': [
            'Major', 'Age', 'Gender', 'GPA', 'GPA Variance',
            'Attendance Rate', 'Failed Courses', 'Credits Last Semester',
            'Moodle Activity Score', 'Library Visits', 'Logins Last Week',
            'Scholarship Amount', 'Part-time Job', 'Distance from Campus (km)'
        ],
        'Value': [
            student_row['major'],
            int(student_row['age']),
            student_row['gender'],
            f"{student_row['gpa']:.2f}",
            f"{student_row['gpa_variance']:.2f}",
            f"{student_row['attendance_rate']:.1%}",
            int(student_row['failed_courses']),
            int(student_row['credits_last_sem']),
            f"{student_row['moodle_activity_score']:.1f}",
            int(student_row['library_visits']),
            int(student_row['login_times_last_week']),
            f"${student_row['scholarship_amount']:.0f}",
            'Yes' if student_row['part_time_job'] == 1 else 'No',
            f"{student_row['distance_from_campus_km']:.1f}"
        ]
    })

    st.dataframe(profile_df, use_container_width=True, hide_index=True)


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">🎓 Student Retention Prediction System</h1>',
               unsafe_allow_html=True)

    # Load data and model
    df = load_data()
    if df is None:
        return

    model, preprocessor, error = load_model_and_preprocessor()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Student Search", "Risk Analysis", "Model Insights", "Bulk Predictions"]
    )

    # Make predictions if model is loaded
    predictions_available = False
    if model is not None and preprocessor is not None:
        with st.spinner("Generating predictions..."):
            try:
                df_for_prediction = df.drop('student_id', axis=1) if 'student_id' in df.columns else df
                X, _ = preprocessor.prepare_features(df_for_prediction, fit=False)
                risk_proba = model.predict_proba(X)[:, 1]
                df['risk_probability'] = risk_proba
                df['risk_category'] = df['risk_probability'].apply(lambda x: get_risk_category(x)[0])
                predictions_available = True
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    else:
        st.warning("⚠️ Model not loaded. Please train the model first by running the training pipeline.")
        st.info("Run: `python src/train_pipeline.py`")

    # Page: Overview
    if page == "Overview":
        st.header("📊 System Overview")

        if predictions_available:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            total_students = len(df)
            high_risk = (df['risk_probability'] >= 0.6).sum()
            medium_risk = ((df['risk_probability'] >= 0.3) & (df['risk_probability'] < 0.6)).sum()
            low_risk = (df['risk_probability'] < 0.3).sum()

            col1.metric("Total Students", f"{total_students:,}")
            col2.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/total_students:.1%}")
            col3.metric("Medium Risk", f"{medium_risk:,}", delta=f"{medium_risk/total_students:.1%}")
            col4.metric("Low Risk", f"{low_risk:,}", delta=f"{low_risk/total_students:.1%}")

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_risk_distribution(df), use_container_width=True)

            with col2:
                risk_counts = df['risk_category'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker_colors=['#2ca02c', '#ff7f0e', '#d62728']
                )])
                fig.update_layout(title="Students by Risk Category", height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Recent high-risk students
            st.subheader("🚨 High-Risk Students Requiring Attention")
            high_risk_df = df[df['risk_probability'] >= 0.6].sort_values(
                'risk_probability', ascending=False
            ).head(10)

            display_cols = ['student_id', 'major', 'gpa', 'failed_courses',
                          'attendance_rate', 'risk_probability', 'risk_category']
            st.dataframe(
                high_risk_df[display_cols].style.format({
                    'gpa': '{:.2f}',
                    'attendance_rate': '{:.1%}',
                    'risk_probability': '{:.1%}'
                }),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Train the model to see predictions and insights.")

    # Page: Student Search
    elif page == "Student Search":
        st.header("🔍 Individual Student Analysis")

        if predictions_available:
            student_id = st.selectbox(
                "Select Student ID",
                options=df['student_id'].tolist()
            )

            if student_id:
                student_row = df[df['student_id'] == student_id].iloc[0]
                risk_prob = student_row['risk_probability']

                show_student_details(student_row, risk_prob)

                # Comparison with cohort
                st.subheader("📈 Comparison with Peer Group")

                peer_group = df[df['major'] == student_row['major']]

                metrics_to_compare = ['gpa', 'attendance_rate', 'moodle_activity_score',
                                     'library_visits', 'risk_probability']

                comparison_data = []
                for metric in metrics_to_compare:
                    comparison_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Student Value': student_row[metric],
                        'Peer Average': peer_group[metric].mean(),
                        'All Students Average': df[metric].mean()
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.format({
                    'Student Value': '{:.2f}',
                    'Peer Average': '{:.2f}',
                    'All Students Average': '{:.2f}'
                }), use_container_width=True, hide_index=True)

        else:
            st.info("Train the model to analyze individual students.")

    # Page: Risk Analysis
    elif page == "Risk Analysis":
        st.header("📉 Risk Factor Analysis")

        if predictions_available:
            # Filters
            st.sidebar.subheader("Filters")
            selected_majors = st.sidebar.multiselect(
                "Filter by Major",
                options=df['major'].unique(),
                default=df['major'].unique()
            )

            gpa_range = st.sidebar.slider(
                "GPA Range",
                float(df['gpa'].min()),
                float(df['gpa'].max()),
                (float(df['gpa'].min()), float(df['gpa'].max()))
            )

            # Apply filters
            filtered_df = df[
                (df['major'].isin(selected_majors)) &
                (df['gpa'] >= gpa_range[0]) &
                (df['gpa'] <= gpa_range[1])
            ]

            st.info(f"Showing {len(filtered_df):,} students after filtering")

            # Risk by major
            st.subheader("Risk Distribution by Major")
            st.plotly_chart(plot_risk_by_category(filtered_df, 'major'), use_container_width=True)

            # Feature impacts
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_feature_impact(filtered_df, 'gpa'), use_container_width=True)

            with col2:
                st.plotly_chart(plot_feature_impact(filtered_df, 'attendance_rate'),
                              use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_feature_impact(filtered_df, 'failed_courses'),
                              use_container_width=True)

            with col2:
                st.plotly_chart(plot_feature_impact(filtered_df, 'moodle_activity_score'),
                              use_container_width=True)

        else:
            st.info("Train the model to see risk analysis.")

    # Page: Model Insights
    elif page == "Model Insights":
        st.header("🤖 Model Performance & Insights")

        if model is not None:
            st.subheader("Model Information")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"**Model Type:** {model.model_type.replace('_', ' ').title()}")

            with col2:
                if model.best_params:
                    st.info(f"**Hyperparameters Tuned:** Yes")

            # Display evaluation plots if available
            st.subheader("Model Evaluation Metrics")

            plot_files = [
                ('confusion_matrix.png', 'Confusion Matrix'),
                ('roc_curve.png', 'ROC Curve'),
                ('precision_recall_curve.png', 'Precision-Recall Curve'),
                ('calibration_curve.png', 'Calibration Curve'),
                ('feature_importance.png', 'Feature Importance'),
                ('shap_summary.png', 'SHAP Summary'),
            ]

            for plot_file, title in plot_files:
                plot_path = f'assets/{plot_file}'
                if os.path.exists(plot_path):
                    st.subheader(title)
                    image = Image.open(plot_path)
                    st.image(image, use_column_width=True)

            # Metrics table
            metrics_path = 'assets/model_comparison_metrics.csv'
            if os.path.exists(metrics_path):
                st.subheader("Model Metrics")
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        else:
            st.info("Train the model to see performance metrics.")

    # Page: Bulk Predictions
    elif page == "Bulk Predictions":
        st.header("📋 Bulk Predictions & Export")

        if predictions_available:
            st.subheader("Filter & Export Students")

            # Risk level filter
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=['Low Risk', 'Medium Risk', 'High Risk'],
                default=['High Risk']
            )

            # Major filter
            major_filter = st.multiselect(
                "Filter by Major",
                options=df['major'].unique(),
                default=df['major'].unique()
            )

            # Apply filters
            filtered_df = df[
                (df['risk_category'].isin(risk_filter)) &
                (df['major'].isin(major_filter))
            ].sort_values('risk_probability', ascending=False)

            st.info(f"Found {len(filtered_df):,} students matching criteria")

            # Display table
            display_cols = ['student_id', 'major', 'age', 'gender', 'gpa',
                          'failed_courses', 'attendance_rate',
                          'risk_probability', 'risk_category']

            st.dataframe(
                filtered_df[display_cols].style.format({
                    'gpa': '{:.2f}',
                    'attendance_rate': '{:.1%}',
                    'risk_probability': '{:.1%}'
                }),
                use_container_width=True,
                hide_index=True
            )

            # Export functionality
            csv = filtered_df[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="student_predictions.csv",
                mime="text/csv"
            )

        else:
            st.info("Train the model to generate bulk predictions.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides insights into student dropout risk using machine learning. "
        "Early identification enables targeted interventions to improve retention rates."
    )


if __name__ == "__main__":
    main()
