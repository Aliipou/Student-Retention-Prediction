"""
Data Generator for Student Retention Prediction
Generates realistic fake student data with probabilistic relationships to dropout risk.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import os


class StudentDataGenerator:
    """Generate synthetic student data with realistic patterns."""

    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        """
        Initialize data generator.

        Args:
            n_samples: Number of student records to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)

    def generate(self) -> pd.DataFrame:
        """
        Generate complete student dataset with realistic patterns.

        Returns:
            DataFrame with student features and dropout risk labels
        """
        # Reset random seed for reproducibility
        np.random.seed(self.random_state)

        # Generate basic features with realistic distributions
        data = {}

        # Academic performance features
        data['student_id'] = [f'STU{str(i).zfill(6)}' for i in range(self.n_samples)]
        data['credits_last_sem'] = np.random.randint(9, 21, self.n_samples)
        data['failed_courses'] = np.random.choice([0, 1, 2, 3, 4, 5],
                                                   self.n_samples,
                                                   p=[0.5, 0.25, 0.15, 0.06, 0.03, 0.01])

        # Engagement features
        data['moodle_activity_score'] = np.random.beta(5, 2, self.n_samples) * 100
        data['library_visits'] = np.random.poisson(8, self.n_samples)
        data['login_times_last_week'] = np.random.poisson(15, self.n_samples)
        data['attendance_rate'] = np.clip(np.random.beta(8, 2, self.n_samples), 0, 1)

        # GPA related
        base_gpa = np.random.beta(7, 3, self.n_samples) * 4.0
        data['gpa'] = np.clip(base_gpa, 0, 4.0)
        data['gpa_variance'] = np.abs(np.random.normal(0.3, 0.2, self.n_samples))

        # Demographics
        data['age'] = np.random.choice(range(18, 30), self.n_samples,
                                       p=[0.3, 0.25, 0.2, 0.1, 0.06, 0.04, 0.02, 0.01, 0.01, 0.005, 0.005, 0.0])
        data['gender'] = np.random.choice(['Male', 'Female', 'Other'],
                                          self.n_samples,
                                          p=[0.48, 0.50, 0.02])

        majors = ['Computer Science', 'Engineering', 'Business', 'Arts',
                  'Science', 'Medicine', 'Education', 'Social Sciences']
        data['major'] = np.random.choice(majors, self.n_samples)

        # Financial indicators
        data['scholarship_amount'] = np.random.choice([0, 1000, 2000, 3000, 5000, 10000],
                                                      self.n_samples,
                                                      p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        data['part_time_job'] = np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4])

        # Distance from campus
        data['distance_from_campus_km'] = np.abs(np.random.gamma(2, 5, self.n_samples))

        df = pd.DataFrame(data)

        # Generate dropout risk based on features (realistic relationships)
        dropout_probability = self._calculate_dropout_probability(df)
        df['dropout_risk'] = (np.random.random(self.n_samples) < dropout_probability).astype(int)

        return df

    def _calculate_dropout_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate dropout probability based on realistic feature relationships.

        Args:
            df: DataFrame with student features

        Returns:
            Array of dropout probabilities
        """
        # Base probability
        prob = np.full(len(df), 0.15)

        # Failed courses impact (strong predictor)
        prob += df['failed_courses'] * 0.12

        # GPA impact
        prob += (4.0 - df['gpa']) * 0.08

        # Attendance impact
        prob += (1.0 - df['attendance_rate']) * 0.15

        # Engagement impact
        prob += (100 - df['moodle_activity_score']) / 100 * 0.10
        prob += np.where(df['library_visits'] < 3, 0.08, 0)
        prob += np.where(df['login_times_last_week'] < 5, 0.10, 0)

        # GPA variance (instability)
        prob += df['gpa_variance'] * 0.10

        # Credits (too few or too many)
        prob += np.where(df['credits_last_sem'] < 12, 0.08, 0)
        prob += np.where(df['credits_last_sem'] > 18, 0.05, 0)

        # Financial stress
        prob += np.where(df['scholarship_amount'] == 0, 0.05, -0.02)
        prob += df['part_time_job'] * 0.03

        # Distance impact
        prob += np.where(df['distance_from_campus_km'] > 30, 0.05, 0)

        # Age impact (non-traditional students)
        prob += np.where(df['age'] > 24, 0.06, 0)

        # Add some random noise
        prob += np.random.normal(0, 0.05, len(df))

        # Clip to valid probability range
        return np.clip(prob, 0, 0.95)

    def save(self, df: pd.DataFrame, output_path: str = 'data/student_data.csv'):
        """
        Save generated data to file.

        Args:
            df: DataFrame to save
            output_path: Path to save file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Dropout rate: {df['dropout_risk'].mean():.2%}")


def main():
    """Generate and save student data."""
    generator = StudentDataGenerator(n_samples=20000, random_state=42)
    df = generator.generate()

    # Display basic statistics
    print("\n=== Dataset Generated ===")
    print(f"Total students: {len(df)}")
    print(f"Dropout rate: {df['dropout_risk'].mean():.2%}")
    print(f"\nFeatures: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())

    # Save data
    generator.save(df)

    # Also save as parquet for better performance
    df.to_parquet('data/student_data.parquet', index=False)
    print("Data also saved as parquet format")


if __name__ == "__main__":
    main()
