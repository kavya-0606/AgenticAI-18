"""
Machine Learning Model Training for Student Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class StudentRiskPredictor:
    """Student risk prediction model using ensemble methods"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare and preprocess the data for training"""
        
        # Define feature columns (excluding target and student identifiers)
        feature_columns = [
            'gpa', 'attendance_rate', 'assignment_completion', 'quiz_scores',
            'participation_score', 'lms_activity', 'late_submissions',
            'office_hours_visits', 'study_group_participation', 'previous_semester_gpa'
        ]
        
        # Create target variable based on risk indicators
        # High risk: GPA < 2.5 OR attendance < 70% OR assignment completion < 60%
        df['risk_score'] = np.where(
            (df['gpa'] < 2.5) | (df['attendance_rate'] < 0.7) | (df['assignment_completion'] < 0.6),
            2,  # High risk
            np.where(
                (df['gpa'] < 3.0) | (df['attendance_rate'] < 0.8) | (df['assignment_completion'] < 0.75),
                1,  # Medium risk
                0   # Low risk
            )
        )
        
        # Handle missing values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        y = df['risk_score'].values
        
        self.feature_names = available_features
        
        return X, y, available_features
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model"""
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.feature_importance))
        }
    
    def predict_risk(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict risk scores and probabilities"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def get_risk_category(self, risk_score: int) -> str:
        """Convert numeric risk score to category"""
        risk_categories = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        return risk_categories.get(risk_score, 'Unknown')
    
    def get_explanation(self, X: np.ndarray, student_idx: int = 0) -> Dict[str, Any]:
        """Get feature importance explanation for a specific student"""
        if self.feature_importance is None:
            return {}
        
        # Get feature values for the student
        student_features = X[student_idx]
        feature_values = dict(zip(self.feature_names, student_features))
        
        # Get top contributing features
        importance_scores = dict(zip(self.feature_names, self.feature_importance))
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_values': feature_values,
            'top_features': sorted_features[:5],
            'risk_factors': [f for f, score in sorted_features[:3] if score > 0.1]
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        instance = cls(model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']
        return instance

def generate_sample_data(n_students: int = 1000) -> pd.DataFrame:
    """Generate sample student data for demonstration"""
    np.random.seed(42)
    
    data = {
        'student_id': [f'STU_{i:04d}' for i in range(1, n_students + 1)],
        'name': [f'Student {i}' for i in range(1, n_students + 1)],
        'email': [f'student{i}@university.edu' for i in range(1, n_students + 1)],
        'department': np.random.choice(['Computer Science', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Engineering'], n_students),
        'branch': np.random.choice(['Computer Science', 'Information Technology', 'Electronics', 'Mechanical', 'Civil', 'Electrical', 'Chemical', 'Aerospace'], n_students),
        'year': np.random.choice(['1st Year', '2nd Year', '3rd Year', '4th Year'], n_students),
        'section': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_students),
        'semester': np.random.choice(['Fall 2024', 'Spring 2024', 'Summer 2024'], n_students),
        'gpa': np.random.normal(3.2, 0.8, n_students).clip(0, 4.0),
        'attendance_rate': np.random.beta(8, 2, n_students),
        'assignment_completion': np.random.beta(7, 3, n_students),
        'quiz_scores': np.random.normal(75, 15, n_students).clip(0, 100),
        'participation_score': np.random.normal(70, 20, n_students).clip(0, 100),
        'lms_activity': np.random.poisson(15, n_students),
        'late_submissions': np.random.poisson(2, n_students),
        'office_hours_visits': np.random.poisson(3, n_students),
        'study_group_participation': np.random.beta(3, 7, n_students),
        'previous_semester_gpa': np.random.normal(3.1, 0.9, n_students).clip(0, 4.0),
        
        # Subject-specific attendance and performance
        'math_attendance': np.random.beta(8, 2, n_students),
        'math_performance': np.random.normal(75, 15, n_students).clip(0, 100),
        'physics_attendance': np.random.beta(7, 3, n_students),
        'physics_performance': np.random.normal(72, 18, n_students).clip(0, 100),
        'chemistry_attendance': np.random.beta(7, 3, n_students),
        'chemistry_performance': np.random.normal(74, 16, n_students).clip(0, 100),
        'english_attendance': np.random.beta(8, 2, n_students),
        'english_performance': np.random.normal(78, 12, n_students).clip(0, 100),
        'programming_attendance': np.random.beta(6, 4, n_students),
        'programming_performance': np.random.normal(70, 20, n_students).clip(0, 100)
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data and train model
    print("Generating sample data...")
    df = generate_sample_data(1000)
    
    print("Training model...")
    predictor = StudentRiskPredictor('xgboost')
    X, y, features = predictor.prepare_data(df)
    
    results = predictor.train(X, y)
    
    print(f"Model Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/student_risk_model.pkl')
    print("Model saved to models/student_risk_model.pkl")
