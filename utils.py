"""
Utility functions for data processing and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def validate_student_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate uploaded student data format"""
    required_columns = [
        'student_id', 'name', 'gpa', 'attendance_rate', 
        'assignment_completion', 'quiz_scores', 'participation_score'
    ]
    
    errors = []
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and ranges
    if 'gpa' in df.columns:
        if not df['gpa'].between(0, 4).all():
            errors.append("GPA values must be between 0 and 4")
    
    if 'attendance_rate' in df.columns:
        if not df['attendance_rate'].between(0, 1).all():
            errors.append("Attendance rate must be between 0 and 1")
    
    if 'assignment_completion' in df.columns:
        if not df['assignment_completion'].between(0, 1).all():
            errors.append("Assignment completion must be between 0 and 1")
    
    if 'quiz_scores' in df.columns:
        if not df['quiz_scores'].between(0, 100).all():
            errors.append("Quiz scores must be between 0 and 100")
    
    return len(errors) == 0, errors

def preprocess_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess student data for model input"""
    df_processed = df.copy()
    
    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Create additional features if not present
    if 'lms_activity' not in df_processed.columns:
        df_processed['lms_activity'] = np.random.poisson(15, len(df_processed))
    
    if 'late_submissions' not in df_processed.columns:
        df_processed['late_submissions'] = np.random.poisson(2, len(df_processed))
    
    if 'office_hours_visits' not in df_processed.columns:
        df_processed['office_hours_visits'] = np.random.poisson(3, len(df_processed))
    
    if 'study_group_participation' not in df_processed.columns:
        df_processed['study_group_participation'] = np.random.beta(3, 7, len(df_processed))
    
    if 'previous_semester_gpa' not in df_processed.columns:
        df_processed['previous_semester_gpa'] = df_processed['gpa'] + np.random.normal(0, 0.2, len(df_processed))
        df_processed['previous_semester_gpa'] = df_processed['previous_semester_gpa'].clip(0, 4)
    
    return df_processed

def calculate_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional risk indicators"""
    df_indicators = df.copy()
    
    # Academic performance indicators
    df_indicators['gpa_trend'] = df_indicators['gpa'] - df_indicators['previous_semester_gpa']
    df_indicators['performance_score'] = (
        df_indicators['gpa'] * 0.3 +
        df_indicators['attendance_rate'] * 0.2 +
        df_indicators['assignment_completion'] * 0.2 +
        df_indicators['quiz_scores'] / 100 * 0.2 +
        df_indicators['participation_score'] / 100 * 0.1
    )
    
    # Engagement indicators
    df_indicators['engagement_score'] = (
        df_indicators['lms_activity'] / 20 * 0.4 +
        df_indicators['office_hours_visits'] / 10 * 0.3 +
        df_indicators['study_group_participation'] * 0.3
    )
    
    # Risk flags
    df_indicators['gpa_risk'] = df_indicators['gpa'] < 2.5
    df_indicators['attendance_risk'] = df_indicators['attendance_rate'] < 0.7
    df_indicators['assignment_risk'] = df_indicators['assignment_completion'] < 0.6
    df_indicators['engagement_risk'] = df_indicators['engagement_score'] < 0.3
    
    return df_indicators

def generate_intervention_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate metrics for intervention planning"""
    total_students = len(df)
    high_risk = len(df[df.get('risk_category', '') == 'High Risk'])
    medium_risk = len(df[df.get('risk_category', '') == 'Medium Risk'])
    low_risk = len(df[df.get('risk_category', '') == 'Low Risk'])
    
    # Calculate intervention needs
    intervention_needs = {
        'tutoring': len(df[df['gpa'] < 2.5]),
        'attendance_support': len(df[df['attendance_rate'] < 0.7]),
        'assignment_help': len(df[df['assignment_completion'] < 0.6]),
        'engagement_boost': len(df[df.get('engagement_score', 0) < 0.3])
    }
    
    return {
        'total_students': total_students,
        'risk_distribution': {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk
        },
        'intervention_needs': intervention_needs,
        'risk_percentage': {
            'high_risk_pct': (high_risk / total_students) * 100,
            'medium_risk_pct': (medium_risk / total_students) * 100,
            'low_risk_pct': (low_risk / total_students) * 100
        }
    }

def create_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a risk factor heatmap"""
    risk_factors = ['gpa', 'attendance_rate', 'assignment_completion', 'quiz_scores', 'participation_score']
    
    # Create risk matrix
    risk_matrix = []
    for _, student in df.iterrows():
        student_risks = []
        for factor in risk_factors:
            if factor in student:
                if factor == 'gpa':
                    risk_level = 1 if student[factor] < 2.5 else (0.5 if student[factor] < 3.0 else 0)
                elif factor in ['attendance_rate', 'assignment_completion']:
                    risk_level = 1 if student[factor] < 0.7 else (0.5 if student[factor] < 0.8 else 0)
                elif factor in ['quiz_scores', 'participation_score']:
                    risk_level = 1 if student[factor] < 60 else (0.5 if student[factor] < 75 else 0)
                else:
                    risk_level = 0
                student_risks.append(risk_level)
            else:
                student_risks.append(0)
        risk_matrix.append(student_risks)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix[:50],  # Show first 50 students
        x=risk_factors,
        y=[f"Student {i+1}" for i in range(min(50, len(risk_matrix)))],
        colorscale='Reds',
        showscale=True
    ))
    
    fig.update_layout(
        title="Student Risk Factor Heatmap",
        xaxis_title="Risk Factors",
        yaxis_title="Students"
    )
    
    return fig

def create_performance_comparison(df: pd.DataFrame) -> go.Figure:
    """Create performance comparison chart"""
    risk_categories = df.get('risk_category', 'Unknown').unique()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPA by Risk Category', 'Attendance by Risk Category', 
                       'Assignment Completion by Risk Category', 'Quiz Scores by Risk Category'),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "box"}, {"type": "box"}]]
    )
    
    for i, category in enumerate(risk_categories):
        category_data = df[df.get('risk_category', 'Unknown') == category]
        
        # GPA
        fig.add_trace(
            go.Box(y=category_data['gpa'], name=f'{category} GPA', showlegend=False),
            row=1, col=1
        )
        
        # Attendance
        fig.add_trace(
            go.Box(y=category_data['attendance_rate'], name=f'{category} Attendance', showlegend=False),
            row=1, col=2
        )
        
        # Assignment Completion
        fig.add_trace(
            go.Box(y=category_data['assignment_completion'], name=f'{category} Assignments', showlegend=False),
            row=2, col=1
        )
        
        # Quiz Scores
        fig.add_trace(
            go.Box(y=category_data['quiz_scores'], name=f'{category} Quiz', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Performance Comparison by Risk Category")
    return fig

def export_student_report(df: pd.DataFrame, student_name: str) -> Dict[str, Any]:
    """Export detailed student report"""
    student_data = df[df['name'] == student_name].iloc[0]
    
    report = {
        'student_info': {
            'name': student_data['name'],
            'student_id': student_data.get('student_id', 'N/A'),
            'gpa': student_data['gpa'],
            'attendance_rate': student_data['attendance_rate'],
            'assignment_completion': student_data['assignment_completion']
        },
        'risk_assessment': {
            'risk_category': student_data.get('risk_category', 'Unknown'),
            'risk_probability': student_data.get('risk_probability', 0)
        },
        'performance_metrics': {
            'quiz_scores': student_data.get('quiz_scores', 0),
            'participation_score': student_data.get('participation_score', 0),
            'lms_activity': student_data.get('lms_activity', 0),
            'late_submissions': student_data.get('late_submissions', 0)
        },
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report

def create_intervention_timeline(intervention_plan: Dict[str, Any]) -> go.Figure:
    """Create intervention timeline visualization"""
    timeline_data = []
    
    for phase, actions in intervention_plan.get('timeline', {}).items():
        for action in actions:
            timeline_data.append({
                'Phase': phase,
                'Action': action,
                'Start': 0 if phase == 'immediate' else (7 if phase == 'week_1' else 30),
                'Duration': 1 if phase == 'immediate' else (7 if phase == 'week_1' else 30)
            })
    
    if not timeline_data:
        return go.Figure()
    
    df_timeline = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        df_timeline,
        x_start="Start",
        x_end="Duration",
        y="Phase",
        color="Phase",
        title="Intervention Timeline"
    )
    
    return fig


