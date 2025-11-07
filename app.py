"""
Agentic AI Early Warning System - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
warnings.filterwarnings('ignore')

# Import our custom modules
from train_model import StudentRiskPredictor, generate_sample_data
from agent import StudentInterventionAgent

# Email helper functions
def validate_email(email):
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def send_email_notification(student_email, student_name, status, risk_category, subject_name=None, recommendations=None):
    """Send email notification to student about intervention status"""
    try:
        smtp_server = st.session_state.get('smtp_server', 'smtp.gmail.com')
        smtp_port = st.session_state.get('smtp_port', 587)
        sender_email = st.session_state.get('sender_email', '')
        sender_password = st.session_state.get('sender_password', '')
        
        debug_mode = st.session_state.get('email_debug_mode', False)
        if debug_mode:
            st.write(f"🔍 **Debug Info:**")
            st.write(f"- SMTP Server: {smtp_server}")
            st.write(f"- SMTP Port: {smtp_port}")
            st.write(f"- Sender Email: {sender_email}")
            st.write(f"- Student Email: {student_email}")
            st.write(f"- Status: {status}")
            st.write(f"- Subject: {subject_name}")
        
        if not sender_email or not sender_password:
            st.warning("Email configuration not set. Please configure email settings in the Settings page.")
            return False
        
        if not validate_email(student_email):
            st.warning(f"Invalid email address: {student_email}")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = student_email
        
        if subject_name:
            msg['Subject'] = f"Academic Support Update - {subject_name} - {status.title()} Intervention"
        else:
            msg['Subject'] = f"Academic Support Update - {status.title()} Intervention"
        
        if status == 'approved':
            if subject_name:
                body = f"""
Dear {student_name},

We are writing to inform you that your academic support intervention for **{subject_name}** has been APPROVED by your academic advisor.

Subject: {subject_name}
Risk Level: {risk_category}
Status: {status.upper()}

This means that we will be providing you with additional academic support specifically for {subject_name} to help you succeed in your studies.

"""
            else:
                body = f"""
Dear {student_name},

We are writing to inform you that your academic support intervention has been APPROVED by your academic advisor.

Risk Level: {risk_category}
Status: {status.upper()}

This means that we will be providing you with additional academic support to help you succeed in your studies.

"""
            
            if recommendations:
                body += "Your personalized support plan includes:\n"
                for i, rec in enumerate(recommendations[:5], 1):
                    body += f"{i}. {rec}\n"
                body += "\n"
            
            body += """
Please expect to hear from your academic advisor within the next 2-3 business days to discuss the next steps.

If you have any questions, please don't hesitate to contact your academic advisor or the student support office.

Best regards,
Academic Support Team
            """
        else:  # declined
            if subject_name:
                body = f"""
Dear {student_name},

We are writing to inform you that your academic support intervention for **{subject_name}** has been DECLINED at this time.

Subject: {subject_name}
Risk Level: {risk_category}
Status: {status.upper()}

This means that we believe you are currently on track in {subject_name} and do not require additional intervention at this moment.

However, we will continue to monitor your academic progress in {subject_name}, and if you feel you need additional support, please don't hesitate to reach out to your academic advisor.

Best regards,
Academic Support Team
                """
            else:
                body = f"""
Dear {student_name},

We are writing to inform you that your academic support intervention has been DECLINED at this time.

Risk Level: {risk_category}
Status: {status.upper()}

This means that we believe you are currently on track and do not require additional intervention at this moment.

However, we will continue to monitor your academic progress, and if you feel you need additional support, please don't hesitate to reach out to your academic advisor.

Best regards,
Academic Support Team
                """
        
        msg.attach(MIMEText(body, 'plain'))
        
        if debug_mode:
            st.write("🔍 **Attempting to connect to SMTP server...**")
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        
        if debug_mode:
            st.write("🔍 **Connected to SMTP server, starting TLS...**")
        
        server.starttls()
        
        if debug_mode:
            st.write("🔍 **TLS started, attempting login...**")
        
        server.login(sender_email, sender_password)
        
        if debug_mode:
            st.write("🔍 **Login successful, sending email...**")
        
        text = msg.as_string()
        server.sendmail(sender_email, student_email, text)
        server.quit()
        
        if debug_mode:
            st.write("🔍 **Email sent successfully!**")
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        st.error(f"❌ **SMTP Authentication Error:** {str(e)}")
        st.error("**Most likely causes:**")
        st.error("- For Gmail: You need an App Password (not your regular Gmail password)")
        st.error("- Enable 2-Factor Authentication first, then generate App Password")
        st.error("- Check if email address is correct")
        with st.expander("📧 How to Generate Gmail App Password"):
            st.markdown("""
            1. Go to https://myaccount.google.com/security
            2. Enable 2-Step Verification (if not already enabled)
            3. Click on "App passwords"
            4. Select "Mail" and "Other (Custom name)"
            5. Enter "Streamlit App" as the name
            6. Copy the 16-character password
            7. Use this password (not your Gmail password)
            """)
        return False
    except smtplib.SMTPConnectError as e:
        st.error(f"❌ **SMTP Connection Error:** {str(e)}")
        st.error("**Possible solutions:**")
        st.error("- Check internet connection")
        st.error("- Verify SMTP server: smtp.gmail.com for Gmail")
        st.error("- Verify port: 587 for TLS")
        st.error("- Check firewall/antivirus settings")
        return False
    except smtplib.SMTPRecipientsRefused as e:
        st.error(f"❌ **Recipient Email Rejected:** {str(e)}")
        st.error("**Check:**")
        st.error("- Email address spelling is correct")
        st.error("- Email address exists")
        st.error("- Recipient's email server is accepting emails")
        return False
    except smtplib.SMTPSenderRefused as e:
        st.error(f"❌ **Sender Email Rejected:** {str(e)}")
        st.error("**Check:**")
        st.error("- Sender email address is correct")
        st.error("- Email account has permission to send emails")
        return False
    except smtplib.SMTPDataError as e:
        st.error(f"❌ **Email Data Error:** {str(e)}")
        st.error("**Possible issues:**")
        st.error("- Email content is too large")
        st.error("- Email format is invalid")
        return False
    except smtplib.SMTPException as e:
        st.error(f"❌ **SMTP Error:** {str(e)}")
        st.error("**General SMTP issue - check all settings**")
        return False
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {str(e)}")
        st.error("**This is an unexpected error. Please check:**")
        st.error("- All email settings are filled correctly")
        st.error("- Internet connection is stable")
        st.error("- Try enabling debug mode for more details")
        return False

# Page configuration
st.set_page_config(
    page_title="AI Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_or_train_model():
    """Load existing model or train a new one"""
    model_path = 'models/student_risk_model.pkl'
    
    if os.path.exists(model_path):
        try:
            return StudentRiskPredictor.load_model(model_path)
        except:
            pass
    
    # Train new model if none exists
    st.info("Training new model with sample data...")
    df = generate_sample_data(1000)
    predictor = StudentRiskPredictor('xgboost')
    X, y, features = predictor.prepare_data(df)
    predictor.train(X, y)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    predictor.save_model(model_path)
    
    return predictor

@st.cache_data
def load_agent():
    """Load the intervention agent"""
    return StudentInterventionAgent()

def create_risk_distribution_chart(df):
    """Create risk distribution pie chart"""
    risk_counts = df['risk_category'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Student Risk Distribution",
        color_discrete_map={
            'High Risk': '#f44336',
            'Medium Risk': '#ff9800',
            'Low Risk': '#4caf50'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_performance_trends(df):
    """Create performance trends chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPA Distribution', 'Attendance Rate', 'Assignment Completion', 'Quiz Scores'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # GPA Distribution
    fig.add_trace(
        go.Histogram(x=df['gpa'], name='GPA', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Attendance Rate
    fig.add_trace(
        go.Histogram(x=df['attendance_rate'], name='Attendance', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Assignment Completion
    fig.add_trace(
        go.Histogram(x=df['assignment_completion'], name='Assignments', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Quiz Scores
    fig.add_trace(
        go.Histogram(x=df['quiz_scores'], name='Quiz Scores', marker_color='lightyellow'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Student Performance Trends")
    return fig

def create_risk_factors_chart(df):
    """Create risk factors analysis chart"""
    # Calculate risk factors for each student
    risk_factors = []
    for _, student in df.iterrows():
        factors = []
        if student['gpa'] < 2.5:
            factors.append('Low GPA')
        if student['attendance_rate'] < 0.7:
            factors.append('Poor Attendance')
        if student['assignment_completion'] < 0.6:
            factors.append('Incomplete Assignments')
        if student['quiz_scores'] < 60:
            factors.append('Low Quiz Performance')
        if student['participation_score'] < 50:
            factors.append('Low Participation')
        
        risk_factors.extend(factors)
    
    factor_counts = pd.Series(risk_factors).value_counts()
    
    fig = px.bar(
        x=factor_counts.values,
        y=factor_counts.index,
        orientation='h',
        title="Most Common Risk Factors",
        color=factor_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(xaxis_title="Number of Students", yaxis_title="Risk Factors")
    return fig


def display_student_details(student_data, predictions, probabilities, agent, model):
    """Display detailed student information and recommendations"""
    st.subheader(f"Student Details: {student_data['name']}")
    
    # Student basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Student ID", student_data['student_id'])
    with col2:
        st.metric("Email", student_data.get('email', 'N/A'))
    with col3:
        st.metric("Department", student_data.get('department', 'N/A'))
    with col4:
        st.metric("Branch/Subject", student_data.get('branch', 'N/A'))
    
    # Additional student info
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Year", student_data.get('year', 'N/A'))
    with col2:
        st.metric("Section", student_data.get('section', 'N/A'))
    with col3:
        st.metric("Semester", student_data.get('semester', 'N/A'))
    with col4:
        st.metric("Risk Level", f"{student_data.get('risk_probability', 0):.1f}%")
    with col5:
        risk_category = model.get_risk_category(predictions[0])
        st.metric("Risk Category", risk_category)
    
    st.divider()
    
    # Academic performance metrics
    st.subheader("📊 Academic Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GPA", f"{student_data['gpa']:.2f}")
        st.metric("Attendance Rate", f"{student_data['attendance_rate']:.1%}")
    
    with col2:
        st.metric("Assignment Completion", f"{student_data['assignment_completion']:.1%}")
        st.metric("Quiz Scores", f"{student_data['quiz_scores']:.1f}")
    
    with col3:
        st.metric("Participation Score", f"{student_data['participation_score']:.1f}")
        st.metric("LMS Activity", f"{student_data['lms_activity']:.0f}")
    
    # Subject-specific performance and approval status
    st.subheader("📚 Subject-Specific Performance & Intervention Status")
    
    subjects = ['math', 'physics', 'chemistry', 'english', 'programming']
    subject_names = ['Mathematics', 'Physics', 'Chemistry', 'English', 'Programming']
    
    # Initialize session state for subject feedback if not exists
    if 'subject_feedback' not in st.session_state:
        st.session_state['subject_feedback'] = {}
    
    # Create columns for each subject
    cols = st.columns(len(subjects))
    
    for i, (subject, subject_name) in enumerate(zip(subjects, subject_names)):
        with cols[i]:
            attendance_col = f"{subject}_attendance"
            performance_col = f"{subject}_performance"
            
            if attendance_col in student_data and performance_col in student_data:
                attendance = student_data[attendance_col]
                performance = student_data[performance_col]
                
                # Determine subject risk level
                if attendance < 0.6 or performance < 50:
                    risk_level = "🔴 High Risk"
                    risk_color = "error"
                elif attendance < 0.75 or performance < 65:
                    risk_level = "🟡 Medium Risk"
                    risk_color = "warning"
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = "success"
                
                # Display subject info with risk level
                if risk_color == "error":
                    st.error(f"**{subject_name}**")
                elif risk_color == "warning":
                    st.warning(f"**{subject_name}**")
                else:
                    st.success(f"**{subject_name}**")
                
                # Show performance metrics
                st.write(f"**Risk Level:** {risk_level}")
                st.write(f"**Attendance:** {attendance:.1%}")
                st.write(f"**Performance:** {performance:.0f}%")
                
                # Check for subject-specific approval status
                student_subject_key = f"{student_data['student_id']}_{subject}"
                
                if student_subject_key in st.session_state['subject_feedback']:
                    feedback = st.session_state['subject_feedback'][student_subject_key]
                    status = feedback['status']
                    timestamp = feedback['timestamp']
                    
                    # Display approval status
                    if status == 'approved':
                        st.success(f"✅ **APPROVED**")
                        st.info(f"Intervention approved on {timestamp.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Show intervention details
                        with st.expander("📋 Intervention Details"):
                            st.write(f"**Subject:** {subject_name}")
                            st.write(f"**Risk Category:** {feedback.get('risk_category', 'Unknown')}")
                            st.write(f"**Attendance at Decision:** {feedback.get('attendance', 0):.1%}")
                            st.write(f"**Performance at Decision:** {feedback.get('performance', 0):.0f}%")
                            st.write(f"**Decision Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Suggested interventions based on subject and risk level
                            st.write("**Suggested Interventions:**")
                            if subject == 'math':
                                st.write("- Extra math tutoring sessions")
                                st.write("- Math problem-solving workshops")
                                st.write("- Peer study groups for mathematics")
                            elif subject == 'physics':
                                st.write("- Physics lab assistance")
                                st.write("- Conceptual physics review sessions")
                                st.write("- Physics problem-solving practice")
                            elif subject == 'chemistry':
                                st.write("- Chemistry lab support")
                                st.write("- Chemical equation balancing practice")
                                st.write("- Chemistry concept reinforcement")
                            elif subject == 'english':
                                st.write("- Writing skills improvement sessions")
                                st.write("- Reading comprehension practice")
                                st.write("- Grammar and vocabulary building")
                            elif subject == 'programming':
                                st.write("- Coding practice sessions")
                                st.write("- Algorithm and logic building")
                                st.write("- Programming project assistance")
                        
                        # Option to change decision
                        st.write("**Change Decision:**")
                        if st.button(f"❌ Change to Decline", key=f"change_to_decline_{student_subject_key}",
                                   use_container_width=True, type="secondary"):
                            # Update to declined status
                            st.session_state['subject_feedback'][student_subject_key] = {
                                'status': 'declined',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student_data['name'],
                                'student_id': student_data['student_id'],
                                'subject': subject_name,
                                'risk_category': risk_level,
                                'attendance': attendance,
                                'performance': performance,
                                'previous_status': 'approved',
                                'changed_from': feedback['timestamp']
                            }
                            
                            # Send email notification about status change
                            student_email = student_data.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner(f"Sending status change email for {subject_name}..."):
                                    if send_email_notification(student_email, student_data['name'], 'declined', 
                                                             risk_level, subject_name):
                                        st.warning(f"❌ Changed to Declined for {subject_name} - Email sent!")
                                    else:
                                        st.warning(f"❌ Changed to Declined for {subject_name} - Email failed")
                            else:
                                st.warning(f"❌ Changed to Declined for {subject_name}")
                            
                            st.rerun()
                    
                    elif status == 'declined':
                        st.warning(f"❌ **DECLINED**")
                        st.info(f"Intervention declined on {timestamp.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Show decline details
                        with st.expander("📋 Decline Details"):
                            st.write(f"**Subject:** {subject_name}")
                            st.write(f"**Risk Category:** {feedback.get('risk_category', 'Unknown')}")
                            st.write(f"**Attendance at Decision:** {feedback.get('attendance', 0):.1%}")
                            st.write(f"**Performance at Decision:** {feedback.get('performance', 0):.0f}%")
                            st.write(f"**Decision Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write("**Reason:** Intervention deemed unnecessary at this time")
                            st.write("**Monitoring:** Continued observation recommended")
                            
                            # Show change history if exists
                            if feedback.get('previous_status'):
                                st.write(f"**Previous Status:** {feedback['previous_status'].title()}")
                                st.write(f"**Changed From:** {feedback.get('changed_from', 'Unknown').strftime('%Y-%m-%d %H:%M:%S') if hasattr(feedback.get('changed_from', ''), 'strftime') else 'Unknown'}")
                        
                        # Option to change decision
                        st.write("**Change Decision:**")
                        if st.button(f"✅ Change to Approve", key=f"change_to_approve_{student_subject_key}",
                                   use_container_width=True, type="primary"):
                            # Update to approved status
                            st.session_state['subject_feedback'][student_subject_key] = {
                                'status': 'approved',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student_data['name'],
                                'student_id': student_data['student_id'],
                                'subject': subject_name,
                                'risk_category': risk_level,
                                'attendance': attendance,
                                'performance': performance,
                                'previous_status': 'declined',
                                'changed_from': feedback['timestamp']
                            }
                            
                            # Send email notification about status change
                            student_email = student_data.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner(f"Sending status change email for {subject_name}..."):
                                    if send_email_notification(student_email, student_data['name'], 'approved', 
                                                             risk_level, subject_name):
                                        st.success(f"✅ Changed to Approved for {subject_name} - Email sent!")
                                    else:
                                        st.success(f"✅ Changed to Approved for {subject_name} - Email failed")
                            else:
                                st.success(f"✅ Changed to Approved for {subject_name}")
                            
                            st.rerun()
                else:
                    # No decision made yet
                    st.info("⏳ **PENDING**")
                    st.write("No intervention decision made yet")
                    
                    # Show recommendation based on risk level
                    if risk_color == "error":
                        st.write("**Recommendation:** Immediate intervention needed")
                    elif risk_color == "warning":
                        st.write("**Recommendation:** Consider intervention")
                    else:
                        st.write("**Recommendation:** Continue monitoring")
                    
                    # Add approve/decline buttons for pending subjects
                    st.write("**Make Decision:**")
                    col_approve, col_decline = st.columns(2)
                    
                    with col_approve:
                        if st.button(f"✅ Approve", key=f"approve_analysis_{student_subject_key}", 
                                   use_container_width=True, type="primary"):
                            # Store subject-specific feedback
                            st.session_state['subject_feedback'][student_subject_key] = {
                                'status': 'approved',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student_data['name'],
                                'student_id': student_data['student_id'],
                                'subject': subject_name,
                                'risk_category': risk_level,
                                'attendance': attendance,
                                'performance': performance
                            }
                            
                            # Send email notification
                            student_email = student_data.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner(f"Sending approval email for {subject_name}..."):
                                    if send_email_notification(student_email, student_data['name'], 'approved', 
                                                             risk_level, subject_name):
                                        st.success(f"✅ Approved {subject_name} intervention - Email sent!")
                                    else:
                                        st.success(f"✅ Approved {subject_name} intervention - Email failed")
                            else:
                                st.success(f"✅ Approved {subject_name} intervention")
                            
                            st.rerun()
                    
                    with col_decline:
                        if st.button(f"❌ Decline", key=f"decline_analysis_{student_subject_key}",
                                   use_container_width=True, type="secondary"):
                            # Store subject-specific feedback
                            st.session_state['subject_feedback'][student_subject_key] = {
                                'status': 'declined',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student_data['name'],
                                'student_id': student_data['student_id'],
                                'subject': subject_name,
                                'risk_category': risk_level,
                                'attendance': attendance,
                                'performance': performance
                            }
                            
                            # Send email notification
                            student_email = student_data.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner(f"Sending decline email for {subject_name}..."):
                                    if send_email_notification(student_email, student_data['name'], 'declined', 
                                                             risk_level, subject_name):
                                        st.warning(f"❌ Declined {subject_name} intervention - Email sent!")
                                    else:
                                        st.warning(f"❌ Declined {subject_name} intervention - Email failed")
                            else:
                                st.warning(f"❌ Declined {subject_name} intervention")
                            
                            st.rerun()
            else:
                # Subject data not available
                st.info(f"**{subject_name}**")
                st.write("Subject data not available")
    
    # Overall intervention status summary
    st.subheader("📋 Overall Intervention Status Summary")
    
    # Count subject-specific decisions
    student_id = student_data['student_id']
    approved_subjects = []
    declined_subjects = []
    pending_subjects = []
    
    for subject, subject_name in zip(subjects, subject_names):
        student_subject_key = f"{student_id}_{subject}"
        if student_subject_key in st.session_state['subject_feedback']:
            feedback = st.session_state['subject_feedback'][student_subject_key]
            if feedback['status'] == 'approved':
                approved_subjects.append(subject_name)
            else:
                declined_subjects.append(subject_name)
        else:
            # Check if subject has data and needs decision
            attendance_col = f"{subject}_attendance"
            performance_col = f"{subject}_performance"
            if attendance_col in student_data and performance_col in student_data:
                attendance = student_data[attendance_col]
                performance = student_data[performance_col]
                if attendance < 0.75 or performance < 65:  # Medium or High risk
                    pending_subjects.append(subject_name)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if approved_subjects:
            st.success(f"**✅ Approved Interventions ({len(approved_subjects)}):**")
            for subject in approved_subjects:
                st.write(f"• {subject}")
        else:
            st.info("**✅ Approved Interventions:** None")
    
    with col2:
        if declined_subjects:
            st.warning(f"**❌ Declined Interventions ({len(declined_subjects)}):**")
            for subject in declined_subjects:
                st.write(f"• {subject}")
        else:
            st.info("**❌ Declined Interventions:** None")
    
    with col3:
        if pending_subjects:
            st.error(f"**⏳ Pending Decisions ({len(pending_subjects)}):**")
            for subject in pending_subjects:
                st.write(f"• {subject}")
            st.write("*These subjects need intervention decisions*")
        else:
            st.info("**⏳ Pending Decisions:** None")
    
    # Overall student intervention status
    if 'student_feedback' not in st.session_state:
        st.session_state['student_feedback'] = {}
    
    if student_id in st.session_state['student_feedback']:
        overall_feedback = st.session_state['student_feedback'][student_id]
        st.subheader("🎯 Overall Student Intervention Status")
        
        status_emoji = "✅" if overall_feedback['status'] == 'approved' else "❌"
        status_color = "success" if overall_feedback['status'] == 'approved' else "warning"
        
        if status_color == "success":
            st.success(f"{status_emoji} **Overall Intervention: {overall_feedback['status'].upper()}**")
        else:
            st.warning(f"{status_emoji} **Overall Intervention: {overall_feedback['status'].upper()}**")
        
        st.info(f"**Decision Date:** {overall_feedback['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(f"**Risk Category:** {overall_feedback.get('risk_category', 'Unknown')}")
        
        # Option to change overall decision
        st.write("**Change Overall Decision:**")
        col1, col2 = st.columns(2)
        
        if overall_feedback['status'] == 'approved':
            with col1:
                if st.button("❌ Change to Decline Overall", key=f"change_overall_to_decline_{student_id}",
                           use_container_width=True, type="secondary"):
                    # Update overall status to declined
                    st.session_state['student_feedback'][student_id] = {
                        'status': 'declined',
                        'timestamp': pd.Timestamp.now(),
                        'student_name': student_data['name'],
                        'risk_category': overall_feedback.get('risk_category', 'Unknown'),
                        'previous_status': 'approved',
                        'changed_from': overall_feedback['timestamp']
                    }
                    
                    # Send email notification
                    student_email = student_data.get('email', '')
                    if student_email and validate_email(student_email):
                        with st.spinner("Sending overall status change email..."):
                            if send_email_notification(student_email, student_data['name'], 'declined', 
                                                     overall_feedback.get('risk_category', 'Unknown')):
                                st.warning("❌ Overall intervention changed to Declined - Email sent!")
                            else:
                                st.warning("❌ Overall intervention changed to Declined - Email failed")
                    else:
                        st.warning("❌ Overall intervention changed to Declined")
                    
                    st.rerun()
        else:
            with col1:
                if st.button("✅ Change to Approve Overall", key=f"change_overall_to_approve_{student_id}",
                           use_container_width=True, type="primary"):
                    # Update overall status to approved
                    st.session_state['student_feedback'][student_id] = {
                        'status': 'approved',
                        'timestamp': pd.Timestamp.now(),
                        'student_name': student_data['name'],
                        'risk_category': overall_feedback.get('risk_category', 'Unknown'),
                        'previous_status': 'declined',
                        'changed_from': overall_feedback['timestamp']
                    }
                    
                    # Send email notification
                    student_email = student_data.get('email', '')
                    if student_email and validate_email(student_email):
                        with st.spinner("Sending overall status change email..."):
                            if send_email_notification(student_email, student_data['name'], 'approved', 
                                                     overall_feedback.get('risk_category', 'Unknown')):
                                st.success("✅ Overall intervention changed to Approved - Email sent!")
                            else:
                                st.success("✅ Overall intervention changed to Approved - Email failed")
                    else:
                        st.success("✅ Overall intervention changed to Approved")
                    
                    st.rerun()
    else:
        st.subheader("🎯 Overall Student Intervention Status")
        st.info("⏳ **Overall Intervention:** No decision made yet")
        
        # Add buttons to make overall decision
        st.write("**Make Overall Decision:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve Overall", key=f"approve_overall_analysis_{student_id}",
                       use_container_width=True, type="primary"):
                # Store overall feedback
                risk_category = model.get_risk_category(predictions[0])
                st.session_state['student_feedback'][student_id] = {
                    'status': 'approved',
                    'timestamp': pd.Timestamp.now(),
                    'student_name': student_data['name'],
                    'risk_category': risk_category
                }
                
                # Send email notification
                student_email = student_data.get('email', '')
                if student_email and validate_email(student_email):
                    with st.spinner("Sending overall approval email..."):
                        if send_email_notification(student_email, student_data['name'], 'approved', risk_category):
                            st.success("✅ Overall intervention approved - Email sent!")
                        else:
                            st.success("✅ Overall intervention approved - Email failed")
                else:
                    st.success("✅ Overall intervention approved")
                
                st.rerun()
        
        with col2:
            if st.button("❌ Decline Overall", key=f"decline_overall_analysis_{student_id}",
                       use_container_width=True, type="secondary"):
                # Store overall feedback
                risk_category = model.get_risk_category(predictions[0])
                st.session_state['student_feedback'][student_id] = {
                    'status': 'declined',
                    'timestamp': pd.Timestamp.now(),
                    'student_name': student_data['name'],
                    'risk_category': risk_category
                }
                
                # Send email notification
                student_email = student_data.get('email', '')
                if student_email and validate_email(student_email):
                    with st.spinner("Sending overall decline email..."):
                        if send_email_notification(student_email, student_data['name'], 'declined', risk_category):
                            st.warning("❌ Overall intervention declined - Email sent!")
                        else:
                            st.warning("❌ Overall intervention declined - Email failed")
                else:
                    st.warning("❌ Overall intervention declined")
                
                st.rerun()
    
    # Risk prediction
    risk_score = predictions[0]
    risk_category = model.get_risk_category(risk_score)
    risk_probability = probabilities[0][risk_score] * 100
    
    st.subheader("Risk Assessment")
    
    if risk_category == "High Risk":
        st.markdown(f'<div class="risk-high"><h4>🔴 {risk_category}</h4><p>Risk Probability: {risk_probability:.1f}%</p></div>', unsafe_allow_html=True)
    elif risk_category == "Medium Risk":
        st.markdown(f'<div class="risk-medium"><h4>🟡 {risk_category}</h4><p>Risk Probability: {risk_probability:.1f}%</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low"><h4>🟢 {risk_category}</h4><p>Risk Probability: {risk_probability:.1f}%</p></div>', unsafe_allow_html=True)
    
    # AI-generated recommendations
    st.subheader("AI-Generated Intervention Recommendations")
    
    # Get agent analysis
    analysis = agent.analyze_student(student_data.to_dict(), risk_score)
    
    # Display recommendations
    for i, recommendation in enumerate(analysis['recommendations'][:5], 1):
        st.write(f"{i}. {recommendation}")
    
    # Display next steps
    st.subheader("Recommended Next Steps")
    for i, step in enumerate(analysis['next_steps'][:3], 1):
        st.write(f"{i}. {step}")
    
    # Feature importance explanation
    st.subheader("Risk Factor Analysis")
    explanation = model.get_explanation(np.array([list(student_data[model.feature_names])]), 0)
    
    if explanation:
        st.write("**Top Contributing Factors:**")
        for factor, importance in explanation['top_features'][:3]:
            st.write(f"• {factor}: {importance:.3f}")
    
    # Natural language summary
    st.subheader("AI Summary")
    summary = agent.generate_natural_language_summary(analysis, student_data['name'])
    st.markdown(summary)
    
    # Intervention Decision for all students
    st.divider()
    st.subheader("🎯 Intervention Decision")
    
    # Initialize session state for feedback if not exists
    if 'student_feedback' not in st.session_state:
        st.session_state['student_feedback'] = {}
    
    student_key = student_data['student_id']
    
    # Show risk level specific styling
    if risk_category == "High Risk":
        st.error("🚨 **HIGH RISK STUDENT** - Immediate intervention required!")
    elif risk_category == "Medium Risk":
        st.warning("⚠️ **MEDIUM RISK STUDENT** - Monitoring and support recommended")
    else:
        st.success("✅ **LOW RISK STUDENT** - Regular monitoring sufficient")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        button_text = "✅ Approve Intervention" if risk_category == "High Risk" else "✅ Approve Support"
        if st.button(button_text, key=f"approve_{student_key}_detail", type="primary", use_container_width=True):
            # Store feedback
            st.session_state['student_feedback'][student_key] = {
                'status': 'approved',
                'timestamp': pd.Timestamp.now(),
                'student_name': student_data['name'],
                'risk_category': risk_category
            }
            st.success(f"✅ Approved intervention for {student_data['name']}")
            st.rerun()
    
    with col2:
        button_text = "❌ Decline Intervention" if risk_category == "High Risk" else "❌ Decline Support"
        if st.button(button_text, key=f"decline_{student_key}_detail", type="secondary", use_container_width=True):
            # Store feedback
            st.session_state['student_feedback'][student_key] = {
                'status': 'declined',
                'timestamp': pd.Timestamp.now(),
                'student_name': student_data['name'],
                'risk_category': risk_category
            }
            st.warning(f"❌ Declined intervention for {student_data['name']}")
            st.rerun()
    
    with col3:
        # Show current feedback status
        if student_key in st.session_state['student_feedback']:
            feedback = st.session_state['student_feedback'][student_key]
            status_emoji = "✅" if feedback['status'] == 'approved' else "❌"
            st.info(f"{status_emoji} **Current Status:** {feedback['status'].title()} on {feedback['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No decision made yet")
            
    # Show intervention details
    st.subheader("📋 Intervention Details")
    st.write("**Recommended Actions:**")
    for i, recommendation in enumerate(analysis['recommendations'][:5], 1):
        st.write(f"{i}. {recommendation}")
    
    st.write("**Next Steps:**")
    for i, step in enumerate(analysis['next_steps'][:3], 1):
        st.write(f"{i}. {step}")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 AI Early Warning System</h1>', unsafe_allow_html=True)
    st.markdown("### Proactive Student Risk Assessment and Intervention Platform")
    
    # Load models
    with st.spinner("Loading AI models..."):
        model = load_or_train_model()
        agent = load_agent()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Data source indicator
    if 'uploaded_data' in st.session_state:
        st.sidebar.success(f"📊 **Live Data**\n{len(st.session_state['uploaded_data']):,} students loaded")
    else:
        st.sidebar.info("📊 **Sample Data**\nUpload data to see live results")
    
    st.sidebar.divider()
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Dashboard", "📁 Upload Data", "👤 Student Analysis", "🔍 Risk Factors", "📋 Approval Status", "⚙️ Settings"]
    )
    
    if page == "📊 Dashboard":
        st.header("📊 System Overview")
        
        # Data source indicator and refresh button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'uploaded_data' in st.session_state:
                df = st.session_state['uploaded_data']
                predictions = st.session_state['predictions']
                probabilities = st.session_state['probabilities']
                st.success(f"📊 **Live Data:** Displaying uploaded dataset ({len(df):,} students)")
            else:
                # Generate sample data for dashboard
                df = generate_sample_data(500)
                X, y, features = model.prepare_data(df)
                predictions, probabilities = model.predict_risk(X)
                
                # Add predictions to dataframe
                df['risk_score'] = predictions
                df['risk_category'] = [model.get_risk_category(score) for score in predictions]
                df['risk_probability'] = [prob[pred] * 100 for prob, pred in zip(probabilities, predictions)]
                st.info("📊 **Sample Data:** Upload your own data in the 'Upload Data' page to see your dataset here.")
        
        with col2:
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2b:
                if 'uploaded_data' in st.session_state:
                    if st.button("🗑️ Clear Data", use_container_width=True):
                        del st.session_state['uploaded_data']
                        del st.session_state['predictions']
                        del st.session_state['probabilities']
                        st.success("Data cleared! Dashboard will show sample data.")
                        st.rerun()
            with col2c:
                # Quick email test
                if st.button("📧 Test Email", use_container_width=True, help="Quick email test"):
                    if st.session_state.get('sender_email') and st.session_state.get('sender_password'):
                        test_email = "chaitutummala2@gmail.com"
                        with st.spinner("Sending test email..."):
                            if send_email_notification(test_email, "Dashboard Test", "approved", "High Risk", "Mathematics"):
                                st.success(f"✅ Test email sent to {test_email}!")
                            else:
                                st.error("❌ Email test failed - check Settings")
                    else:
                        st.warning("⚠️ Configure email in Settings first")
        
        # Multi-level filtering
        st.subheader("🎯 Advanced Filtering Options")
        
        # Get unique values for filtering
        available_branches = df['branch'].unique() if 'branch' in df.columns else ['All Branches']
        available_years = df['year'].unique() if 'year' in df.columns else ['All Years']
        available_sections = df['section'].unique() if 'section' in df.columns else ['All Sections']
        
        # First row of filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_branch = st.selectbox(
                "Select Branch/Subject:",
                ['All Branches'] + sorted(list(available_branches)),
                help="Filter dashboard data by specific branch/subject"
            )
        
        with col2:
            selected_year = st.selectbox(
                "Select Year:",
                ['All Years'] + sorted(list(available_years)),
                help="Filter dashboard data by academic year"
            )
        
        with col3:
            selected_section = st.selectbox(
                "Select Section:",
                ['All Sections'] + sorted(list(available_sections)),
                help="Filter dashboard data by section"
            )
        
        # Apply filters progressively
        filtered_df = df.copy()
        filter_info = []
        
        if selected_branch != 'All Branches':
            filtered_df = filtered_df[filtered_df['branch'] == selected_branch]
            filter_info.append(f"**Branch:** {selected_branch}")
        
        if selected_year != 'All Years':
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
            filter_info.append(f"**Year:** {selected_year}")
        
        if selected_section != 'All Sections':
            filtered_df = filtered_df[filtered_df['section'] == selected_section]
            filter_info.append(f"**Section:** {selected_section}")
        
        # Show filter summary and metrics
        if filter_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Filtered Students", len(filtered_df))
            with col2:
                high_risk_filtered = len(filtered_df[filtered_df['risk_category'] == 'High Risk'])
                st.metric("High Risk", high_risk_filtered)
            with col3:
                if len(filtered_df) > 0:
                    avg_gpa_filtered = filtered_df['gpa'].mean()
                    st.metric("Avg GPA", f"{avg_gpa_filtered:.2f}")
                else:
                    st.metric("Avg GPA", "N/A")
            with col4:
                if len(filtered_df) > 0:
                    avg_attendance_filtered = filtered_df['attendance_rate'].mean()
                    st.metric("Avg Attendance", f"{avg_attendance_filtered:.1%}")
                else:
                    st.metric("Avg Attendance", "N/A")
            
            st.info(f"📊 Showing data for: {' | '.join(filter_info)} ({len(filtered_df)} students)")
            df = filtered_df
        else:
            st.info("📊 Showing data for all students")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(df))
        
        with col2:
            high_risk = len(df[df['risk_category'] == 'High Risk'])
            st.metric("High Risk Students", high_risk, delta=f"{high_risk/len(df)*100:.1f}%")
        
        with col3:
            avg_gpa = df['gpa'].mean()
            st.metric("Average GPA", f"{avg_gpa:.2f}")
        
        with col4:
            avg_attendance = df['attendance_rate'].mean()
            st.metric("Average Attendance", f"{avg_attendance:.1%}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_risk_distribution_chart(df), config={'displayModeBar': False})
        
        with col2:
            st.plotly_chart(create_risk_factors_chart(df), config={'displayModeBar': False})
        
        # Performance trends
        st.plotly_chart(create_performance_trends(df), config={'displayModeBar': False})
        
        # Multi-dimensional risk analysis
        st.subheader("📊 Multi-Dimensional Risk Analysis")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Branch Analysis", "📅 Year Analysis", "📝 Section Analysis", "🔍 Combined Analysis"])
        
        with tab1:
            if 'branch' in df.columns and len(df) > 0:
                # Calculate risk statistics by branch
                branch_risk_stats = df.groupby('branch').agg({
                    'risk_category': 'count',
                    'gpa': 'mean',
                    'attendance_rate': 'mean',
                    'risk_probability': 'mean'
                }).round(2)
                
                # Add high-risk count by branch
                high_risk_by_branch = df[df['risk_category'] == 'High Risk'].groupby('branch').size()
                branch_risk_stats['high_risk_count'] = high_risk_by_branch
                branch_risk_stats['high_risk_count'] = branch_risk_stats['high_risk_count'].fillna(0).astype(int)
                branch_risk_stats['high_risk_percentage'] = (branch_risk_stats['high_risk_count'] / branch_risk_stats['risk_category'] * 100).round(1)
                
                # Rename columns for display
                branch_risk_stats.columns = ['Total Students', 'Avg GPA', 'Avg Attendance', 'Avg Risk %', 'High Risk Count', 'High Risk %']
                
                # Sort by high risk percentage
                branch_risk_stats = branch_risk_stats.sort_values('High Risk %', ascending=False)
                
                st.dataframe(branch_risk_stats, width='stretch')
                
                # Branch risk chart
                fig_branch = px.bar(
                    branch_risk_stats.reset_index(),
                    x='branch',
                    y='High Risk %',
                    title="High Risk Percentage by Branch/Subject",
                    color='High Risk %',
                    color_continuous_scale='Reds'
                )
                fig_branch.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_branch, config={'displayModeBar': False})
            else:
                st.info("No branch data available for analysis.")
        
        with tab2:
            if 'year' in df.columns and len(df) > 0:
                # Calculate risk statistics by year
                year_risk_stats = df.groupby('year').agg({
                    'risk_category': 'count',
                    'gpa': 'mean',
                    'attendance_rate': 'mean',
                    'risk_probability': 'mean'
                }).round(2)
                
                # Add high-risk count by year
                high_risk_by_year = df[df['risk_category'] == 'High Risk'].groupby('year').size()
                year_risk_stats['high_risk_count'] = high_risk_by_year
                year_risk_stats['high_risk_count'] = year_risk_stats['high_risk_count'].fillna(0).astype(int)
                year_risk_stats['high_risk_percentage'] = (year_risk_stats['high_risk_count'] / year_risk_stats['risk_category'] * 100).round(1)
                
                # Rename columns for display
                year_risk_stats.columns = ['Total Students', 'Avg GPA', 'Avg Attendance', 'Avg Risk %', 'High Risk Count', 'High Risk %']
                
                # Sort by year
                year_risk_stats = year_risk_stats.sort_index()
                
                st.dataframe(year_risk_stats, width='stretch')
                
                # Year risk chart
                fig_year = px.bar(
                    year_risk_stats.reset_index(),
                    x='year',
                    y='High Risk %',
                    title="High Risk Percentage by Academic Year",
                    color='High Risk %',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_year, config={'displayModeBar': False})
            else:
                st.info("No year data available for analysis.")
        
        with tab3:
            if 'section' in df.columns and len(df) > 0:
                # Calculate risk statistics by section
                section_risk_stats = df.groupby('section').agg({
                    'risk_category': 'count',
                    'gpa': 'mean',
                    'attendance_rate': 'mean',
                    'risk_probability': 'mean'
                }).round(2)
                
                # Add high-risk count by section
                high_risk_by_section = df[df['risk_category'] == 'High Risk'].groupby('section').size()
                section_risk_stats['high_risk_count'] = high_risk_by_section
                section_risk_stats['high_risk_count'] = section_risk_stats['high_risk_count'].fillna(0).astype(int)
                section_risk_stats['high_risk_percentage'] = (section_risk_stats['high_risk_count'] / section_risk_stats['risk_category'] * 100).round(1)
                
                # Rename columns for display
                section_risk_stats.columns = ['Total Students', 'Avg GPA', 'Avg Attendance', 'Avg Risk %', 'High Risk Count', 'High Risk %']
                
                # Sort by section
                section_risk_stats = section_risk_stats.sort_index()
                
                st.dataframe(section_risk_stats, width='stretch')
                
                # Section risk chart
                fig_section = px.bar(
                    section_risk_stats.reset_index(),
                    x='section',
                    y='High Risk %',
                    title="High Risk Percentage by Section",
                    color='High Risk %',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_section, config={'displayModeBar': False})
            else:
                st.info("No section data available for analysis.")
        
        with tab4:
            if 'branch' in df.columns and 'year' in df.columns and 'section' in df.columns and len(df) > 0:
                # Combined analysis: Branch + Year + Section
                combined_stats = df.groupby(['branch', 'year', 'section']).agg({
                    'risk_category': 'count',
                    'gpa': 'mean',
                    'risk_probability': 'mean'
                }).round(2)
                
                # Add high-risk count
                high_risk_combined = df[df['risk_category'] == 'High Risk'].groupby(['branch', 'year', 'section']).size()
                combined_stats['high_risk_count'] = high_risk_combined
                combined_stats['high_risk_count'] = combined_stats['high_risk_count'].fillna(0).astype(int)
                combined_stats['high_risk_percentage'] = (combined_stats['high_risk_count'] / combined_stats['risk_category'] * 100).round(1)
                
                # Rename columns for display
                combined_stats.columns = ['Total Students', 'Avg GPA', 'Avg Risk %', 'High Risk Count', 'High Risk %']
                
                # Sort by high risk percentage
                combined_stats = combined_stats.sort_values('High Risk %', ascending=False)
                
                st.dataframe(combined_stats, width='stretch')
                
                # Show top risk combinations
                if len(combined_stats) > 0:
                    st.subheader("🚨 Top Risk Combinations")
                    top_risk = combined_stats.head(5)
                    
                    for idx, (index, row) in enumerate(top_risk.iterrows(), 1):
                        branch, year, section = index
                        risk_pct = row['High Risk %']
                        total_students = row['Total Students']
                        
                        if risk_pct > 0:
                            st.warning(f"**#{idx}** {branch} - {year} - Section {section}: {risk_pct:.1f}% high risk ({total_students} students)")
            else:
                st.info("Combined analysis requires branch, year, and section data.")
        
        # High-risk students alerts with subject-specific analysis
        st.subheader("🚨 High-Risk Students Alert - Subject-Specific Analysis")
        high_risk_students = df[df['risk_category'] == 'High Risk'].sort_values('risk_probability', ascending=False)
        
        if not high_risk_students.empty:
            # Initialize session state for feedback if not exists
            if 'student_feedback' not in st.session_state:
                st.session_state['student_feedback'] = {}
            if 'subject_feedback' not in st.session_state:
                st.session_state['subject_feedback'] = {}
            
            # Show alerts for each high-risk student
            for idx, (_, student) in enumerate(high_risk_students.head(5).iterrows()):
                with st.container():
                    # Student header information
                    st.markdown(f"""
                    ### 🔴 **{student['name']}** (`{student['student_id']}`)
                    **Overall Risk Level:** {student['risk_probability']:.1f}% | **GPA:** {student['gpa']:.2f} | **Branch:** {student.get('branch', 'N/A')} | **Year:** {student.get('year', 'N/A')} | **Section:** {student.get('section', 'N/A')}
                    """)
                    
                    # Subject-specific analysis
                    st.write("**📚 Subject-Specific Risk Analysis:**")
                    
                    subjects = ['math', 'physics', 'chemistry', 'english', 'programming']
                    subject_names = ['Mathematics', 'Physics', 'Chemistry', 'English', 'Programming']
                    
                    # Create columns for subject analysis
                    cols = st.columns(len(subjects))
                    
                    for i, (subject, subject_name) in enumerate(zip(subjects, subject_names)):
                        with cols[i]:
                            attendance_col = f"{subject}_attendance"
                            performance_col = f"{subject}_performance"
                            
                            if attendance_col in student and performance_col in student:
                                attendance = student[attendance_col]
                                performance = student[performance_col]
                                
                                # Determine subject risk level
                                if attendance < 0.6 or performance < 50:
                                    risk_level = "🔴 High Risk"
                                    risk_color = "error"
                                elif attendance < 0.75 or performance < 65:
                                    risk_level = "🟡 Medium Risk"
                                    risk_color = "warning"
                                else:
                                    risk_level = "🟢 Low Risk"
                                    risk_color = "success"
                                
                                # Display subject info
                                if risk_color == "error":
                                    st.error(f"**{subject_name}**\n{risk_level}")
                                elif risk_color == "warning":
                                    st.warning(f"**{subject_name}**\n{risk_level}")
                                else:
                                    st.success(f"**{subject_name}**\n{risk_level}")
                                
                                st.write(f"Attendance: {attendance:.1%}")
                                st.write(f"Performance: {performance:.0f}%")
                                
                                # Individual subject approval buttons
                                student_subject_key = f"{student['student_id']}_{subject}"
                                
                                col_approve, col_decline = st.columns(2)
                                
                                with col_approve:
                                    if st.button("✅", key=f"approve_{student_subject_key}", 
                                               help=f"Approve intervention for {subject_name}"):
                                        # Store subject-specific feedback
                                        st.session_state['subject_feedback'][student_subject_key] = {
                                            'status': 'approved',
                                            'timestamp': pd.Timestamp.now(),
                                            'student_name': student['name'],
                                            'student_id': student['student_id'],
                                            'subject': subject_name,
                                            'risk_category': risk_level,
                                            'attendance': attendance,
                                            'performance': performance
                                        }
                                        
                                        # Send email notification
                                        student_email = student.get('email', '')
                                        if student_email and validate_email(student_email):
                                            with st.spinner(f"Sending email for {subject_name}..."):
                                                if send_email_notification(student_email, student['name'], 'approved', 
                                                                         risk_level, subject_name):
                                                    st.success(f"✅ Approved {subject_name} intervention - Email sent!")
                                                else:
                                                    st.success(f"✅ Approved {subject_name} intervention - Email failed")
                                        else:
                                            st.success(f"✅ Approved {subject_name} intervention")
                                        
                                        st.rerun()
                                
                                with col_decline:
                                    if st.button("❌", key=f"decline_{student_subject_key}",
                                               help=f"Decline intervention for {subject_name}"):
                                        # Store subject-specific feedback
                                        st.session_state['subject_feedback'][student_subject_key] = {
                                            'status': 'declined',
                                            'timestamp': pd.Timestamp.now(),
                                            'student_name': student['name'],
                                            'student_id': student['student_id'],
                                            'subject': subject_name,
                                            'risk_category': risk_level,
                                            'attendance': attendance,
                                            'performance': performance
                                        }
                                        
                                        # Send email notification
                                        student_email = student.get('email', '')
                                        if student_email and validate_email(student_email):
                                            with st.spinner(f"Sending email for {subject_name}..."):
                                                if send_email_notification(student_email, student['name'], 'declined', 
                                                                         risk_level, subject_name):
                                                    st.warning(f"❌ Declined {subject_name} intervention - Email sent!")
                                                else:
                                                    st.warning(f"❌ Declined {subject_name} intervention - Email failed")
                                        else:
                                            st.warning(f"❌ Declined {subject_name} intervention")
                                        
                                        st.rerun()
                                
                                # Show current status if exists
                                if student_subject_key in st.session_state['subject_feedback']:
                                    feedback = st.session_state['subject_feedback'][student_subject_key]
                                    status_emoji = "✅" if feedback['status'] == 'approved' else "❌"
                                    st.info(f"{status_emoji} {feedback['status'].title()}")
                    
                    # Overall student approval (for general intervention)
                    st.write("**🎯 Overall Student Intervention:**")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    student_key = student['student_id']
                    
                    with col1:
                        if st.button(f"✅ Approve Overall", key=f"approve_overall_{student_key}", type="primary"):
                            # Store overall feedback
                            st.session_state['student_feedback'][student_key] = {
                                'status': 'approved',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student['name'],
                                'risk_category': 'High Risk'
                            }
                            
                            # Send overall email notification
                            student_email = student.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner("Sending overall intervention email..."):
                                    if send_email_notification(student_email, student['name'], 'approved', 'High Risk'):
                                        st.success(f"✅ Approved overall intervention - Email sent!")
                                    else:
                                        st.success(f"✅ Approved overall intervention - Email failed")
                            else:
                                st.success(f"✅ Approved overall intervention")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button(f"❌ Decline Overall", key=f"decline_overall_{student_key}", type="secondary"):
                            # Store overall feedback
                            st.session_state['student_feedback'][student_key] = {
                                'status': 'declined',
                                'timestamp': pd.Timestamp.now(),
                                'student_name': student['name'],
                                'risk_category': 'High Risk'
                            }
                            
                            # Send overall email notification
                            student_email = student.get('email', '')
                            if student_email and validate_email(student_email):
                                with st.spinner("Sending overall intervention email..."):
                                    if send_email_notification(student_email, student['name'], 'declined', 'High Risk'):
                                        st.warning(f"❌ Declined overall intervention - Email sent!")
                                    else:
                                        st.warning(f"❌ Declined overall intervention - Email failed")
                            else:
                                st.warning(f"❌ Declined overall intervention")
                            
                            st.rerun()
                    
                    with col3:
                        # Show overall feedback status if exists
                        if student_key in st.session_state['student_feedback']:
                            feedback = st.session_state['student_feedback'][student_key]
                            status_emoji = "✅" if feedback['status'] == 'approved' else "❌"
                            st.info(f"{status_emoji} **Overall Status:** {feedback['status'].title()} on {feedback['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        else:
                            st.info("No overall decision made yet")
                    
                    st.divider()
            
            # Show summary of all high-risk students
            if len(high_risk_students) > 5:
                st.info(f"Showing top 5 of {len(high_risk_students)} high-risk students. Use 'Student Analysis' page for detailed view.")
            
            # High-risk subjects summary
            if 'branch' in high_risk_students.columns:
                st.subheader("⚠️ High-Risk Subjects Summary")
                
                high_risk_by_subject = high_risk_students.groupby('branch').agg({
                    'student_id': 'count',
                    'risk_probability': 'mean',
                    'gpa': 'mean'
                }).round(2)
                high_risk_by_subject.columns = ['High Risk Students', 'Avg Risk %', 'Avg GPA']
                high_risk_by_subject = high_risk_by_subject.sort_values('High Risk Students', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(high_risk_by_subject, width='stretch')
                
                with col2:
                    # Most at-risk subject
                    most_at_risk = high_risk_by_subject.index[0]
                    most_at_risk_count = high_risk_by_subject.iloc[0]['High Risk Students']
                    most_at_risk_avg_risk = high_risk_by_subject.iloc[0]['Avg Risk %']
                    
                    st.metric("Most At-Risk Subject", most_at_risk)
                    st.metric("High Risk Students", most_at_risk_count)
                    st.metric("Avg Risk Level", f"{most_at_risk_avg_risk:.1f}%")
                    
                    # Alert for critical subjects
                    if most_at_risk_avg_risk > 80:
                        st.error(f"🚨 **CRITICAL:** {most_at_risk} has extremely high risk levels!")
                    elif most_at_risk_avg_risk > 60:
                        st.warning(f"⚠️ **WARNING:** {most_at_risk} requires immediate attention!")
        else:
            st.success("No high-risk students identified!")
    
    elif page == "📁 Upload Data":
        st.header("📁 Upload Student Data")
        
        # Dataset size adjuster
        st.subheader("📊 Dataset Configuration")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dataset_size = st.slider(
                "Number of students to generate:",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Adjust the size of the generated sample dataset"
            )
        
        with col2:
            st.info(f"Current dataset size: **{dataset_size:,} students**")
        
        # Generate dataset button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🎲 Generate Sample Dataset", type="primary", use_container_width=True):
                with st.spinner(f"Generating {dataset_size} student records..."):
                    # Generate sample data
                    df = generate_sample_data(dataset_size)
                    
                    # Process data
                    X, y, features = model.prepare_data(df)
                    predictions, probabilities = model.predict_risk(X)
                    
                    # Add predictions to dataframe
                    df['risk_score'] = predictions
                    df['risk_category'] = [model.get_risk_category(score) for score in predictions]
                    df['risk_probability'] = [prob[pred] * 100 for prob, pred in zip(probabilities, predictions)]
                    
                    # Store in session state
                    st.session_state['uploaded_data'] = df
                    st.session_state['predictions'] = predictions
                    st.session_state['probabilities'] = probabilities
                    
                    st.success(f"✅ Generated {len(df)} student records successfully!")
                    
                    # Show AI-generated summary
                    st.subheader("🤖 AI-Generated Risk Analysis")
                    risk_counts = df['risk_category'].value_counts()
                    st.write("**AI-Generated Risk Distribution:**")
                    for category, count in risk_counts.items():
                        percentage = count/len(df)*100
                        if category == "High Risk":
                            st.write(f"🔴 **{category}:** {count} students ({percentage:.1f}%)")
                        elif category == "Medium Risk":
                            st.write(f"🟡 **{category}:** {count} students ({percentage:.1f}%)")
                        else:
                            st.write(f"🟢 **{category}:** {count} students ({percentage:.1f}%)")
                    
                    # Show how AI determined the risk categories
                    with st.expander("🔍 How AI Generated Risk Categories", expanded=True):
                        st.write("**The AI model automatically calculated risk levels for each student based on:**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**🔴 High Risk Criteria:**")
                            st.write("- GPA < 2.5")
                            st.write("- Attendance < 70%")
                            st.write("- Assignments < 60%")
                        
                        with col2:
                            st.write("**🟡 Medium Risk Criteria:**")
                            st.write("- GPA < 3.0")
                            st.write("- Attendance < 80%")
                            st.write("- Assignments < 75%")
                        
                        with col3:
                            st.write("**🟢 Low Risk Criteria:**")
                            st.write("- GPA ≥ 3.0")
                            st.write("- Attendance ≥ 80%")
                            st.write("- Assignments ≥ 75%")
                        
                        st.info("💡 **No pre-labeled risk data was needed!** The AI model learned patterns from student performance metrics and automatically assigned risk categories.")
                    
                    # Show data preview (original data without AI predictions)
                    with st.expander("📋 Preview Generated Data (Original Data)", expanded=True):
                        # Create a copy without AI-generated columns for preview
                        preview_df = df.drop(columns=['risk_score', 'risk_category', 'risk_probability'], errors='ignore')
                        st.dataframe(preview_df.head(10), width='stretch')
                        
                        st.info("💡 **Note:** The original dataset contains only performance metrics. Risk categories are automatically generated by AI when data is processed.")
                        
                        # Download button for original data (without AI predictions)
                        csv_original = preview_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Original Dataset (No Risk Categories)",
                            data=csv_original,
                            file_name=f"student_dataset_original_{dataset_size}_records.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Download button for processed data (with AI predictions)
                        csv_processed = df.to_csv(index=False)
                        st.download_button(
                            label="🤖 Download Processed Dataset (With AI Risk Predictions)",
                            data=csv_processed,
                            file_name=f"student_dataset_processed_{dataset_size}_records.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        # Current dataset status
        if 'uploaded_data' in st.session_state:
            st.divider()
            st.subheader("📈 Current Dataset Status")
            current_df = st.session_state['uploaded_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", f"{len(current_df):,}")
            with col2:
                high_risk = len(current_df[current_df['risk_category'] == 'High Risk'])
                st.metric("High Risk", f"{high_risk:,}")
            with col3:
                avg_gpa = current_df['gpa'].mean()
                st.metric("Avg GPA", f"{avg_gpa:.2f}")
            with col4:
                avg_attendance = current_df['attendance_rate'].mean()
                st.metric("Avg Attendance", f"{avg_attendance:.1%}")
        
        st.divider()
        
        # File upload section
        st.subheader("📤 Upload Your Own Data")
        st.info("""
        Upload a CSV file with student performance data. The file should include columns for:
        - student_id, name, gpa, attendance_rate, assignment_completion, quiz_scores,
        - participation_score, lms_activity, late_submissions, office_hours_visits,
        - study_group_participation, previous_semester_gpa
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded {len(df)} student records!")
                
                # Display sample data (original data without AI predictions)
                st.subheader("📋 Original Data Preview")
                st.dataframe(df.head(), width='stretch')
                st.info("💡 **Note:** This is your original data with only performance metrics. Risk categories will be automatically generated by AI when you process the data.")
                
                # Process data
                if st.button("Process Data and Generate Predictions"):
                    with st.spinner("Processing data..."):
                        X, y, features = model.prepare_data(df)
                        predictions, probabilities = model.predict_risk(X)
                        
                        # Add predictions to dataframe
                        df['risk_score'] = predictions
                        df['risk_category'] = [model.get_risk_category(score) for score in predictions]
                        df['risk_probability'] = [prob[pred] * 100 for prob, pred in zip(probabilities, predictions)]
                        
                        # Store in session state
                        st.session_state['uploaded_data'] = df
                        st.session_state['predictions'] = predictions
                        st.session_state['probabilities'] = probabilities
                        
                        st.success("✅ Data processed successfully! AI has automatically generated risk categories.")
                        
                        # Show AI-generated results
                        st.subheader("🤖 AI-Generated Risk Analysis")
                        risk_counts = df['risk_category'].value_counts()
                        st.write("**AI-Generated Risk Distribution:**")
                        for category, count in risk_counts.items():
                            percentage = count/len(df)*100
                            if category == "High Risk":
                                st.write(f"🔴 **{category}:** {count} students ({percentage:.1f}%)")
                            elif category == "Medium Risk":
                                st.write(f"🟡 **{category}:** {count} students ({percentage:.1f}%)")
                            else:
                                st.write(f"🟢 **{category}:** {count} students ({percentage:.1f}%)")
                        
                        # Show how AI determined the risk categories
                        with st.expander("🔍 How AI Generated Risk Categories", expanded=True):
                            st.write("**The AI model automatically calculated risk levels for each student based on:**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**🔴 High Risk Criteria:**")
                                st.write("- GPA < 2.5")
                                st.write("- Attendance < 70%")
                                st.write("- Assignments < 60%")
                            
                            with col2:
                                st.write("**🟡 Medium Risk Criteria:**")
                                st.write("- GPA < 3.0")
                                st.write("- Attendance < 80%")
                                st.write("- Assignments < 75%")
                            
                            with col3:
                                st.write("**🟢 Low Risk Criteria:**")
                                st.write("- GPA ≥ 3.0")
                                st.write("- Attendance ≥ 80%")
                                st.write("- Assignments ≥ 75%")
                            
                            st.info("💡 **No pre-labeled risk data was needed!** The AI model learned patterns from student performance metrics and automatically assigned risk categories.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif page == "👤 Student Analysis":
        st.header("👤 Individual Student Analysis")
        
        if 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            predictions = st.session_state['predictions']
            probabilities = st.session_state['probabilities']
            
            # Multi-level filtering for Student Analysis
            st.subheader("🎯 Advanced Student Filters")
            
            # Get unique values for filtering
            available_branches = ['All Branches'] + sorted(list(df['branch'].unique())) if 'branch' in df.columns else ['All Branches']
            available_years = ['All Years'] + sorted(list(df['year'].unique())) if 'year' in df.columns else ['All Years']
            available_sections = ['All Sections'] + sorted(list(df['section'].unique())) if 'section' in df.columns else ['All Sections']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_branch = st.selectbox(
                    "Filter by Branch/Subject:",
                    available_branches,
                    help="Filter students by specific branch/subject"
                )
            
            with col2:
                selected_year = st.selectbox(
                    "Filter by Year:",
                    available_years,
                    help="Filter students by academic year"
                )
            
            with col3:
                selected_section = st.selectbox(
                    "Filter by Section:",
                    available_sections,
                    help="Filter students by section"
                )
            
            # Apply filters progressively
            filtered_df = df.copy()
            filter_info = []
            
            if selected_branch != 'All Branches':
                filtered_df = filtered_df[filtered_df['branch'] == selected_branch]
                filter_info.append(f"**Branch:** {selected_branch}")
            
            if selected_year != 'All Years':
                filtered_df = filtered_df[filtered_df['year'] == selected_year]
                filter_info.append(f"**Year:** {selected_year}")
            
            if selected_section != 'All Sections':
                filtered_df = filtered_df[filtered_df['section'] == selected_section]
                filter_info.append(f"**Section:** {selected_section}")
            
            # Show filter summary
            if filter_info:
                st.info(f"📊 Showing students for: {' | '.join(filter_info)} ({len(filtered_df)} students)")
                df = filtered_df
            else:
                st.info(f"📊 Showing all students ({len(df)} students)")
            
            # Risk level filtering
            st.subheader("🔍 Risk Level Filter")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_filter = st.selectbox(
                    "Filter by Risk Level:",
                    ['All Risk Levels', 'High Risk', 'Medium Risk', 'Low Risk'],
                    help="Filter students by risk category"
                )
            
            with col2:
                if risk_filter != 'All Risk Levels':
                    risk_df = df[df['risk_category'] == risk_filter]
                    st.metric(f"{risk_filter} Students", len(risk_df))
                else:
                    st.metric("Total Students", len(df))
            
            with col3:
                if risk_filter != 'All Risk Levels':
                    avg_risk = risk_df['risk_probability'].mean() if len(risk_df) > 0 else 0
                    st.metric("Avg Risk %", f"{avg_risk:.1f}%")
            
            # Apply risk filter
            if risk_filter != 'All Risk Levels':
                df = df[df['risk_category'] == risk_filter]
                if len(df) == 0:
                    st.warning(f"No {risk_filter} students found in the selected branch.")
                    return
                st.info(f"📊 Showing **{risk_filter}** students from **{selected_branch}** branch ({len(df)} students)")
            
            # Student selector with enhanced information
            st.subheader("👤 Select Student for Analysis")
            
            # Create comprehensive student options with all available info
            student_options = []
            for _, row in df.iterrows():
                info_parts = [row['student_id'], row['name']]
                
                if 'branch' in row:
                    info_parts.append(f"({row['branch']})")
                if 'year' in row:
                    info_parts.append(f"{row['year']}")
                if 'section' in row:
                    info_parts.append(f"Sec-{row['section']}")
                
                info_parts.append(f"[{row['risk_category']}]")
                student_options.append(" - ".join(info_parts))
            
            selected_option = st.selectbox("Select a student to analyze", student_options)
            
            if selected_option:
                student_id = selected_option.split(' - ')[0]
                student_data = df[df['student_id'] == student_id].iloc[0]
                student_idx = df[df['student_id'] == student_id].index[0]
                
                # Branch comparison section
                if 'branch' in student_data and selected_branch != 'All Branches':
                    st.subheader("📊 Branch Comparison")
                    
                    branch_comparison = df.groupby('branch').agg({
                        'gpa': 'mean',
                        'attendance_rate': 'mean',
                        'risk_probability': 'mean',
                        'student_id': 'count'
                    }).round(2)
                    branch_comparison.columns = ['Avg GPA', 'Avg Attendance', 'Avg Risk %', 'Student Count']
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(branch_comparison, width='stretch')
                    
                    with col2:
                        student_branch = student_data['branch']
                        branch_stats = branch_comparison.loc[student_branch]
                        
                        st.metric("Student's Branch", student_branch)
                        st.metric("Branch Avg GPA", f"{branch_stats['Avg GPA']:.2f}")
                        st.metric("Branch Avg Risk", f"{branch_stats['Avg Risk %']:.1f}%")
                        
                        # Compare student to branch average
                        gpa_diff = student_data['gpa'] - branch_stats['Avg GPA']
                        risk_diff = student_data['risk_probability'] - branch_stats['Avg Risk %']
                        
                        if gpa_diff > 0:
                            st.success(f"📈 GPA: {gpa_diff:+.2f} above branch average")
                        else:
                            st.warning(f"📉 GPA: {gpa_diff:+.2f} below branch average")
                        
                        if risk_diff < 0:
                            st.success(f"📉 Risk: {risk_diff:+.1f}% below branch average")
                        else:
                            st.warning(f"📈 Risk: {risk_diff:+.1f}% above branch average")
                
                display_student_details(
                    student_data,
                    [predictions[student_idx]],
                    [probabilities[student_idx]],
                    agent,
                    model
                )
        else:
            st.warning("Please upload data first in the 'Upload Data' page.")
    
    elif page == "🔍 Risk Factors":
        st.header("🔍 Risk Factors Analysis")
        
        if 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            
            # Branch-wise filtering for risk factors
            if 'branch' in df.columns:
                st.subheader("🎯 Branch/Subject Filter")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    available_branches = ['All Branches'] + list(df['branch'].unique())
                    selected_branch = st.selectbox(
                        "Filter by Branch/Subject:",
                        available_branches,
                        key="risk_factors_branch_filter",
                        help="Filter risk analysis by specific branch/subject"
                    )
                
                with col2:
                    if selected_branch != 'All Branches':
                        branch_df = df[df['branch'] == selected_branch]
                        st.metric(f"{selected_branch} Students", len(branch_df))
                    else:
                        st.metric("Total Students", len(df))
                
                # Filter data based on selected branch
                if selected_branch != 'All Branches':
                    df = df[df['branch'] == selected_branch]
                    st.info(f"📊 Showing risk analysis for **{selected_branch}** branch ({len(df)} students)")
            
            # Feature importance
            st.subheader("Model Feature Importance")
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                importance_df = pd.DataFrame({
                    'Feature': model.feature_names,
                    'Importance': model.feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Risk Prediction"
                )
                st.plotly_chart(fig, config={'displayModeBar': False})
            
            # Risk factor correlations
            st.subheader("Risk Factor Correlations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, config={'displayModeBar': False})
            
            # Branch-specific risk factor analysis
            if 'branch' in df.columns and len(df) > 0:
                st.subheader("📊 Branch-Specific Risk Factor Analysis")
                
                # Risk factors by branch
                branch_risk_factors = {}
                for branch in df['branch'].unique():
                    branch_data = df[df['branch'] == branch]
                    risk_factors = []
                    
                    for _, student in branch_data.iterrows():
                        if student['gpa'] < 2.5:
                            risk_factors.append('Low GPA')
                        if student['attendance_rate'] < 0.7:
                            risk_factors.append('Poor Attendance')
                        if student['assignment_completion'] < 0.6:
                            risk_factors.append('Incomplete Assignments')
                        if student['quiz_scores'] < 60:
                            risk_factors.append('Low Quiz Performance')
                        if student['participation_score'] < 50:
                            risk_factors.append('Low Participation')
                    
                    # Count risk factors
                    from collections import Counter
                    risk_factor_counts = Counter(risk_factors)
                    branch_risk_factors[branch] = risk_factor_counts
                
                # Create risk factors comparison chart
                risk_factors_df = pd.DataFrame(branch_risk_factors).fillna(0)
                risk_factors_df = risk_factors_df.T  # Transpose for better visualization
                
                if not risk_factors_df.empty:
                    fig_risk_factors = px.bar(
                        risk_factors_df.reset_index(),
                        x='index',
                        y=risk_factors_df.columns,
                        title="Risk Factors by Branch/Subject",
                        barmode='group'
                    )
                    fig_risk_factors.update_layout(
                        xaxis_title="Branch/Subject",
                        yaxis_title="Number of Students",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_risk_factors, config={'displayModeBar': False})
                    
                    # Risk factors summary table
                    st.subheader("📋 Risk Factors Summary by Branch")
                    st.dataframe(risk_factors_df, width='stretch')
            
        else:
            st.warning("Please upload data first in the 'Upload Data' page.")
    
    elif page == "📋 Approval Status":
        st.header("📋 Student Intervention Approval Status")
        
        if 'student_feedback' not in st.session_state or not st.session_state['student_feedback']:
            st.info("No intervention decisions made yet. Visit the Dashboard or Student Analysis to review students.")
        else:
            feedback_data = st.session_state['student_feedback']
            
            # Get student data for additional information
            if 'uploaded_data' in st.session_state:
                df = st.session_state['uploaded_data']
            else:
                df = generate_sample_data(1000)
            
            # Create comprehensive student data with feedback
            student_data_dict = {row['student_id']: row.to_dict() for _, row in df.iterrows()}
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_feedback = len(feedback_data)
            approved_count = sum(1 for f in feedback_data.values() if f['status'] == 'approved')
            declined_count = sum(1 for f in feedback_data.values() if f['status'] == 'declined')
            
            with col1:
                st.metric("Total Decisions", total_feedback)
            with col2:
                st.metric("Approved", approved_count, delta=f"{approved_count/total_feedback*100:.1f}%" if total_feedback > 0 else "0%")
            with col3:
                st.metric("Declined", declined_count, delta=f"{declined_count/total_feedback*100:.1f}%" if total_feedback > 0 else "0%")
            with col4:
                approval_rate = approved_count / total_feedback * 100 if total_feedback > 0 else 0
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
            
            st.divider()
            
            # Branch-wise filtering
            st.subheader("🎯 Branch/Subject Filter")
            
            # Get all branches from students with feedback
            feedback_branches = set()
            for student_id, feedback in feedback_data.items():
                if student_id in student_data_dict:
                    branch = student_data_dict[student_id].get('branch', 'Unknown')
                    feedback_branches.add(branch)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_branch = st.selectbox(
                    "Filter by Branch/Subject:",
                    ['All Branches'] + sorted(list(feedback_branches)),
                    help="Filter approval status by specific branch/subject"
                )
            
            with col2:
                if selected_branch != 'All Branches':
                    branch_feedback = {k: v for k, v in feedback_data.items() 
                                     if k in student_data_dict and 
                                     student_data_dict[k].get('branch') == selected_branch}
                    st.metric(f"{selected_branch} Decisions", len(branch_feedback))
                else:
                    st.metric("Total Decisions", len(feedback_data))
            
            with col3:
                if selected_branch != 'All Branches':
                    branch_approved = sum(1 for f in branch_feedback.values() if f['status'] == 'approved')
                    branch_rate = branch_approved / len(branch_feedback) * 100 if branch_feedback else 0
                    st.metric("Branch Approval Rate", f"{branch_rate:.1f}%")
            
            # Filter data based on selected branch
            if selected_branch != 'All Branches':
                filtered_feedback = {k: v for k, v in feedback_data.items() 
                                   if k in student_data_dict and 
                                   student_data_dict[k].get('branch') == selected_branch}
                if not filtered_feedback:
                    st.warning(f"No decisions found for {selected_branch} branch.")
                    return
                st.info(f"📊 Showing decisions for **{selected_branch}** branch ({len(filtered_feedback)} decisions)")
            else:
                filtered_feedback = feedback_data
            
            st.divider()
            
            # Detailed approval status table
            st.subheader("📊 Detailed Approval Status")
            
            # Create comprehensive dataframe with student details
            detailed_data = []
            for student_id, feedback in filtered_feedback.items():
                if student_id in student_data_dict:
                    student_info = student_data_dict[student_id]
                    detailed_data.append({
                        'Student ID': student_id,
                        'Student Name': feedback['student_name'],
                        'Email': student_info.get('email', 'N/A'),
                        'Branch/Subject': student_info.get('branch', 'N/A'),
                        'Year': student_info.get('year', 'N/A'),
                        'Section': student_info.get('section', 'N/A'),
                        'Semester': student_info.get('semester', 'N/A'),
                        'GPA': f"{student_info.get('gpa', 0):.2f}",
                        'Risk Category': feedback.get('risk_category', 'Unknown'),
                        'Status': feedback['status'].title(),
                        'Decision Date': feedback['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'Days Since': (pd.Timestamp.now() - feedback['timestamp']).days
                    })
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                
                # Sort by decision date (most recent first)
                detailed_df = detailed_df.sort_values('Decision Date', ascending=False)
                
                # Display with styling
                st.dataframe(detailed_df, width='stretch')
                
                # Branch-wise summary
                if selected_branch == 'All Branches':
                    st.subheader("📊 Branch-wise Summary")
                    
                    branch_summary = detailed_df.groupby('Branch/Subject').agg({
                        'Student ID': 'count',
                        'Status': lambda x: (x == 'Approved').sum(),
                        'GPA': lambda x: pd.to_numeric(x).mean()
                    }).round(2)
                    branch_summary.columns = ['Total Decisions', 'Approved', 'Avg GPA']
                    branch_summary['Approval Rate'] = (branch_summary['Approved'] / branch_summary['Total Decisions'] * 100).round(1)
                    branch_summary = branch_summary.sort_values('Approval Rate', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(branch_summary, width='stretch')
                    
                    with col2:
                        # Branch approval chart
                        fig_branch = px.bar(
                            branch_summary.reset_index(),
                            x='Branch/Subject',
                            y='Approval Rate',
                            title="Approval Rate by Branch/Subject",
                            color='Approval Rate',
                            color_continuous_scale='RdYlGn'
                        )
                        fig_branch.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_branch, config={'displayModeBar': False})
                
                # Export data
                csv = detailed_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Approval Status Report",
                    data=csv,
                    file_name=f"approval_status_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Clear feedback button
                if st.button("🗑️ Clear All Decisions", type="secondary"):
                    st.session_state['student_feedback'] = {}
                    st.success("All decisions cleared!")
                    st.rerun()
            else:
                st.warning("No student data found for the selected criteria.")
    
    elif page == "⚙️ Settings":
        st.header("⚙️ System Settings")
        
        st.subheader("Model Information")
        st.write(f"Model Type: {model.model_type}")
        st.write(f"Features Used: {', '.join(model.feature_names)}")
        
        st.subheader("Intervention Agent")
        st.write("LangGraph-based agent for generating personalized intervention strategies")
        
        # API Key settings
        st.subheader("OpenAI API Configuration")
        api_key = st.text_input("OpenAI API Key (optional)", type="password", help="Enter your OpenAI API key for enhanced AI recommendations")
        
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("API key configured successfully!")
        
        # Email Configuration
        st.subheader("📧 Email Notification Settings")
        st.info("Configure email settings to send notifications to students when intervention decisions are made.")
        
        # Debug mode toggle
        debug_mode = st.checkbox(
            "🔍 Enable Email Debug Mode", 
            value=st.session_state.get('email_debug_mode', False),
            help="Enable detailed logging for email sending process"
        )
        st.session_state['email_debug_mode'] = debug_mode
        
        if debug_mode:
            st.warning("🔍 **Debug mode enabled** - You'll see detailed email sending information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smtp_server = st.text_input(
                "SMTP Server", 
                value=st.session_state.get('smtp_server', 'smtp.gmail.com'),
                help="SMTP server for sending emails (e.g., smtp.gmail.com for Gmail)"
            )
            smtp_port = st.number_input(
                "SMTP Port", 
                value=st.session_state.get('smtp_port', 587),
                min_value=1,
                max_value=65535,
                help="SMTP port (usually 587 for TLS, 465 for SSL)"
            )
            
            # Quick setup buttons for common providers
            st.write("**Quick Setup:**")
            col_gmail, col_outlook = st.columns(2)
            with col_gmail:
                if st.button("📧 Gmail Setup", use_container_width=True):
                    st.session_state['smtp_server'] = 'smtp.gmail.com'
                    st.session_state['smtp_port'] = 587
                    st.success("Gmail settings applied! Use App Password for authentication.")
            with col_outlook:
                if st.button("📧 Outlook Setup", use_container_width=True):
                    st.session_state['smtp_server'] = 'smtp-mail.outlook.com'
                    st.session_state['smtp_port'] = 587
                    st.success("Outlook settings applied!")
        
        with col2:
            sender_email = st.text_input(
                "Sender Email", 
                value=st.session_state.get('sender_email', ''),
                help="Email address to send notifications from"
            )
            sender_password = st.text_input(
                "Sender Password/App Password", 
                value=st.session_state.get('sender_password', ''),
                type="password",
                help="Password or app-specific password for the sender email"
            )
            
            # Gmail App Password instructions
            if smtp_server == 'smtp.gmail.com':
                with st.expander("📧 Gmail App Password Setup Instructions"):
                    st.markdown("""
                    **For Gmail, you need to use an App Password instead of your regular password:**
                    
                    1. **Enable 2-Factor Authentication** on your Google account
                    2. Go to **Google Account Settings** → **Security**
                    3. Under **2-Step Verification**, click **App passwords**
                    4. Select **Mail** and **Other (Custom name)**
                    5. Enter "Streamlit App" as the name
                    6. Copy the generated 16-character password
                    7. Use this App Password in the field above (not your regular Gmail password)
                    
                    **Important:** Never use your regular Gmail password for SMTP authentication!
                    """)
        
        # Save email configuration
        if st.button("💾 Save Email Configuration", type="primary"):
            st.session_state['smtp_server'] = smtp_server
            st.session_state['smtp_port'] = int(smtp_port)
            st.session_state['sender_email'] = sender_email
            st.session_state['sender_password'] = sender_password
            st.success("✅ Email configuration saved!")
        
        # Enhanced email testing
        st.subheader("🧪 Email Testing & Debugging")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Test email to self
            if st.button("📧 Test Email (Send to Self)", type="primary"):
                if sender_email and sender_password and validate_email(sender_email):
                    with st.spinner("Testing email configuration..."):
                        test_email = sender_email  # Send test email to self
                        st.write(f"🔍 **Sending test email to:** {test_email}")
                        
                        if send_email_notification(test_email, "Test User", "approved", "Test", "Test Subject", ["Test recommendation"]):
                            st.success("✅ Test email sent successfully!")
                            st.info("📧 **Check your inbox** (including spam folder) for the test email.")
                            st.write("**Email Subject:** Academic Support Update - Test Subject - Approved Intervention")
                        else:
                            st.error("❌ Failed to send test email. Please check your configuration.")
                else:
                    st.error("Please configure email settings first.")
        
        with col2:
            # Test email to custom address
            custom_test_email = st.text_input("Test Email Address", 
                                            value="chaitutummala2@gmail.com",
                                            help="Enter any email address to test")
            
            if st.button("📧 Test Email (Custom Address)", type="secondary"):
                if sender_email and sender_password and validate_email(sender_email):
                    if validate_email(custom_test_email):
                        with st.spinner(f"Sending test email to {custom_test_email}..."):
                            st.write(f"🔍 **Sending test email to:** {custom_test_email}")
                            
                            if send_email_notification(custom_test_email, "Test Student", "approved", "High Risk", "Mathematics", ["Extra tutoring sessions", "Weekly progress meetings"]):
                                st.success(f"✅ Test email sent successfully to {custom_test_email}!")
                                st.info("📧 **Ask the recipient to check their inbox** (including spam folder).")
                                st.write("**Email Subject:** Academic Support Update - Mathematics - Approved Intervention")
                            else:
                                st.error(f"❌ Failed to send test email to {custom_test_email}")
                    else:
                        st.error("❌ Invalid test email address format")
                else:
                    st.error("Please configure email settings first.")
        
        # Troubleshooting guide
        with st.expander("🔧 Email Troubleshooting Guide", expanded=False):
            st.markdown("""
            ### Common Email Issues and Solutions:
            
            #### 1. Gmail Configuration Issues
            - **Problem**: Authentication failed
            - **Solution**: 
              - Enable 2-Factor Authentication on your Google account
              - Generate an App Password (not your regular Gmail password)
              - Use the 16-character App Password in the settings above
            
            #### 2. SMTP Connection Issues
            - **Problem**: Connection timeout or refused
            - **Solution**:
              - Check your internet connection
              - Verify SMTP server: `smtp.gmail.com` for Gmail
              - Verify SMTP port: `587` for TLS
              - Check if firewall/antivirus is blocking the connection
            
            #### 3. Email Not Received
            - **Check spam/junk folder** - automated emails often go there
            - **Check email address spelling** - ensure it's correct
            - **Gmail delivery delay** - can take 1-5 minutes
              
            #### 4. For Gmail Users
            - Go to: https://myaccount.google.com/security
            - Enable 2-Step Verification
            - Generate App Password for "Mail"
            - Use the App Password (not your Gmail password)
            
            #### 5. Alternative Email Providers
            **Outlook/Hotmail:**
            - SMTP Server: `smtp-mail.outlook.com`
            - Port: `587`
            - Use your regular email and password
            
            **Yahoo:**
            - SMTP Server: `smtp.mail.yahoo.com`
            - Port: `587`
            - Generate App Password
            """)
        
        # Connection test
        st.subheader("🔌 SMTP Connection Test")
        if st.button("🧪 Test SMTP Connection Only"):
            if smtp_server and smtp_port and sender_email and sender_password:
                with st.spinner("Testing SMTP connection..."):
                    try:
                        st.write(f"🔍 **Testing connection to:** {smtp_server}:{smtp_port}")
                        
                        import socket
                        # Test basic connectivity
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(10)
                        result = sock.connect_ex((smtp_server, int(smtp_port)))
                        sock.close()
                        
                        if result == 0:
                            st.success(f"✅ Successfully connected to {smtp_server}:{smtp_port}")
                            
                            # Test SMTP authentication
                            try:
                                server = smtplib.SMTP(smtp_server, int(smtp_port))
                                server.starttls()
                                server.login(sender_email, sender_password)
                                server.quit()
                                st.success("✅ SMTP authentication successful!")
                            except smtplib.SMTPAuthenticationError:
                                st.error("❌ SMTP authentication failed - check email/password")
                            except Exception as e:
                                st.error(f"❌ SMTP error: {str(e)}")
                        else:
                            st.error(f"❌ Cannot connect to {smtp_server}:{smtp_port}")
                            st.write("**Possible issues:**")
                            st.write("- Check internet connection")
                            st.write("- Verify SMTP server address")
                            st.write("- Check if port is blocked by firewall")
                            
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {str(e)}")
            else:
                st.error("Please configure all email settings first.")
        
        # Email status
        if st.session_state.get('sender_email') and st.session_state.get('sender_password'):
            st.success(f"📧 Email notifications enabled for: {st.session_state.get('sender_email')}")
        else:
            st.warning("⚠️ Email notifications disabled. Configure email settings above to enable.")
        
        
        # Model retraining
        st.subheader("Model Management")
        if st.button("Retrain Model with New Data"):
            with st.spinner("Retraining model..."):
                # Generate new sample data
                df = generate_sample_data(1000)
                X, y, features = model.prepare_data(df)
                results = model.train(X, y)
                
                # Save updated model
                model.save_model('models/student_risk_model.pkl')
                
                st.success("Model retrained successfully!")
                st.write(f"New accuracy: {results['accuracy']:.3f}")

if __name__ == "__main__":
    main()
