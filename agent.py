"""
LangGraph Agent Workflow for Student Intervention Recommendations
"""

from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
import json
import os

class StudentInterventionAgent:
    """AI agent for generating personalized intervention strategies"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            # Use a mock LLM for demonstration
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=self.openai_api_key
            )
    
    def _define_success_metrics(self, risk_category: str, risk_factors: List[str]) -> List[str]:
        """Define success metrics for the intervention plan"""
        metrics = []
        
        if "Low GPA" in risk_factors:
            metrics.append("GPA improvement of at least 0.3 points")
        if "Poor Attendance" in risk_factors:
            metrics.append("Attendance rate above 85%")
        if "Incomplete Assignments" in risk_factors:
            metrics.append("Assignment completion rate above 90%")
        if "Low Participation" in risk_factors:
            metrics.append("Participation score improvement of 20+ points")
        
        # General metrics
        if risk_category == "High Risk":
            metrics.extend([
                "No academic probation status",
                "Regular advisor meetings attended",
                "Tutoring sessions completed as scheduled"
            ])
        elif risk_category == "Medium Risk":
            metrics.extend([
                "Maintained or improved current performance",
                "Completed recommended workshops",
                "Regular check-ins with advisor"
            ])
        
        return metrics
    
    def analyze_student(self, student_data: Dict[str, Any], risk_score: int) -> Dict[str, Any]:
        """Main method to analyze a student and generate recommendations"""
        # Determine risk category
        if risk_score == 0:
            risk_category = "Low Risk"
        elif risk_score == 1:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        # Identify risk factors
        risk_factors = []
        if student_data.get('gpa', 4.0) < 2.5:
            risk_factors.append("Low GPA")
        if student_data.get('attendance_rate', 1.0) < 0.7:
            risk_factors.append("Poor Attendance")
        if student_data.get('assignment_completion', 1.0) < 0.6:
            risk_factors.append("Incomplete Assignments")
        if student_data.get('quiz_scores', 100) < 60:
            risk_factors.append("Low Quiz Performance")
        if student_data.get('participation_score', 100) < 50:
            risk_factors.append("Low Participation")
        if student_data.get('lms_activity', 0) < 5:
            risk_factors.append("Minimal LMS Engagement")
        if student_data.get('late_submissions', 0) > 3:
            risk_factors.append("Frequent Late Submissions")
        if student_data.get('office_hours_visits', 0) == 0:
            risk_factors.append("No Office Hours Visits")
        
        # Generate interventions
        recommendations = []
        if risk_category == "High Risk":
            recommendations.extend([
                "Immediate one-on-one meeting with academic advisor",
                "Mandatory tutoring sessions (2-3 times per week)",
                "Parent/guardian notification and involvement",
                "Academic probation monitoring",
                "Peer mentoring program assignment",
                "Regular check-ins with faculty advisor"
            ])
        elif risk_category == "Medium Risk":
            recommendations.extend([
                "Weekly check-ins with academic advisor",
                "Tutoring referral for specific subjects",
                "Study group participation encouragement",
                "Time management workshop attendance",
                "Office hours requirement (minimum 2 visits per month)"
            ])
        else:
            recommendations.extend([
                "Regular progress monitoring",
                "Optional study skills workshops",
                "Encouragement to maintain current performance",
                "Peer study group suggestions"
            ])
        
        # Add specific interventions based on risk factors
        if "Low GPA" in risk_factors:
            recommendations.append("GPA recovery plan with specific grade targets")
        if "Poor Attendance" in risk_factors:
            recommendations.append("Attendance tracking and intervention protocol")
        if "Incomplete Assignments" in risk_factors:
            recommendations.append("Assignment completion support and deadline management")
        if "Low Participation" in risk_factors:
            recommendations.append("Participation improvement strategies and engagement techniques")
        
        # Create intervention plan
        if risk_category == "High Risk":
            timeline = {
                "immediate": [r for r in recommendations if "immediate" in r.lower() or "mandatory" in r.lower()],
                "week_1": [r for r in recommendations if "weekly" in r.lower() or "tutoring" in r.lower()],
                "month_1": [r for r in recommendations if "month" in r.lower() or "workshop" in r.lower()],
                "ongoing": [r for r in recommendations if "monitoring" in r.lower() or "regular" in r.lower()]
            }
        elif risk_category == "Medium Risk":
            timeline = {
                "week_1": [r for r in recommendations if "weekly" in r.lower()],
                "month_1": [r for r in recommendations if "month" in r.lower() or "workshop" in r.lower()],
                "ongoing": [r for r in recommendations if "monitoring" in r.lower() or "encouragement" in r.lower()]
            }
        else:
            timeline = {
                "ongoing": recommendations
            }
        
        intervention_plan = {
            "timeline": timeline,
            "priority_level": risk_category,
            "estimated_duration": "4-8 weeks" if risk_category != "Low Risk" else "2-4 weeks",
            "success_metrics": self._define_success_metrics(risk_category, risk_factors)
        }
        
        # Define next steps
        next_steps = []
        if risk_category == "High Risk":
            next_steps.extend([
                "Schedule immediate meeting with student",
                "Notify academic advisor and department head",
                "Set up tutoring appointments",
                "Create monitoring schedule"
            ])
        elif risk_category == "Medium Risk":
            next_steps.extend([
                "Schedule weekly check-in meeting",
                "Refer to appropriate support services",
                "Set up progress tracking"
            ])
        else:
            next_steps.extend([
                "Continue regular monitoring",
                "Provide encouragement and support",
                "Track progress indicators"
            ])
        
        return {
            "risk_category": risk_category,
            "risk_factors": risk_factors,
            "intervention_plan": intervention_plan,
            "recommendations": recommendations,
            "next_steps": next_steps,
            "analysis_complete": True
        }
    
    def generate_natural_language_summary(self, analysis_result: Dict[str, Any], student_name: str) -> str:
        """Generate a natural language summary of the analysis"""
        risk_category = analysis_result["risk_category"]
        risk_factors = analysis_result["risk_factors"]
        recommendations = analysis_result["recommendations"][:3]  # Top 3 recommendations
        
        summary = f"""
        **Student Risk Analysis Summary for {student_name}**
        
        **Risk Level:** {risk_category}
        
        **Key Risk Factors Identified:**
        {', '.join(risk_factors) if risk_factors else 'No specific risk factors identified'}
        
        **Recommended Interventions:**
        {chr(10).join(f'• {rec}' for rec in recommendations)}
        
        **Next Steps:**
        {chr(10).join(f'• {step}' for step in analysis_result['next_steps'][:3])}
        """
        
        return summary.strip()

# Mock LLM responses for demonstration when OpenAI API is not available
def get_mock_llm_response(prompt: str) -> str:
    """Generate mock responses for demonstration purposes"""
    if "intervention" in prompt.lower():
        return """
        Based on the student's risk profile, I recommend:
        1. Immediate academic support intervention
        2. Regular progress monitoring
        3. Peer mentoring program
        4. Study skills development
        """
    elif "explanation" in prompt.lower():
        return """
        The student's risk factors include low GPA, poor attendance, and incomplete assignments.
        These factors combined indicate a high probability of academic difficulty.
        """
    else:
        return "This is a mock response for demonstration purposes."

if __name__ == "__main__":
    # Test the agent
    agent = StudentInterventionAgent()
    
    # Sample student data
    sample_student = {
        'gpa': 2.1,
        'attendance_rate': 0.65,
        'assignment_completion': 0.55,
        'quiz_scores': 45,
        'participation_score': 30,
        'lms_activity': 3,
        'late_submissions': 5
    }
    
    # Analyze student
    result = agent.analyze_student(sample_student, risk_score=2)
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Generate summary
    summary = agent.generate_natural_language_summary(result, "John Doe")
    print("\nSummary:")
    print(summary)
