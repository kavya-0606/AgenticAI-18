# 🎓 AI Early Warning System

A comprehensive Agentic AI Early Warning System web application designed to identify at-risk students and provide proactive intervention strategies for educators.

## 🌟 Features

### Backend & Machine Learning
- **Advanced ML Models**: XGBoost and RandomForest classifiers for student risk prediction
- **Risk Classification**: Categorizes students into Low, Medium, and High risk levels
- **Feature Engineering**: Analyzes GPA, attendance, participation, LMS activity, and more
- **Model Persistence**: Saves and loads trained models for consistent predictions

### LangGraph Agent Workflow
- **Intelligent Analysis**: AI agent analyzes predictions and generates personalized interventions
- **Workflow Automation**: Multi-step process for risk assessment and recommendation generation
- **Natural Language Processing**: Generates human-readable summaries and explanations
- **Intervention Planning**: Creates structured timelines and success metrics

### Streamlit Frontend
- **Interactive Dashboard**: Real-time visualization of student risk distribution
- **Data Upload**: CSV file upload with validation and preprocessing
- **Individual Analysis**: Detailed student profiles with AI-generated recommendations
- **Risk Factor Analysis**: Feature importance and correlation analysis
- **Real-time Alerts**: Immediate flagging of high-risk students

### Enhanced Features
- **Explainability**: Shows which features influenced each student's risk prediction
- **Proactive Alerts**: Flags high-risk students immediately with visual indicators
- **Interactive Visualizations**: Plotly charts for trends, distributions, and comparisons
- **Data Validation**: Ensures data quality and format compliance
- **Export Capabilities**: Generate detailed student reports

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### First Run

The application will automatically:
- Generate sample student data (1000 records)
- Train an XGBoost model
- Save the model for future use
- Display the interactive dashboard

## 📊 Data Format

### Required CSV Columns
When uploading student data, ensure your CSV includes these columns:

- `student_id`: Unique student identifier
- `name`: Student name
- `gpa`: Grade Point Average (0-4 scale)
- `attendance_rate`: Attendance percentage (0-1 scale)
- `assignment_completion`: Assignment completion rate (0-1 scale)
- `quiz_scores`: Average quiz scores (0-100 scale)
- `participation_score`: Participation score (0-100 scale)

### Optional Columns
- `lms_activity`: Learning Management System activity count
- `late_submissions`: Number of late submissions
- `office_hours_visits`: Office hours visits count
- `study_group_participation`: Study group participation rate (0-1)
- `previous_semester_gpa`: Previous semester GPA

## 🎯 Usage Guide

### 1. Dashboard Overview
- View system-wide metrics and risk distribution
- Monitor high-risk student alerts
- Analyze performance trends across all students

### 2. Upload Data
- Upload CSV files with student performance data
- Automatic data validation and preprocessing
- Generate predictions for uploaded students

### 3. Student Analysis
- Select individual students for detailed analysis
- View AI-generated intervention recommendations
- Access feature importance explanations
- Read natural language summaries

### 4. Risk Factors
- Explore model feature importance
- Analyze correlations between different factors
- Understand what drives risk predictions

### 5. Settings
- Configure OpenAI API key for enhanced AI features
- Retrain models with new data
- View system information

## 🤖 AI Agent Workflow

The LangGraph agent follows this workflow:

1. **Risk Analysis**: Determines risk category based on ML predictions
2. **Factor Identification**: Identifies specific risk factors (low GPA, poor attendance, etc.)
3. **Intervention Generation**: Creates personalized intervention strategies
4. **Plan Creation**: Develops structured intervention timelines
5. **Recommendation Finalization**: Provides actionable next steps

## 📈 Risk Categories

- **Low Risk**: GPA ≥ 3.0, Attendance ≥ 80%, Assignment completion ≥ 75%
- **Medium Risk**: GPA 2.5-3.0, Attendance 70-80%, Assignment completion 60-75%
- **High Risk**: GPA < 2.5, Attendance < 70%, Assignment completion < 60%

## 🔧 Configuration

### OpenAI API (Optional)
For enhanced AI recommendations, set your OpenAI API key in the Settings page or as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Model Customization
- Modify `train_model.py` to adjust model parameters
- Update feature engineering in `utils.py`
- Customize risk thresholds in the model training

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── train_model.py         # ML model training and prediction
├── agent.py              # LangGraph agent workflow
├── utils.py              # Data processing utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Saved ML models (created automatically)
```

## 🛠️ Technical Details

### Machine Learning
- **Algorithm**: XGBoost Classifier (default) or RandomForest
- **Features**: 10+ student performance indicators
- **Validation**: Cross-validation with stratified sampling
- **Performance**: Typically achieves 85%+ accuracy

### Frontend
- **Framework**: Streamlit with custom CSS
- **Visualizations**: Plotly for interactive charts
- **State Management**: Session state for data persistence
- **Caching**: Optimized for performance

### Agent System
- **Framework**: LangGraph for workflow orchestration
- **LLM**: OpenAI GPT-3.5-turbo (optional)
- **Fallback**: Mock responses when API unavailable

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**: The app will automatically train a new model on first run
2. **CSV upload errors**: Check column names and data formats
3. **API errors**: Ensure OpenAI API key is valid (if using)
4. **Memory issues**: Reduce sample data size in `generate_sample_data()`

### Performance Tips

- Use smaller datasets for faster processing
- Enable Streamlit caching for repeated operations
- Close unused browser tabs to free memory

## 🔮 Future Enhancements

- Real-time data integration with LMS systems
- Email/SMS alerts for high-risk students
- Integration with academic advising systems
- Advanced NLP for intervention text analysis
- Mobile-responsive design
- Multi-language support

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or support, please open an issue in the project repository.

---

**Built with ❤️ for educators and students worldwide**


