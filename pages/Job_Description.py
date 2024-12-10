import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import plotly.express as px  # data visualization
import plotly.graph_objects as go
import json
import re
from Resume_analysis import parse_pdf, parse_docx
from pdf2image import convert_from_bytes
from PIL import Image
from utils.pdf import show_pdf_in_sidebar

# Apply custom CSS for professional look
custom_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1f2d;
        color: white;
    }
    .main {
        color: black;
        padding: 30px;
        border-radius: 10px;
    }
    div[data-testid="stSidebarNav"] li div a span {
        color: white;
        padding: 0.5rem;
        width: 300px;
        border-radius: 0.5rem;
    }
    div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
    }
    [data-testid="stSidebarNav"]::before {
        content: "📄 Resume Parser AI 🤖";
        margin-left: 20px;
        margin-top: 20px;
        font-size: 30px;
        position: relative;
        top: -50px;
    }
    </style>
"""

# Custom CSS for styling
st.markdown("""
    <style>
        
        .stButton button {
            background-color: #2B4F76;
            color: white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            border: none;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background-color: #1E3D59;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stProgress > div > div {
            background-color: #2ECC71;
        }

        
        .match-score-card {
            text-align: center;
            padding: 2rem;
            background-color: #F8FAFC;
            border-radius: 8px;
            margin: 2rem 0;
        }
        
        .match-score-value {
            font-size: 3rem;
            color: #2ECC71;
            font-weight: bold;
        }
        
    </style>
""", unsafe_allow_html=True)

st.markdown(custom_style, unsafe_allow_html=True)

def create_skills_extraction_chain():
    """
    Create and return a LangChain for skills extraction
    """
    llm = ChatGroq(
        groq_api_key="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",  
        model_name="gemma-7b-it",
        temperature=0.5,
    )
    
    prompt = PromptTemplate(
        input_variables=["text_type", "content"],
        template="""Extract all skills from the following {text_type} and categorize them. 
        Return the result as a JSON string with the following structure:
        {{
            "technical_skills": [
                {{"skill": "skill_name", "priority": "Required/Preferred", "confidence": 95}}
            ],
            "soft_skills": [
                {{"skill": "skill_name", "priority": "Required/Preferred", "confidence": 90}}
            ],
            "domain_knowledge": [
                {{"skill": "skill_name", "priority": "Required/Preferred", "confidence": 85}}
            ]
        }}
        
        For Job Descriptions, set confidence to 100 if it's a required skill and 75 if it's preferred.
        For Resumes, set confidence based on how strongly the skill is demonstrated (0-100).
        
        Content:
        {content}
        
        Return ONLY the JSON string without any additional text."""
    )
    return LLMChain(llm=llm, prompt=prompt)

def extract_and_compare_skills(jd_text, resume_text):
    """
    Extract skills from both JD and resume, then compare them
    """
    skills_chain = create_skills_extraction_chain()
    
    # Extract skills from JD
    jd_response = skills_chain.run(text_type="Job Description", content=jd_text) #It invokes the AI model (via LangChain) to analyze the job description text.
    jd_skills = json.loads(re.search(r'\{.*\}', jd_response, re.DOTALL).group()) #\{.*\}: Matches everything inside curly braces {}.//re.DOTALL: Allows matching multi-line content
                #json.loads :- Converts the JSON string into a Python dictionary.
                
    # Extract skills from Resume
    resume_response = skills_chain.run(text_type="Resume", content=resume_text)
    resume_skills = json.loads(re.search(r'\{.*\}', resume_response, re.DOTALL).group())
    
    # Prepare comparison data
    comparison_data = []
    
    # Process JD skills
    for category in ['technical_skills', 'soft_skills', 'domain_knowledge']:
        for jd_skill in jd_skills.get(category, []):
            skill_name = jd_skill['skill'].lower()
            matching_skill = next(
                (s for s in resume_skills.get(category, []) 
                 if s['skill'].lower() == skill_name),
                None
            )
            
            comparison_data.append({
                'Skill': jd_skill['skill'],
                'Category': category.replace('_', ' ').title(),
                'Required in JD': jd_skill['priority'] == 'Required',
                'JD Priority': jd_skill['priority'],
                'Resume Match Score': matching_skill['confidence'] if matching_skill else 0,
                'Status': 'Found in both' if matching_skill else 'Missing in Resume'
            })
    
    # Add additional resume skills
    for category in ['technical_skills', 'soft_skills', 'domain_knowledge']:
        for resume_skill in resume_skills.get(category, []):
            skill_name = resume_skill['skill'].lower()
            if not any(d['Skill'].lower() == skill_name for d in comparison_data):
                comparison_data.append({
                    'Skill': resume_skill['skill'],
                    'Category': category.replace('_', ' ').title(),
                    'Required in JD': False,
                    'JD Priority': 'Not Mentioned',
                    'Resume Match Score': resume_skill['confidence'],
                    'Status': 'Additional Skill'
                })
    
    return pd.DataFrame(comparison_data)

def create_skills_visualizations(skills_df):
    """
    Create and return three visualizations for skills analysis
    """
    # 1. Skills Match Overview
    fig1 = px.bar(
        skills_df,
        x='Skill',
        y='Resume Match Score',
        color='Status',
        title='Skills Match Analysis',
        labels={'Resume Match Score': 'Match Score (%)', 'Skill': 'Skills'},
        color_discrete_map={
            'Found in both': '#2ECC71',
            'Missing in Resume': '#E74C3C',
            'Additional Skill': '#3498DB'
        }
    )
    fig1.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=24, color='#1E3D59'),
        margin=dict(t=80, b=100)
    )
    
    # 2. Category Distribution
    category_counts = skills_df.groupby(['Category', 'Status']).size().reset_index(name='Count')
    fig2 = px.bar(
        category_counts,
        x='Category',
        y='Count',
        color='Status',
        title='Skills Distribution by Category',
        barmode='group',
        color_discrete_map={
            'Found in both': '#2ECC71',
            'Missing in Resume': '#E74C3C',
            'Additional Skill': '#3498DB'
        }
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=24, color='#1E3D59')
    )
    
    # 3. Required Skills Radar
    required_skills = skills_df[skills_df['Required in JD']].copy()
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatterpolar(
        r=required_skills['Resume Match Score'],
        theta=required_skills['Skill'],
        fill='toself',
        name='Match Score',
        line=dict(color='#2ECC71')
    ))
    
    fig3.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#E8EEF2'
            ),
            angularaxis=dict(
                gridcolor='#E8EEF2'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title='Required Skills Match Analysis',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=24, color='#1E3D59'),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig1, fig2, fig3

def main():
    """
    Main application function
    """
    st.title("🎯 Resume Skills Match Analyzer")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        st.markdown(
            '<p class="section-description">Upload your resume in PDF, DOCX format</p>',
            unsafe_allow_html=True
        )
        resume_file = st.file_uploader("Drop your resume file here", type=["pdf", "docx"])
        if resume_file is not None:
            show_pdf_in_sidebar(resume_file)
            st.success("PDF is displayed in the sidebar!")
        else:
            st.info("Please upload a PDF file.")
        
    with col2:
        st.markdown("### 📝 Job Description")
        st.markdown(
            '<p class="section-description">Paste the job description you want to analyze</p>',
            unsafe_allow_html=True
        )
        jd_text = st.text_area("Paste the job description", height=200)
    
    if resume_file and jd_text:
        if resume_file.type == "text/plain":
            resume_text = resume_file.read().decode()
        elif resume_file.type == "application/pdf":
            resume_text = parse_pdf(resume_file)
        else:
            resume_text = parse_docx(resume_file)
        

        if st.button("🔍 Analyze Skills Match", use_container_width=True):
            with st.spinner("✨ Analyzing skills match..."):
                # Extract and compare skills
                skills_df = extract_and_compare_skills(jd_text, resume_text)
                
                # Create visualizations
                bar_chart, category_dist, radar_chart = create_skills_visualizations(skills_df)
                
                # Calculate match statistics
                required_skills = skills_df[skills_df['Required in JD']]
                match_score = required_skills['Resume Match Score'].mean()
                
                # Display overall match score
                st.markdown(
                    f"""
                    <div class="match-score-card">
                        <h2>Overall Skills Match Score</h2>
                        <div class="match-score-value">{round(match_score, 1)}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display visualizations in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Skills Overview",
                    "📈 Category Distribution",
                    "🎯 Required Skills",
                    "📝 Detailed Analysis"
                ])
                
                with tab1:
                    st.plotly_chart(bar_chart)
                
                with tab2:
                    st.plotly_chart(category_dist)
                
                with tab3:
                    st.plotly_chart(radar_chart)
                
                with tab4:
                    st.dataframe(
                        skills_df.style.background_gradient(
                            subset=['Resume Match Score'],
                            cmap='RdYlGn' #Specifies the color scheme to be used for the gradient. //RdYlGn:- This is a built-in colormap in the matplotlib library
                        )
                    )
                
                csv = skills_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Skills Analysis Report",
                    data=csv,
                    file_name="skills_analysis.csv",
                    mime="text/csv"
                )


if __name__ == '__main__':
    main()