import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import docx
from utils.pdf import show_pdf_in_sidebar

st.set_page_config(layout='wide')

#API key....
GROQ_API_KEY = "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"

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
st.markdown(custom_style, unsafe_allow_html=True)

# Create LLM chain for resume analysis
def create_llm_chain():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="gemma-7b-it",
        temperature=0.5,
    )   
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    return LLMChain(llm=llm, prompt=prompt)

llm_chain = create_llm_chain()

def extract_skills(resume_text):
    try:
        prompt = f"""Analyze the following resume and extract ONLY the skills. Categorize them into:
                1. Technical Skills (programming languages, tools, frameworks, etc.)
                2. Soft Skills (communication, leadership, etc.)
                3. Domain Knowledge (industry-specific knowledge)

                Resume:
                {resume_text}

                Please format the output as a clear list under each category. If a category has no skills, indicate 'None found'.
                Also provide a confidence score (0-100%) for each identified skill based on the context and evidence in the resume."""

        response = llm_chain.run(question=prompt)
        return response, True
    except Exception as e:
        st.error(f"An error occurred while extracting skills: {str(e)}")
        return None, False

def generate_answer(resume_text):
    try:
        prompt = f"""Analyze the following resume and extract key information:

                {resume_text}

                Provide a detailed analysis including:
                0. Personal Introduction: Write a brief personal introduction suitable for an interview setting. This should summarize the candidate's key qualifications, relevant experiences, and personal traits that make them a strong fit for the role.
                1. Contact Information
                2. Education
                3. Work Experience
                4. Skills
                5. Projects
                6. Certifications
                7. Summary of qualifications

                For each section, provide detailed information in a structured format."""

        response = llm_chain.run(question=prompt)
        return response, True
    except Exception as e:
        st.error(f"An error occurred while processing the resume: {str(e)}")
        return None, False

# Function to parse PDF files
def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to parse DOCX files
def parse_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Streamlit UI
st.title("📄 Resume Parser and Analyzer")
st.write("""
### Welcome to the Resume Parser AI tool! 
Easily upload a resume and get a detailed analysis with breakdowns of skills, education, experience, and more. You can also download the analysis as a CSV for future reference.
""")

# File uploader
st.subheader("Upload Your Resume")
uploaded_file = st.file_uploader("Supported file types: PDF, DOCX", type=["pdf", "docx"])

if uploaded_file is not None:
    show_pdf_in_sidebar(uploaded_file)
    st.success("PDF is displayed in the sidebar!")
    # Read the file content based on file type
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode()
    elif uploaded_file.type == "application/pdf":
        resume_text = parse_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = parse_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a TXT, PDF, or DOCX file.")
        st.stop()

    # Add buttons for different analysis options
    col1, col2 = st.columns(2)
    with col1:
        # if st.button("Full Resume Analysis"):
            with st.spinner('Analyzing resume... Please wait.'):
                analysis, success = generate_answer(resume_text)

            if success and analysis:
                st.subheader("Complete Resume Analysis")
                st.success("Your resume has been successfully analyzed. Here's the breakdown:")
                st.write(analysis)

                # Convert analysis to DataFrame
                analysis_lines = analysis.split('\n')
                data = {}
                current_section = ""
                for line in analysis_lines:
                    if line.strip() and ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip() #Add the key-value pair to the data dictionary after stripping any extra spaces.
                    elif line.strip() and '.' in line:
                        current_section = line.split('.', 1)[1].strip()
                    elif line.strip():
                        if current_section in data:
                            data[current_section] += f" {line.strip()}"
                        else:
                            data[current_section] = line.strip()

                df = pd.DataFrame([data])

                # Save to CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis as CSV",
                    data=csv,
                    file_name="resume_analysis.csv",
                    mime="text/csv",
                )

    with col2:
            with st.spinner('Extracting skills... Please wait.'):
                skills_analysis, success = extract_skills(resume_text)

            if success and skills_analysis:
                st.subheader("Skills Analysis")
                st.success("Skills have been successfully extracted from your resume:")
                st.write(skills_analysis)

                # Create a downloadable version of skills
                skills_df = pd.DataFrame({"Skills Analysis": [skills_analysis]})
                skills_csv = skills_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Skills Analysis as CSV",
                    data=skills_csv,
                    file_name="skills_analysis.csv",
                    mime="text/csv",
                )
else:
    st.info("Please upload a resume file to begin the analysis.")