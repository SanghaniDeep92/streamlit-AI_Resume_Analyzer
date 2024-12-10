import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Custom style for the app
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
"""

# Inject custom CSS
st.markdown(custom_style, unsafe_allow_html=True)

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from Word documents
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text


# Function to process resume and extract skills (this is a placeholder function)
def extract_skills(text):
    skills = ['Python', 'Java', 'JavaScript', 'C++', 'SQL', 'Machine Learning', 'Deep Learning', 'Data Science', 'Django', 'Flask', 'HTML', 'CSS']
    found_skills = [skill for skill in skills if skill.lower() in text.lower()]
    return found_skills


# Function to compute match score based on cosine similarity
def compute_match_score(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    cosine_sim = cosine_similarity(vectors[0], vectors[1])
    return cosine_sim[0][0] * 100  # return score as a percentage


# Main Streamlit app function
def main():
    st.title("🎯 Resume Comparison")

    # Directly show the "Best Resume Find" panel
    st.header("Select Multiple Resumes")
    resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_desc_text = st.text_area("Paste the job description here")

    if resume_files and job_desc_text:
        st.write(f"Number of resumes uploaded: {len(resume_files)}")
        all_resumes_analysis = []

        for resume_file in resume_files:
            file_extension = resume_file.name.split('.')[-1]
            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(resume_file)
            elif file_extension == 'docx':
                resume_text = extract_text_from_docx(resume_file)

            # Extract skills and calculate match score
            skills = extract_skills(resume_text)
            match_score = compute_match_score(resume_text, job_desc_text)

            all_resumes_analysis.append({
                "Resume Name": resume_file.name,
                "Skills Found": ", ".join(skills),
                "Match Score": match_score  # Store match score as a float, not a string
            })

        # Convert to DataFrame for easy display
        df = pd.DataFrame(all_resumes_analysis)
        st.write("Match Results:")
        st.dataframe(df)

        # Find best match (highest score)
        df['Match Score'] = df['Match Score'].astype(float)  # Ensure the 'Match Score' column is of type float
        best_resume = df.loc[df['Match Score'].idxmax()]
        resume_name_without_extension = os.path.splitext(best_resume['Resume Name'])[0]

        # Styled output
        st.markdown(f"<h3 style='color: green;'>Best Resume: <strong>{resume_name_without_extension}</strong> with Match Score: <span style='color: blue;'>{best_resume['Match Score']:.2f}%</span></h3>", unsafe_allow_html=True)


    else:
        st.info("Please upload resumes and provide a job description.")

if __name__ == "__main__":
    main()
