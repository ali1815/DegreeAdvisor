import streamlit as st
import pytesseract
from PIL import Image
import openai
import pandas as pd

# Load the data
careers_df = pd.read_csv('/mnt/data/careers.csv')
programs_df = pd.read_csv('/mnt/data/programs.csv')
scholarships_df = pd.read_csv('/mnt/data/scholarships.csv')
universities_df = pd.read_csv('/mnt/data/universities.csv')

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# Streamlit UI
st.title("Degree Recommendation System for Pakistani Students")

# File upload section
st.subheader("Upload your Transcript")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    # Process the uploaded file
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Convert image to text using OCR
        img = Image.open(uploaded_file)
        transcript_text = pytesseract.image_to_string(img)
    elif uploaded_file.type == "application/pdf":
        # For PDF, you would need a different approach, but for simplicity, we'll leave it out
        st.warning("PDF support coming soon!")
        transcript_text = ""
    
    # Display the transcript content
    st.subheader("Transcript Extracted Text")
    st.text_area("Extracted Text", transcript_text, height=300)

    # If transcript is extracted, ask for degree suggestion
    if transcript_text:
        # Process the extracted text with GPT-3 for recommendations
        st.subheader("Degree Suggestions Based on Your Transcript")
        degree_recommendation = generate_recommendations(transcript_text)
        st.write(degree_recommendation)

        # Fetch Scholarship and University suggestions
        st.subheader("Scholarships & Universities Suggestions")
        scholarships, universities, application_steps = fetch_scholarships_and_universities(degree_recommendation)
        st.write(f"**Available Scholarships**: {scholarships}")
        st.write(f"**Recommended Universities**: {universities}")
        st.write(f"**Application Procedure**: {application_steps}")


# Function to generate degree recommendations using GPT-3
def generate_recommendations(transcript_data):
    prompt = f"Suggest the best degree options for a student based on the following academic transcript:\n{transcript_data}\nPlease provide a list of degrees, scholarships, universities, and application procedures."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response['choices'][0]['text'].strip()

# Function to fetch scholarships and universities based on the degree suggestion
def fetch_scholarships_and_universities(degree_recommendation):
    # Example hardcoded data (Replace with actual data fetching from a database or API)
    scholarships = scholarships_df[['name', 'eligibility', 'coverage', 'application_deadline']].to_string(index=False)
    universities = universities_df[['name', 'location', 'ranking', 'tuition_range']].to_string(index=False)
    application_steps = "1. Register on the university portal\n2. Submit application form\n3. Attach required documents"

    return scholarships, universities, application_steps
