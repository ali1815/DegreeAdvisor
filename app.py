import streamlit as st
import easyocr
import pandas as pd
import openai

# Load the data
careers_df = pd.read_csv('careers.csv')
programs_df = pd.read_csv('programs.csv')
scholarships_df = pd.read_csv('scholarships.csv')
universities_df = pd.read_csv('universities.csv')

# Set OpenAI API key for generating degree recommendations
openai.api_key = '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'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add more languages if needed

# Streamlit UI
st.title("Degree Recommendation System for Pakistani Students")

# File upload section
st.subheader("Upload your Transcript")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])

def extract_text_from_image(image):
    """Extracts text from an image using EasyOCR."""
    result = reader.readtext(image)
    text = ' '.join([item[1] for item in result])  # Extract the text from OCR results
    return text

if uploaded_file is not None:
    # Process the uploaded file (image format)
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        transcript_text = extract_text_from_image(uploaded_file)
    else:
        st.warning("Only image files (JPEG, PNG, JPG) are supported for OCR.")
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
    scholarships = scholarships_df[['name', 'eligibility', 'coverage', 'application_deadline']].to_string(index=False)
    universities = universities_df[['name', 'location', 'ranking', 'tuition_range']].to_string(index=False)
    application_steps = "1. Register on the university portal\n2. Submit application form\n3. Attach required documents"

    return scholarships, universities, application_steps
