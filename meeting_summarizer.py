import streamlit as st
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import PyPDF2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def summarize_key_points(transcript):
    """Generate key points using Gemini."""
    prompt = (
        "Summarize the following meeting transcript into concise key points (bullet points, max 5). "
        "Focus on main topics, decisions, and outcomes:\n\n" + transcript
    )
    response = model.generate_content(prompt)
    return response.text

def extract_action_items(transcript):
    """Extract action items with owners and deadlines using Gemini."""
    prompt = (
        "From the following meeting transcript, extract action items as a bullet list. "
        "Include the responsible person and any deadlines if mentioned:\n\n" + transcript
    )
    response = model.generate_content(prompt)
    return response.text

def analyze_sentiment(transcript):
    """Analyze sentiment of the transcript using VADER."""
    scores = analyzer.polarity_scores(transcript)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return f"{sentiment} (Compound Score: {compound:.2f})"

def main():
    st.title("📝 Meeting Notes Summarizer")
    st.write("Upload a transcript (PDF/text) or paste it below to generate key points, action items, and sentiment analysis.")

    # File upload
    uploaded_file = st.file_uploader("Upload Transcript (PDF or Text)", type=["pdf", "txt"])
    transcript = ""

    # Text input
    text_input = st.text_area("Or Paste Transcript Here", height=200)

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            transcript = extract_text_from_pdf(uploaded_file)
        else:
            transcript = uploaded_file.read().decode("utf-8")
        st.text_area("Extracted Transcript", transcript, height=200)

    if text_input:
        transcript = text_input

    if st.button("Summarize"):
        if transcript:
            with st.spinner("Processing..."):
                key_points = summarize_key_points(transcript)
                action_items = extract_action_items(transcript)
                sentiment = analyze_sentiment(transcript)

                # Display results in a structured format
                st.markdown("### Summary")
                st.markdown("#### Key Points")
                st.markdown(key_points)
                st.markdown("#### Action Items")
                st.markdown(action_items)
                st.markdown("#### Sentiment Analysis")
                st.markdown(sentiment)
        else:
            st.error("Please provide a transcript via upload or text input.")

if __name__ == "__main__":
    main()