import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import google.generativeai as genai
from langdetect import detect  # For language detection (optional)
from googletrans import Translator  # For translation
import os
import re
from pdf_generator_for_yt import generate_pdf, download_pdf  # Importing PDF functions
from speech import speak_text  # Importing speech function
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from templates import HTML_TEMPLATES
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListItem, ListFlowable
from reportlab.lib.styles import ParagraphStyle



# Load environment variables
load_dotenv()

# Set up the Streamlit App
st.set_page_config(page_title="YT & PDF Summarizer")

# Sidebar for navigation
st.sidebar.title("Choose the Task")
task = st.sidebar.radio("What would you like to summarize?", ("YouTube Video", "PDF Document"))

# YouTube Summarizer
if task == "YouTube Video":
    # Load environment variables
    load_dotenv()

    # Configure Google Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Prompt for summarization
    prompt = """You are a YouTube video summarizer. You will be taking the transcript text 
    and summarizing the entire video and providing the important summary in points 
    within 400 words. Please provide the summary of the text given here: """

    # Function to extract transcript from YouTube video
    def extract_transcript_details(youtube_video_url):
        try:
            video_id = youtube_video_url.split("=")[1]
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([i["text"] for i in transcript_text])
            return transcript
        except NoTranscriptFound:
            return None  # Return None if no transcript is found
        except Exception as e:
            raise e

    # Function to translate text if needed
    def translate_text(text, target_language='en'):
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text

    # Function to generate summary using Google Gemini
    def generate_gemini_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    # YouTube Summarizer with Conversational Chat

    # Function to generate an answer based on summary content
    def generate_answer(summary, question):
        question_prompt = f"""You are a helpful assistant. Based on the following summary of a YouTube video:
        
    Summary:
    {summary}
    
    The user has asked the following question:
    {question}
    
    Please provide a clear and concise answer based on the information in the summary."""
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(question_prompt)
        return response.text
    
    # Function to handle user input for the YouTube Summarizer
    def youtube_user_input(user_question, summary):
        # Generate the answer based on the summary and the user's question
        answer = generate_answer(summary, user_question)
        
        # Store the question and answer in session state
        st.session_state.youtube_chat_history.insert(0, {"question": user_question, "answer": answer})
    
    # Load CSS
    def load_css():
        with open("styles.css") as f:
            st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Function to display the chat history with HTML and CSS
    def display_youtube_chat_history():
        if st.session_state.youtube_chat_history:
            for entry in st.session_state.youtube_chat_history:
                st.write(HTML_TEMPLATES["user"].replace("{{MSG}}", entry['question']), unsafe_allow_html=True)
                st.markdown(HTML_TEMPLATES["bot"].replace("{{MSG}}", entry['answer']), unsafe_allow_html=True)

# Streamlit UI
st.title("YT Video Summarizer")

# Load CSS
load_css()

youtube_link = st.text_input("Paste the YouTube Video Link:")

st.markdown("""
<div style='color: #888888; font-size: 0.9em;'>
    **Note:** Please make sure the YouTube link only contains the video ID after the `v=` parameter. 
    If the link has additional parameters (like `&` followed by other text), 
    they will be ignored to correctly retrieve the video thumbnail and transcript.
</div>
""", unsafe_allow_html=True)

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        # Detect language (optional)
        detected_language = detect(transcript_text)
        st.write(f"Detected Language: {detected_language}")

        # Translate if needed
        if detected_language != 'en':
            transcript_text = translate_text(transcript_text, target_language='en')

        summary = generate_gemini_content(transcript_text, prompt)
        st.session_state.summary = summary  # Store the summary in session state
        st.session_state.youtube_chat_history = []  # Initialize chat history
        st.markdown("## Detailed Notes:")
        st.write(summary)
    else:
        st.error("No transcript available for this video. Summary cannot be generated.")

# Check if the summary exists in the session state
if 'summary' in st.session_state:
    summary = st.session_state.summary

    # Option to download the summary as a PDF
    if st.button("Download Summary as PDF"):
        pdf_path = os.path.join("temp", "summary.pdf")
        if not os.path.exists("temp"):
            os.makedirs("temp")
        generate_pdf(summary, pdf_path)
        download_pdf(pdf_path, "summary.pdf")

    # Option to speak the summary aloud
    if st.button("Speak Summary"):
        speak_text(summary)

    # Follow-up Question Feature
    st.markdown("## Have a Question About the Video?")
    question = st.text_input("Enter your question:")

    if question and st.button("Get Answer"):
        youtube_user_input(question, summary)
        display_youtube_chat_history()



# PDF Summarizer
elif task == "PDF Document":
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Load CSS
    def load_css():
        with open("styles.css") as f:
            st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def clean_text(text):
        # Remove or replace problematic characters
        return text.encode('utf-8', 'replace').decode('utf-8')

    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = clean_text(page_text)
                    text += cleaned_text
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        cleaned_chunks = [clean_text(chunk) for chunk in text_chunks]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(cleaned_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible in bullet points from the provided context. Highlight key information using markdown for bold text (e.g., **this is important**).
        If the answer is not in the provided context, just say, "answer is not available in the context." Don't provide a wrong answer.
        
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


    def summarize_pdf_content(text_chunks, summary_type, custom_length):
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        
        # Adjust summarization chain based on summary type
        if summary_type == "Brief":
            summary_chain = load_summarize_chain(model, chain_type="map_reduce")
        elif summary_type == "Detailed":
            summary_chain = load_summarize_chain(model, chain_type="refine")
        else:  # Custom Length
            summary_chain = load_summarize_chain(model, chain_type="map_reduce")
        
        summary = summary_chain({"input_documents": docs})
        summary_text = summary["output_text"]
        
        # If custom length is selected, truncate to the desired number of sentences
        if summary_type == "Custom Length":
            sentences = summary_text.split('. ')
            summary_text = '. '.join(sentences[:custom_length]) + '.'

        # Add bullet points for formatting
        bullet_summary = "\n".join([f"- {line.strip()}" for line in summary_text.split('. ') if line.strip()])
        
        return bullet_summary


    # Create PDF with bullet point summary
    def create_pdf(summary_text):
        pdf_file_path = "summary.pdf"
        doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
        elements = []
        
        bullet_style = ParagraphStyle(
            'Bullet',
            leftIndent=20,
            bulletIndent=10,
            spaceBefore=5,
            spaceAfter=5,
        )
        
        bullet_points = summary_text.split("\n")
        items = [ListItem(Paragraph(point, bullet_style)) for point in bullet_points if point.strip()]
        
        bullet_list = ListFlowable(items, bulletType='bullet', bulletFontSize=10)
        elements.append(bullet_list)
        
        doc.build(elements)
        
        return pdf_file_path

    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.session_state.chat_history.insert(0, {"question": user_question, "answer": response["output_text"]})

    def display_chat_history():
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.write(HTML_TEMPLATES["user"].replace("{{MSG}}", entry['question']), unsafe_allow_html=True)
                # Use Markdown to display the response with highlighted key information
                st.markdown(HTML_TEMPLATES["bot"].replace("{{MSG}}", entry['answer']), unsafe_allow_html=True)


    def main():
        st.header("Chat with PDF")
        load_css()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)
            display_chat_history()

        with st.sidebar:
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            
            # Add UI elements for summary customization
            summary_type = st.selectbox("Choose the summary type:", ["Brief", "Detailed", "Custom Length"])
            custom_length = st.slider("Select the number of sentences for Custom Length:", 1, 100, 5)
            
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        
                        summary = summarize_pdf_content(text_chunks, summary_type, custom_length)
                        
                        pdf_path = create_pdf(summary)
                        st.download_button(
                            label="Download Summary as PDF",
                            data=open(pdf_path, "rb"),
                            file_name="summary.pdf",
                            mime="application/pdf",
                        )
                        
                        st.success("Processing and summarization completed!")
                else:
                    st.warning("Please upload at least one PDF file.")


    if __name__ == "__main__":
        main()
