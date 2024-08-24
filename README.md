---
title: Summarizer
emoji: 🌖
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.37.1
app_file: main_app.py
pinned: false
license: gemma
---

# YT & PDF Summarizer

## Overview

This application is built using Streamlit and integrates with various APIs to provide summarization and Q&A features for YouTube videos and PDF documents. The app extracts key points and allows users to interactively query the summarized content. The app also offers additional features such as generating PDFs of the summaries and speaking the summaries aloud.

## Features

- **YouTube Video Summarization**: 
  - Extracts transcripts from YouTube videos.
  - Summarizes the transcript using Google's Gemini API.
  - Allows users to download the summary as a PDF or have it read aloud.
  - Interactive Q&A feature for asking questions about the summarized content.

- **PDF Document Summarization**:
  - Extracts text from uploaded PDF documents.
  - Summarizes the content using different summarization strategies (Brief, Detailed, Custom Length).
  - Generates PDF summaries with bullet points.
  - Allows users to query the summarized content interactively.

## Prerequisites

- Python 3.7 or later
- A Google Cloud account with access to Gemini API and Google Speech-to-Text API.
- Necessary Python libraries as mentioned in the `requirements.txt` file.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/yt-pdf-summarizer.git
    cd yt-pdf-summarizer
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
   - Create a `.env` file in the project root directory.
   - Add your Google API key:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open the app in your web browser (usually at `http://localhost:8501`).

3. Choose between summarizing a YouTube video or a PDF document using the sidebar.

### YouTube Video Summarization

- Paste the YouTube video link.
- Click on "Get Detailed Notes" to generate the summary.
- Download the summary as a PDF or listen to it using the available buttons.
- Ask follow-up questions to get more insights from the summary.

### PDF Document Summarization

- Upload one or more PDF documents.
- Choose the type of summary (Brief, Detailed, Custom Length).
- Click on "Submit & Process" to generate and download the summary as a PDF.
- Ask follow-up questions to query the summarized content interactively.

## File Structure

- `app.py`: The main Streamlit app script.
- `pdf_generator.py`: Contains functions to generate and download PDF summaries.
- `speech.py`: Handles text-to-speech conversion.
- `styles.css`: Custom CSS for the Streamlit app.
- `templates.py`: HTML templates for displaying chat history.
- `.env`: Environment variables (not included in the repository for security reasons).

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
- [Google Gemini API](https://cloud.google.com/)
- [Google Translate API](https://pypi.org/project/googletrans/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)


#### Made By [Pavankumar](https://www.linkedin.com/in/pavankumar-kurapati/)
For Queries, Reach out on [LinkedIn](https://www.linkedin.com/in/pavankumar-kurapati/)  
Resume Analyzer - Making Job Applications Easier

#### Check out my project : https://yt-pdf-summarizer.streamlit.app/

