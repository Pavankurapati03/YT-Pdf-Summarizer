from fpdf import FPDF
import streamlit as st

def generate_pdf(summary_text, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output(file_path)

def download_pdf(file_path, download_name):
    with open(file_path, "rb") as f:
        st.download_button(label="Download PDF", data=f, file_name=download_name)
