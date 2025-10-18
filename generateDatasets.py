import pandas as pd
import json, re, sys
import os
from pathlib import Path
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
import html
import re

raw_data_path = Path("Patent_Data/Raw_data")

train_data_pdf = Path("Patent_Data/Train_data/pdfs")
train_data_labels = Path("Patent_Data/Train_data/labels")
val_data_pdf = Path("Patent_Data/Val_data/pdfs")
val_data_labels = Path("Patent_Data/Val_data/labels")


# Create directories if they don't exist
for path in [raw_data_path, train_data_pdf, train_data_labels, val_data_pdf, val_data_labels]:
    path.mkdir(parents=True, exist_ok=True)



def get_IP_data(limit=20):
    ip_files = []
    files = sorted(os.listdir(raw_data_path))[:limit]
    
    for i, file in enumerate(files):
        file_path = raw_data_path / file
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                ip_files.append(json.load(f))
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try with latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    ip_files.append(json.load(f))
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in file {file}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error with file {file}: {str(e)}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")

    relevant_fields = [
        # Identifiers
        "publication_number",
        "application_number",
        "patent_number",
        # Dates
        "date_published",
        "filing_date",
        "patent_issue_date",
        "abandon_date",
        # Status & Classes
        "decision",
        "main_cpc_label",
        "main_ipcr_label",
        # Retrievable Text
        "title",
        "abstract",
        "summary",
        'claims',
    ]

    ip_files = [{key: value for key, value in file.items() if key in relevant_fields} for file in ip_files]
    return ip_files


def sanitize_text(text):
    """
    Sanitize text for safe use in ReportLab paragraphs.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Unescape any HTML entities first
    text = html.unescape(text)
    
    # Replace problematic characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    
    # Clean up any remaining XML/HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace any remaining problematic characters
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#39;')
    
    # Remove any null bytes
    text = text.replace('\x00', '')
    
    return text

def add_page_number(canvas, doc):
    """Adds page number at the bottom of each page."""
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont('Helvetica', 9)
    canvas.drawRightString(7.5 * inch, 0.5 * inch, text)


def construct_pdf(patent_data, output_path):
    """Constructs a styled PDF document from a patent JSON object."""

    # --- Base doc setup ---
    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=36,
        title=patent_data.get("title", "Patent Document"),
        author=patent_data.get("applicant", "Unknown")
    )

    # --- Styles ---
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MetaData', fontSize=10, textColor=colors.grey))
    styles.add(ParagraphStyle(name='SectionHeader', fontSize=14, spaceAfter=6, leading=16, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='ClaimText', fontSize=11, leftIndent=20, spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyJustified', parent=styles['BodyText'], alignment=4, leading=14))

    Story = []

    # =============== COVER PAGE ===============
    title = sanitize_text(patent_data.get("title", "No Title Available"))
    Story.append(Paragraph(title, styles['Title']))
    Story.append(Spacer(1, 20))

    meta_fields = [
        ("Patent Number", patent_data.get("patent_number", "N/A")),
        ("Application Number", patent_data.get("application_number", "N/A")),
        ("Publication Date", patent_data.get("publication_date", "N/A")),
        ("Applicant", patent_data.get("applicant", "N/A")),
        ("Inventors", ", ".join(patent_data.get("inventors", [])) if isinstance(patent_data.get("inventors"), list) else patent_data.get("inventors", "N/A")),
    ]

    for label, value in meta_fields:
        Story.append(Paragraph(f"<b>{label}:</b> {value}", styles['MetaData']))
    Story.append(Spacer(1, 30))
    Story.append(Paragraph("This document contains information about the patent’s abstract, claims, and detailed description.", styles['BodyText']))
    Story.append(PageBreak())

    # =============== ABSTRACT ===============
    Story.append(Paragraph("Abstract", styles['SectionHeader']))
    Story.append(Paragraph(sanitize_text(patent_data.get("abstract", "No abstract available.")), styles['BodyJustified']))
    Story.append(Spacer(1, 12))

    # =============== SUMMARY ===============
    if "summary" in patent_data:
        Story.append(Paragraph("Summary", styles['SectionHeader']))
        Story.append(Paragraph(sanitize_text(patent_data.get("summary", "No summary available.")), styles['BodyJustified']))
        Story.append(Spacer(1, 12))

    # =============== CLAIMS ===============
    Story.append(Paragraph("Claims", styles['SectionHeader']))
    claims = patent_data.get("claims", [])
    if isinstance(claims, list):
        for i, claim in enumerate(claims, 1):
            Story.append(Paragraph(f"<b>Claim {i}.</b> {sanitize_text(claim)}", styles['ClaimText']))
    else:
        Story.append(Paragraph(sanitize_text(str(claims)), styles['BodyJustified']))
    Story.append(Spacer(1, 12))

    # =============== DESCRIPTION ===============
    description = patent_data.get("description", "")
    if description:
        Story.append(PageBreak())
        Story.append(Paragraph("Detailed Description", styles['SectionHeader']))
        if isinstance(description, list):
            for paragraph in description:
                Story.append(Paragraph(sanitize_text(paragraph), styles['BodyJustified']))
                Story.append(Spacer(1, 6))
        else:
            for para in str(description).split("\n\n"):
                Story.append(Paragraph(sanitize_text(para), styles['BodyJustified']))
                Story.append(Spacer(1, 6))

    # =============== IMAGES / FIGURES ===============
    figures = patent_data.get("figures", [])
    if figures and isinstance(figures, list):
        Story.append(PageBreak())
        Story.append(Paragraph("Figures", styles['SectionHeader']))
        for fig in figures:
            if os.path.exists(fig):
                img = Image(fig, width=5*inch, height=3*inch)
                img.hAlign = 'CENTER'
                Story.append(img)
                Story.append(Spacer(1, 12))
            Story.append(Paragraph(os.path.basename(fig), styles['MetaData']))

    # --- Build the PDF ---
    doc.build(Story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output_path



def generate_pdfs_from_data(data, train_output_dir=train_data_pdf, val_output_dir=val_data_pdf):
    """Generates train / val split PDFs for each patent"""

    # Ensure output directories exist
    Path(train_output_dir).mkdir(parents=True, exist_ok=True)
    Path(val_output_dir).mkdir(parents=True, exist_ok=True)

    split_index = int(0.85 * len(data))  # 85-15 train-val split

    for i, patent in enumerate(data):
        pdf_output_path = os.path.join(
            train_output_dir if i < split_index else val_output_dir,
            f"patent_{patent.get('publication_number', i)}.pdf"
        )
        label_output_path = os.path.join(
            train_data_labels if i < split_index else val_data_labels,
            f"patent_{patent.get('publication_number', i)}.json")
        
        # Save both the PDF and the JSON labels
        construct_pdf(patent, pdf_output_path)

        with open(label_output_path, 'w') as f:
            json.dump(patent, f)


if __name__ == "__main__":
    data = get_IP_data(limit=700)  # Adjust limit as needed
    generate_pdfs_from_data(data)