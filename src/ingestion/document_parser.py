import os
import io
import re
import logging
from typing import List, Dict, Any, Tuple
import chardet
import pandas as pd
from pypdf import PdfReader
import docx

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Removes excessive whitespace, linebreaks, and artifacts."""
    if not text:
        return ""
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def format_csv_row(row_dict: dict) -> str:
    """Converts a CSV row dictionary into a clean searchable line."""
    items = [f"{col}: {val}" for col, val in row_dict.items() if str(val).strip() != "" and str(val).lower() != "nan"]
    return ", ".join(items)

class DocumentParser:
    """Unified multi-format document parser supporting PDF, DOCX, TXT, and CSV formats."""

    @staticmethod
    def parse_file(file_path: str, filename: str = None, domain: str = "General") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = filename or os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path)

        content_blocks: List[Dict[str, Any]] = []

        # 1. TXT
        if ext == ".txt":
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected["encoding"] or "utf-8"
            text = raw_data.decode(encoding, errors="ignore")
            cleaned = clean_text(text)
            if cleaned:
                content_blocks.append({"content": cleaned, "page_number": 1, "section": "Body"})

        # 2. PDF
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                cleaned = clean_text(extracted)
                if cleaned:
                    content_blocks.append({
                        "content": cleaned,
                        "page_number": page_idx + 1,
                        "section": f"Page {page_idx + 1}"
                    })

        # 3. DOCX
        elif ext == ".docx":
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                cleaned_para = clean_text(para.text)
                if cleaned_para:
                    full_text.append(cleaned_para)
            if full_text:
                content_blocks.append({
                    "content": "\n".join(full_text),
                    "page_number": 1,
                    "section": "Document Body"
                })

        # 4. CSV
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                row_str = format_csv_row(row.to_dict())
                if row_str:
                    content_blocks.append({
                        "content": row_str,
                        "page_number": 1,
                        "section": f"Row {idx + 1}"
                    })
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        metadata = {
            "file_name": file_name,
            "file_type": ext,
            "file_size": file_size,
            "domain": domain
        }

        return content_blocks, metadata