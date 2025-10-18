import json
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset


# Fields expected in gold JSONs
RELEVANT_FIELDS: List[str] = [
    "publication_number",
    "application_number",
    "patent_number",
    "date_published",
    "filing_date",
    "patent_issue_date",
    "abandon_date",
    "decision",
    "main_cpc_label",
    "main_ipcr_label",
    "title",
    "abstract",
    "summary",
    "claims",
]


def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


class PatentIEDataset(Dataset):
    """
    Dataset for RLVR/GRPO IE on patent PDFs.

    Expects a manifest DataFrame with columns:
      - patent_id
      - pdf_path
      - gold_json_path
      - text (optional if `preload_text=True`)
    """

    def __init__(self, manifest_df: pd.DataFrame, preload_text: bool = False):
        self.df = manifest_df.reset_index(drop=True)
        self.preload_text = preload_text

        if self.preload_text and "text" not in self.df.columns:
            # Pre-extract on the fly (prefer preprocessing pass for speed)
            self.df = self.df.copy()
            self.df["text"] = self.df["pdf_path"].apply(self._load_pdf_text)

    def __len__(self):
        return len(self.df)

    def _docling_converter(self):
        try:
            from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except Exception as e:
            return None
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_ocr=False,
                        force_backend_text=True,
                        do_table_structure=False,
                        generate_picture_images=False,
                        generate_page_images=False,
                        generate_table_images=False,
                    )
                )
            }
        )

    def _load_pdf_text(self, pdf_path: str) -> str:
        converter = self._docling_converter()
        if converter is not None:
            res = converter.convert(str(pdf_path))
            return res.document.export_to_text()
        # Fallback to PyMuPDF if Docling unavailable
        import fitz  # type: ignore
        doc = fitz.open(pdf_path)
        return "\n\n".join(page.get_text("text") for page in doc)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        text = (
            row["text"] if ("text" in row and self.preload_text) else self._load_pdf_text(row["pdf_path"])
        )
        gold = json.load(open(row["gold_json_path"], "r"))

        fields_str = ", ".join(RELEVANT_FIELDS)
        prompt = (
            "Extract the following fields as JSON only (no extra text). "
            f"Fields: {{{fields_str}}}\n\n"
            f"DOCUMENT:\n{text}\n\n"
            "Return strictly a single JSON object with those keys."
        )

        return {
            "input_text": prompt,
            "gold": gold,
            "patent_id": row["patent_id"],
        }

