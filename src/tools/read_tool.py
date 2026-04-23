import json
import os
import posixpath
import zipfile
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, unquote
import xml.etree.ElementTree as ET

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from PIL import Image, ExifTags
from pypdf import PdfReader

from .base_tool import BaseTool


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
PKG_REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL_NS = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
SSML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


class ReadTool(BaseTool):
    """Read common local/remote file types for the tag-based TIR pipeline."""

    def __init__(
        self,
        allowed_roots: Optional[Sequence[str]] = None,
        timeout: int = 30,
        max_chars: int = 8000,
        max_pages: int = 10,
        max_rows: int = 200,
        max_sheets: int = 5,
        enable_image_ocr: bool = False,
    ):
        self.allowed_roots = [self._norm_path(p) for p in (allowed_roots or []) if p]
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_pages = max_pages
        self.max_rows = max_rows
        self.max_sheets = max_sheets
        self.enable_image_ocr = enable_image_ocr

    @property
    def name(self) -> str:
        return "read"

    @property
    def trigger_tag(self) -> str:
        return "read"

    async def execute(self, content: str, **kwargs) -> str:
        spec = self._parse_spec(content)
        max_chars = int(spec.get("max_chars", self.max_chars) or self.max_chars)
        timeout = int(kwargs.get("timeout", spec.get("timeout", self.timeout) or self.timeout))

        if spec.get("url"):
            text = self._read_remote(spec["url"], spec, timeout=timeout)
        else:
            text = self._read_local(spec.get("path", ""), spec)

        return self._truncate(text, max_chars)

    # ------------------------------
    # Request parsing / dispatch
    # ------------------------------
    def _parse_spec(self, content: str) -> Dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            raise ValueError("Empty read request")

        if raw.startswith("{"):
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise ValueError("<read> payload must be a JSON object")
            return spec

        if raw.startswith("http://") or raw.startswith("https://"):
            return {"url": raw}
        return {"path": raw}

    def _read_local(self, path: str, spec: Dict[str, Any]) -> str:
        if not path:
            raise ValueError("Missing 'path' in read request")

        full_path = self._resolve_local_path(path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        self._check_path_allowed(full_path)

        with open(full_path, "rb") as f:
            data = f.read()
        suffix = os.path.splitext(full_path)[1].lower()
        return self._read_from_bytes(data, spec, source_label=full_path, suffix_hint=suffix)

    def _read_remote(self, url: str, spec: Dict[str, Any], timeout: int) -> str:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        parsed = urlparse(url)
        suffix = os.path.splitext(parsed.path)[1].lower()
        content_type = (response.headers.get("Content-Type") or "").lower()
        return self._read_from_bytes(
            response.content,
            spec,
            source_label=url,
            suffix_hint=suffix,
            content_type=content_type,
        )

    def _read_from_bytes(
        self,
        data: bytes,
        spec: Dict[str, Any],
        source_label: str,
        suffix_hint: str = "",
        content_type: str = "",
    ) -> str:
        file_kind = self._infer_file_kind(suffix_hint, content_type)

        if file_kind == "text":
            body = self._read_text_bytes(data)
            return self._format_result(source_label, suffix_hint or "text", body)
        if file_kind == "markdown":
            body = self._read_text_bytes(data)
            return self._format_result(source_label, suffix_hint or "markdown", body)
        if file_kind == "json":
            body = self._read_json_bytes(data)
            return self._format_result(source_label, suffix_hint or "json", body)
        if file_kind == "csv":
            body = self._read_delimited_bytes(data, sep=spec.get("sep", ","), max_rows=int(spec.get("max_rows", self.max_rows) or self.max_rows))
            return self._format_result(source_label, suffix_hint or "csv", body)
        if file_kind == "tsv":
            body = self._read_delimited_bytes(data, sep="\t", max_rows=int(spec.get("max_rows", self.max_rows) or self.max_rows))
            return self._format_result(source_label, suffix_hint or "tsv", body)
        if file_kind == "html":
            body = self._read_html_bytes(data)
            return self._format_result(source_label, suffix_hint or "html", body)
        if file_kind == "pdf":
            body = self._read_pdf_bytes(data, spec)
            return self._format_result(source_label, suffix_hint or "pdf", body)
        if file_kind == "docx":
            body = self._read_docx_bytes(data)
            return self._format_result(source_label, suffix_hint or "docx", body)
        if file_kind == "xlsx":
            body = self._read_xlsx_bytes(data, spec)
            return self._format_result(source_label, suffix_hint or "xlsx", body)
        if file_kind == "image":
            body = self._read_image_bytes(data, spec)
            return self._format_result(source_label, suffix_hint or "image", body)

        unsupported = suffix_hint or (content_type.split(";")[0] if content_type else "unknown")
        if unsupported in {".doc", ".xls"}:
            raise ValueError(
                f"Unsupported legacy Office format: {unsupported}. "
                "This implementation supports .docx/.xlsx directly. "
                "For .doc/.xls you need an extra converter dependency or to save the file as modern Office XML format first."
            )
        raise ValueError(
            f"Unsupported file type: {unsupported}. "
            "Supported: txt/md/json/jsonl/csv/tsv/html/pdf/docx/xlsx/xlsm/jpg/jpeg/png/webp/bmp/gif/tiff"
        )

    # ------------------------------
    # Format readers
    # ------------------------------
    def _read_text_bytes(self, data: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")

    def _read_json_bytes(self, data: bytes) -> str:
        text = self._read_text_bytes(data)
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return text

    def _read_delimited_bytes(self, data: bytes, sep: str = ",", max_rows: int = 200) -> str:
        try:
            df = pd.read_csv(BytesIO(data), sep=sep)
            if max_rows > 0:
                df = df.head(max_rows)
            return df.to_csv(index=False, sep=sep)
        except Exception:
            return self._read_text_bytes(data)

    def _read_html_bytes(self, data: bytes) -> str:
        soup = BeautifulSoup(self._read_text_bytes(data), "lxml")
        return soup.get_text(separator="\n", strip=True)

    def _read_pdf_bytes(self, data: bytes, spec: Dict[str, Any]) -> str:
        target_page = spec.get("page")
        max_pages = int(spec.get("max_pages", self.max_pages) or self.max_pages)
        texts: List[str] = []

        def _normalize_page(page_value: Any) -> Optional[int]:
            if page_value is None:
                return None
            page_num = int(page_value)
            if page_num < 1:
                raise ValueError("PDF page number should be 1-based")
            return page_num - 1

        page_idx = _normalize_page(target_page)

        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                if page_idx is not None:
                    if page_idx >= len(pdf.pages):
                        raise ValueError(f"Invalid page number: {target_page}")
                    return pdf.pages[page_idx].extract_text() or ""

                for page in pdf.pages[:max_pages]:
                    text = page.extract_text() or ""
                    if text.strip():
                        texts.append(text)
                if texts:
                    return "\n\n".join(texts)
        except Exception:
            pass

        reader = PdfReader(BytesIO(data))
        if page_idx is not None:
            if page_idx >= len(reader.pages):
                raise ValueError(f"Invalid page number: {target_page}")
            return reader.pages[page_idx].extract_text() or ""

        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text)
        return "\n\n".join(texts)

    def _read_docx_bytes(self, data: bytes) -> str:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            xml_paths = [name for name in zf.namelist() if name == "word/document.xml" or name.startswith("word/header") or name.startswith("word/footer")]
            texts: List[str] = []
            for xml_path in xml_paths:
                root = ET.fromstring(zf.read(xml_path))
                for paragraph in root.findall(".//w:p", WORD_NS):
                    paragraph_text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)).strip()
                    if paragraph_text:
                        texts.append(paragraph_text)
            return "\n".join(texts)

    def _read_xlsx_bytes(self, data: bytes, spec: Dict[str, Any]) -> str:
        max_rows = int(spec.get("max_rows", self.max_rows) or self.max_rows)
        max_sheets = int(spec.get("max_sheets", self.max_sheets) or self.max_sheets)
        requested_sheet = spec.get("sheet")

        with zipfile.ZipFile(BytesIO(data)) as zf:
            shared_strings = self._parse_shared_strings(zf)
            sheets = self._parse_sheet_targets(zf)
            if not sheets:
                return ""

            selected: List[Tuple[str, str]] = []
            if requested_sheet is None:
                selected = sheets[:max_sheets]
            elif isinstance(requested_sheet, int):
                if requested_sheet < 1 or requested_sheet > len(sheets):
                    raise ValueError(f"Invalid sheet index: {requested_sheet}")
                selected = [sheets[requested_sheet - 1]]
            else:
                for sheet_name, target in sheets:
                    if sheet_name == str(requested_sheet):
                        selected = [(sheet_name, target)]
                        break
                if not selected:
                    raise ValueError(f"Sheet not found: {requested_sheet}")

            rendered: List[str] = []
            for sheet_name, target in selected:
                xml_bytes = zf.read(target)
                rows = self._parse_sheet_rows(xml_bytes, shared_strings, max_rows=max_rows)
                if not rows:
                    rendered.append(f"# Sheet: {sheet_name}\n[empty]")
                    continue
                rendered_rows = ["\t".join(row).rstrip() for row in rows]
                rendered.append(f"# Sheet: {sheet_name}\n" + "\n".join(rendered_rows))
            return "\n\n".join(rendered)

    def _read_image_bytes(self, data: bytes, spec: Dict[str, Any]) -> str:
        with Image.open(BytesIO(data)) as image:
            info_lines = [
                f"format: {image.format}",
                f"size: {image.width}x{image.height}",
                f"mode: {image.mode}",
            ]

            exif_lines: List[str] = []
            try:
                raw_exif = image.getexif()
                if raw_exif:
                    tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw_exif.items()}
                    for key in sorted(tag_map.keys())[:20]:
                        exif_lines.append(f"{key}: {tag_map[key]}")
            except Exception:
                pass

            should_try_ocr = bool(spec.get("ocr", self.enable_image_ocr))
            ocr_text = ""
            ocr_note = ""
            if should_try_ocr:
                try:
                    import pytesseract  # type: ignore

                    ocr_text = pytesseract.image_to_string(image)
                except Exception as exc:
                    ocr_note = f"OCR unavailable: {exc}"
            else:
                ocr_note = "OCR disabled. This implementation can still return image metadata."

        parts = ["[image metadata]", *info_lines]
        if exif_lines:
            parts.append("[image exif]")
            parts.extend(exif_lines)
        if ocr_text.strip():
            parts.append("[image ocr text]")
            parts.append(ocr_text.strip())
        elif ocr_note:
            parts.append(ocr_note)
        return "\n".join(parts)

    # ------------------------------
    # Helpers for XLSX / OOXML parsing
    # ------------------------------
    def _parse_shared_strings(self, zf: zipfile.ZipFile) -> List[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        values: List[str] = []
        for si in root.findall("main:si", SSML_NS):
            texts = [t.text or "" for t in si.findall(".//main:t", SSML_NS)]
            values.append("".join(texts))
        return values

    def _parse_sheet_targets(self, zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
        workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map: Dict[str, str] = {}
        for rel in rels_xml.findall("pr:Relationship", PKG_REL_NS):
            rel_id = rel.attrib.get("Id")
            target = rel.attrib.get("Target")
            if rel_id and target:
                rel_map[rel_id] = self._normalize_zip_target(target)

        out: List[Tuple[str, str]] = []
        for sheet in workbook_xml.findall("main:sheets/main:sheet", SSML_NS):
            name = sheet.attrib.get("name", "Sheet")
            rel_id = sheet.attrib.get(f"{{{SSML_NS['r']}}}id")
            if rel_id and rel_id in rel_map:
                out.append((name, rel_map[rel_id]))
        return out

    def _parse_sheet_rows(self, sheet_xml: bytes, shared_strings: Sequence[str], max_rows: int) -> List[List[str]]:
        root = ET.fromstring(sheet_xml)
        rows: List[List[str]] = []
        for row in root.findall(".//main:sheetData/main:row", SSML_NS):
            row_cells: List[str] = []
            current_col = 0
            for cell in row.findall("main:c", SSML_NS):
                ref = cell.attrib.get("r", "")
                expected_col = self._column_index_from_ref(ref)
                while current_col < expected_col:
                    row_cells.append("")
                    current_col += 1
                row_cells.append(self._extract_xlsx_cell_value(cell, shared_strings))
                current_col += 1
            rows.append(row_cells)
            if max_rows > 0 and len(rows) >= max_rows:
                break
        return rows

    def _extract_xlsx_cell_value(self, cell: ET.Element, shared_strings: Sequence[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return "".join(t.text or "" for t in cell.findall(".//main:t", SSML_NS))

        value_node = cell.find("main:v", SSML_NS)
        formula_node = cell.find("main:f", SSML_NS)
        raw_value = value_node.text if value_node is not None and value_node.text is not None else ""

        if cell_type == "s":
            try:
                return shared_strings[int(raw_value)]
            except Exception:
                return raw_value
        if cell_type == "b":
            return "TRUE" if raw_value == "1" else "FALSE"
        if formula_node is not None and not raw_value:
            return f"={formula_node.text or ''}"
        return raw_value

    # ------------------------------
    # Generic utilities
    # ------------------------------
    def _resolve_local_path(self, path: str) -> str:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return self._norm_path(expanded)
        return self._norm_path(os.path.join(os.getcwd(), expanded))

    def _check_path_allowed(self, path: str) -> None:
        if not self.allowed_roots:
            return
        norm_path = self._norm_path(path)
        for root in self.allowed_roots:
            if norm_path == root or norm_path.startswith(root + os.sep):
                return
        raise PermissionError(f"Path not allowed: {path}")

    def _infer_file_kind(self, suffix: str, content_type: str = "") -> str:
        suffix = (suffix or "").lower()
        content_type = (content_type or "").lower()

        if suffix in {".txt", ".log", ".py", ".yaml", ".yml", ".xml", ".rst"}:
            return "text"
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".json", ".jsonl"}:
            return "json"
        if suffix == ".csv":
            return "csv"
        if suffix == ".tsv":
            return "tsv"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".pdf" or "pdf" in content_type:
            return "pdf"
        if suffix == ".docx":
            return "docx"
        if suffix in {".xlsx", ".xlsm"}:
            return "xlsx"
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"} or content_type.startswith("image/"):
            return "image"
        if content_type.startswith("text/"):
            return "text"
        if "json" in content_type:
            return "json"
        if "html" in content_type:
            return "html"
        return "unknown"

    def _format_result(self, source_label: str, file_type: str, body: str) -> str:
        cleaned_body = (body or "").strip()
        if not cleaned_body:
            cleaned_body = "[empty]"
        return f"source: {source_label}\ntype: {file_type}\ncontent:\n{cleaned_body}"

    def _truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _norm_path(self, path: str) -> str:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))

    def _normalize_zip_target(self, target: str) -> str:
        normalized = target.replace("\\", "/")
        if normalized.startswith("/"):
            normalized = normalized[1:]
        if normalized.startswith("xl/"):
            return normalized
        return posixpath.normpath(posixpath.join("xl", normalized))

    def _column_index_from_ref(self, ref: str) -> int:
        letters = ""
        for ch in ref:
            if ch.isalpha():
                letters += ch.upper()
            else:
                break
        if not letters:
            return 0
        index = 0
        for ch in letters:
            index = index * 26 + (ord(ch) - ord("A") + 1)
        return max(index - 1, 0)
