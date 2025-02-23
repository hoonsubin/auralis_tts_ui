import markdown
import pdfplumber
from pathlib import Path
from tts_ui.utils import split_text_into_chunks, extract_text_from_epub, text_from_file


class DocumentProcessor:
    def __init__(self, max_word_chunk_size=4000):
        self.max_word_chunk_size: int = max_word_chunk_size  # Characters per chunk

    def process_doc(self, file_path: Path) -> list[str]:
        # get the file extension from the path
        ext: str = file_path.name.split(".")[-1].lower()

        # call the processor based on the file extension
        processors = {
            "pdf": self._process_pdf,
            "epub": self._process_epub,
            "md": self._process_markdown,
            "txt": self._process_text,
        }

        if not ext:
            raise Exception(f"No file found in {file_path}")

        return processors[ext](file_path)

    def _process_pdf(self, file_path: str) -> list[str]:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return self._chunk_text(text)

    def _process_epub(self, file_path: str) -> list[str]:
        text = extract_text_from_epub(file_path)
        return self._chunk_text(text)

    def _process_markdown(self, file_path: str) -> list[str]:
        with open(file_path, "r") as f:
            md_text: str = f.read()
        return self._chunk_text(markdown.markdown(md_text))

    def _process_text(self, file_path: str) -> list[str]:
        text = text_from_file(file_path)
        return self._chunk_text(text)

    def _chunk_text(self, text: str) -> list[str]:
        return split_text_into_chunks(text, self.max_word_chunk_size)
