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

        match ext:
            case "pdf":
                return self._process_pdf(file_path)
            case "epub":
                return self._process_epub(file_path)
            case "md":
                return self._process_markdown(file_path)
            case "txt":
                return self._process_text(file_path)
            case _:
                raise Exception(f"File {file_path} is not supported")

    def _process_pdf(self, file_path: str) -> list[str]:
        text: str = ""
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
