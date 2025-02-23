import base64
import uuid
import shutil
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create a temporary directory to store short-named files
temp_dir = Path("/tmp/auralis")
temp_dir.mkdir(exist_ok=True)


def shorten_filename(original_path: str) -> str:
    """Copies the given file to a temporary directory with a shorter, random filename."""
    ext: str = Path(original_path).suffix
    short_name: str = "file_" + uuid.uuid4().hex[:8] + ext
    short_path: Path = temp_dir / short_name
    shutil.copyfile(original_path, short_path)
    return str(short_path)


def extract_text_from_epub(epub_path: str, output_path=None):
    """
    Extracts text from an EPUB file and optionally saves it to a text file.

    Args:
        epub_path (str): Path to the EPUB file
        output_path (str, optional): Path where to save the text file

    Returns:
        str: The extracted text
    """
    # Load the book
    book = epub.read_epub(epub_path)

    # List to hold extracted text
    chapters = []

    # Extract text from each chapter
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Get HTML content
            html_content = item.get_content().decode("utf-8")

            # Use BeautifulSoup to extract text
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text: str = soup.get_text()

            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            chapters.append(text)

    # Join all chapters
    full_text = "\n\n".join(chapters)

    # Save text if output path is specified
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    return full_text.replace("»", '"').replace("«", '"')


def text_from_file(txt_file_path: str) -> str:
    # Shorten filename before reading
    txt_short_path: str = shorten_filename(txt_file_path)
    with open(txt_short_path, "r") as f:
        text: str = f.read()
    return text


def clone_voice(audio_path: str) -> str:
    """Clone a voice from an audio path."""
    # Shorten filename before reading
    audio_short_path: str = shorten_filename(audio_path)
    with open(audio_short_path, "rb") as f:
        audio_data: str = base64.b64encode(f.read()).decode("utf-8")
    return audio_data


def calculate_byte_size(text: str) -> int:
    """Calculate UTF-8 encoded byte size of text"""
    return len(text.encode("utf-8"))


def split_text_into_chunks(
    text: str, chunk_size: int = 800, chunk_overlap: int = 10
) -> list[str]:
    """Split text into chunks respecting byte limits and natural boundaries"""

    japanese_separators: list[str] = [
        "\n\n",
        "\n",
        "。",
        "．",
        "？",
        "！",
        "?",
        "!",
        ",",
        "、",
        "，",
        "」",
        "』",
        "\u3002",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        separators=japanese_separators,
        chunk_size=chunk_size,  # Optimized for TTS context windows
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_text(text)
