import base64
import uuid
import shutil
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from yakinori import Yakinori
import regex as re
import numpy as np
import jaconv
import bunkai

# Create a temporary directory to store short-named files
tmp_dir = Path("/tmp/auralis")
tmp_dir.mkdir(exist_ok=True)


def shorten_filename(original_path: str) -> str:
    """Copies the given file to a temporary directory with a shorter, random filename."""
    ext: str = Path(original_path).suffix
    short_name: str = "file_" + uuid.uuid4().hex[:8] + ext
    short_path: Path = tmp_dir / short_name
    shutil.copyfile(original_path, short_path)
    return str(short_path)


def extract_text_from_epub(epub_path: str, output_path=None) -> str:
    """
    Extracts text from an EPUB file and optionally saves it to a text file.

    Args:
        epub_path (str): Path to the EPUB file
        output_path (str, optional): Path where to save the text file

    Returns:
        str: The extracted text
    """
    # Load the book
    book: epub.EpubBook = epub.read_epub(epub_path)

    # List to hold extracted text
    chapters: list[str] = []

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
    full_text: str = "\n\n".join(chapters)

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


def is_japanese(text) -> bool:
    # Regex patterns for Hiragana, Katakana, and common Kanji/CJK unified blocks
    hiragana = r"[\p{Hiragana}]"
    katakana = r"[\p{Katakana}]"

    # Check for Hiragana or Katakana (unique to Japanese)
    return bool(re.search(hiragana, text) or re.search(katakana, text))


def preprocess_japanese_text(text: str) -> str:
    alpha2kana: str = jaconv.alphabet2kana(text)
    normalized_jp: str = jaconv.normalize(alpha2kana)

    yakinori = Yakinori()

    splitter = bunkai.Bunkai()

    sentences: np.Iterator[str] = splitter(normalized_jp)

    final: str = ""

    for sentence in sentences:
        parsed_list: list[str] = yakinori.get_parsed_list(sentence)
        final += yakinori.get_hiragana_sentence(parsed_list, is_hatsuon=True)

    return final


def convert_audio(data: np.ndarray) -> np.ndarray:
    """Convert any float format to proper 16-bit PCM"""
    if data.dtype in [np.float16, np.float32, np.float64]:
        # Normalize first to [-1, 1] range
        data = data.astype(np.float32) / np.max(np.abs(data))
        # Scale to 16-bit int range
        data = (data * 32767).astype(np.int16)
    return data


def split_text_into_chunks(
    text: str, chunk_size: int = 2000, chunk_overlap: int = 100
) -> list[str]:
    """
    Split text into chunks respecting byte limits and natural boundaries.
    This function also automatically converts Japanese Kanji into Kana for better readability.
    """

    text_to_process = text

    text_separators: list[str] = [
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

    if is_japanese(text_to_process):
        text_to_process = preprocess_japanese_text(text_to_process)

    splitter = RecursiveCharacterTextSplitter(
        separators=text_separators,
        chunk_size=chunk_size,  # Optimized for TTS context windows
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_text(text)
