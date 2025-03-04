import base64
import uuid
import shutil
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import regex as re
import numpy as np
import jaconv
import bunkai
import torch
import torchaudio
import hashlib


def get_hash_from_data(data: bytes | str, first_chars: int = 8) -> str:
    data_to_hash: bytes = data if not isinstance(data, str) else data.encode("utf-8")

    # Create a hash object (using SHA-256 for this example)
    hash_object: hashlib.HASH = hashlib.sha256(data_to_hash)

    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()[:first_chars]


def shorten_filename(original_path: str, tmp_dir: Path = Path("/tmp/uploads")) -> str:
    """
    Copies the given file to a temporary directory with a shorter, random filename.

    Args:
        original_path (str): Path to the original file that will be kept in the temp location
        tmp_dir (Path): The base path that will be used as the temp location. Defaults to `tmp/uploads`

    Returns:
        str: The short file path saved in a temporary location
    """

    file_path = Path(original_path)

    tmp_dir.mkdir(exist_ok=True)
    base_name: str = file_path.name
    ext: str = file_path.suffix
    short_name: str = f"file_{hash(base_name)}_{uuid.uuid4().hex[:8]}{ext}"
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
    """Calculate UTF-8 encoded byte size of provided text"""
    return len(text.encode("utf-8"))


def torchaudio_stretch(
    audio_data: np.ndarray, sample_rate: int, speed_factor: float
) -> np.ndarray:
    if speed_factor == 1.0:
        return audio_data
    audio_tensor = torch.from_numpy(audio_data.astype(np.float32))

    # If audio is mono, add batch dimension
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    effects = [
        ["tempo", str(speed_factor)],
        ["rate", str(sample_rate)],  # Ensure sample rate remains the same
    ]

    # Apply effects
    # Note: depends on `sox libsox-dev``
    modified_waveform, modified_sample_rate = (
        torchaudio.sox_effects.apply_effects_tensor(
            tensor=audio_tensor, sample_rate=sample_rate, effects=effects
        )
    )

    return modified_waveform.T.numpy()


def is_japanese(text) -> bool:
    # Regex patterns for Hiragana, Katakana, and common Kanji/CJK unified blocks
    hiragana = r"[\p{Hiragana}]"
    katakana = r"[\p{Katakana}]"

    # Check for Hiragana or Katakana (unique to Japanese)
    return bool(re.search(hiragana, text) or re.search(katakana, text))


def preprocess_japanese_text(text: str) -> str:
    removed_special_char = (
        text.replace("♡", "")
        .replace("♥", "")
        .replace("❤️", "")
        .replace("︎❤︎", "")
        .replace("゛", "")
        .replace("\n─", "")
        .replace("―", "")
        .replace("─", "")
        .replace("」", " ")
        .replace("「", " ")
        .replace("【", " ")
        .replace("】", " ")
        .replace("『", " ")
        .replace("』", " ")
    )
    normalized_jp: str = jaconv.normalize(removed_special_char)
    alpha2kana: str = jaconv.alphabet2kana(normalized_jp)

    return alpha2kana


def remove_empty_item(dirty_list: list):
    return list(filter(None, dirty_list))


def convert_audio_to_int16(data: np.ndarray) -> np.ndarray:
    """Convert any float format to proper 16-bit PCM"""
    if data.dtype in [np.float16, np.float32, np.float64]:
        # Normalize first to [-1, 1] range
        data = data.astype(np.float32) / np.max(np.abs(data))
        # Scale to 16-bit int range
        data = (data * 32767).astype(np.int16)
    return data


def optimize_text_input(
    text: str, chunk_size: int = 608, chunk_overlap: int = 5
) -> list[str]:
    """
    Split text into chunks respecting byte limits and natural boundaries.
    This function also automatically converts Japanese Kanji into Kana for better readability.
    """

    text_to_process: str = text

    # Based on Auralis TTS model
    max_bytes = 49149

    if is_japanese(text_to_process):
        text_to_process = preprocess_japanese_text(text_to_process)

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

    optimized_list: list[str] = []

    if len(text_to_process) > chunk_size:
        # todo: optimize this function so it stores batches locally in the temp folder instead of loading everything in memory
        # todo: implement a tokenized context batch size manager using https://huggingface.co/AstraMindAI/xtts2-gpt/blob/main/tokenizer.py
        langchain_splitter = RecursiveCharacterTextSplitter(
            separators=text_separators,
            chunk_size=chunk_size,  # Optimized for TTS context windows
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        split_chunks: list[str] = langchain_splitter.split_text(text_to_process)

        for current_text in split_chunks:
            # add the text if it's shorter than the max chunk size
            # Todo: this has to be further optimized since we flipflop between bytes and char sizes
            if calculate_byte_size(current_text) < max_bytes:
                optimized_list.append(current_text)
            else:
                local_chunk: list[str] = []
                print(f"Found a large chunk: {current_text}")
                # further split the chunk
                if is_japanese(current_text):
                    splitter = bunkai.Bunkai()
                    for local_batch in splitter(current_text):
                        local_chunk.append(local_batch)
                else:
                    for local_batch in remove_empty_item(
                        current_text.split("\n", chunk_overlap)
                    ):
                        local_chunk.append(local_batch)

                optimized_list.extend(local_chunk)
    else:
        optimized_list.append(text_to_process)

    return optimized_list
