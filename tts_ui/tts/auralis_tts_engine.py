from logging import Logger
from typing import Generator, Optional
from auralis import (
    TTS,
    TTSRequest,
    TTSOutput,
    setup_logger,
)
from gradio import File, Files
import torch
from utils import (
    split_text_into_chunks,
    temp_dir,
    extract_text_from_epub,
    text_from_file,
)
from utils.doc_processor import DocumentProcessor
import hashlib
import torchaudio
import time
from pathlib import Path

# Loading the TTS engine first and assign it to the class.
# This looks ugly, but it works
logger: Logger = setup_logger(__file__)

tts = TTS()
model_path = "AstraMindAI/xttsv2"  # change this if you have a different model
gpt_model = "AstraMindAI/xtts2-gpt"

try:
    tts: TTS = tts.from_pretrained(
        scheduler_max_concurrency=12,
        model_name_or_path=model_path,
        gpt_model=gpt_model,
        device="cpu",
        use_mps=True,
    )
    logger.info(f"Successfully loaded model {model_path}")
except Exception as e:
    error_msg = f"Failed to load model: {e}."
    logger.error(error_msg)
    raise Exception(error_msg)


class AuralisTTSEngine:
    def __init__(self):
        self.logger: Logger = logger
        self.tts: TTS = tts
        self.model_path: str = model_path
        self.gpt_model: str = gpt_model
        self.tmp_dir: Path = temp_dir
        self.doc_processor: DocumentProcessor = DocumentProcessor

    def process_text_and_generate(
        self,
        input_text: str,
        ref_audio_files: str | list[str] | bytes | list[bytes],
        speed: float,
        enhance_speech: bool,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
        language: str = "auto",
        *args,
    ):
        """Process text and generate audio."""
        log_messages: str = ""
        if not ref_audio_files:
            log_messages += "Please provide at least one reference audio!\n"
            return None, log_messages

        # clone voices from all file paths (shorten them)
        base64_voices: str | list[str] | bytes | list[bytes] = ref_audio_files[:5]

        # todo: clean the `input_text` if it's Japanese

        request = TTSRequest(
            text=input_text,
            speaker_files=base64_voices,
            stream=False,
            enhance_speech=enhance_speech,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            language=language,
        )

        try:
            with torch.no_grad():
                output: Generator[TTSOutput, None, None] | TTSOutput = (
                    self.tts.generate_speech(request)
                )

                if output:
                    if speed != 1:
                        output.change_speed(speed)
                    log_messages += f"✅ Successfully Generated audio\n"
                    # return the sample rate and the audio file as a byte array
                    return (output.sample_rate, output.array), log_messages

                else:
                    log_messages += "❌ No output was generated. Check that the model was correctly loaded\n"
                    return None, log_messages
        except Exception as e:
            self.logger.error(f"Error: {e}")
            log_messages += f"❌ An Error occured: {e}\n"
            return None, log_messages

    def process_large_text(
        self,
        input_full_text: str,
        ref_audio_files: str | list[str] | bytes | list[bytes],
        speed: float,
        enhance_speech: bool,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
        language: str = "auto",
    ):
        """Process text in chunks and combine results"""
        log_messages: str = ""

        if not ref_audio_files:
            log_messages += "Please provide at least one reference audio!\n"
            return None, log_messages

        base64_voices: str | list[str] | bytes | list[bytes] = ref_audio_files[:5]

        chunks: list[str] = split_text_into_chunks(input_full_text)
        print(f"Created {len(chunks)} chunks")

        audio_segments: list[TTSOutput] = []
        for idx, chunk in enumerate(chunks):
            request = TTSRequest(
                text=chunk,
                speaker_files=base64_voices,
                stream=False,
                enhance_speech=enhance_speech,
                temperature=temperature,
                speed=speed,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                language=language,
            )

            try:
                with torch.no_grad():
                    audio = self.tts.generate_speech(request)
                    audio_segments.append(audio)
                    self.logger.info(f"Processed {idx + 1} chunks out of {len(chunks)}")

            except Exception as e:
                log_messages += f"❌ Chunk processing failed: {e}\n"
                return None, log_messages

        if len(audio_segments) <= 0:
            log_messages += f"❌ Chunk processing failed. Chunk size: {len(chunks)}\n"
            return None, log_messages

        combined_output: TTSOutput = TTSOutput.combine_outputs(audio_segments)

        log_messages += f"✅ Successfully Generated audio\n"
        # return combined_output
        return (combined_output.sample_rate, combined_output.array), log_messages

    def process_file_and_generate(
        self,
        file_input: File,
        ref_audio_files_file: Files,
        speed_file,
        enhance_speech_file,
        temperature_file,
        top_p_file,
        top_k_file,
        repetition_penalty_file,
        language_file,
    ):
        # todo: refactor this to use the document processor object
        if file_input:
            print(file_input)
            file_extension: str = Path(file_input.name).suffix
            if file_extension == ".epub":
                input_text: str = extract_text_from_epub(file_input.name)
            elif file_extension == ".txt" or file_extension == ".md":
                input_text = text_from_file(file_input.name)
            else:
                return (
                    None,
                    "Unsupported file format, it needs to be either .epub or .txt",
                )

            return self.process_large_text(
                input_text,
                ref_audio_files_file,
                speed_file,
                enhance_speech_file,
                temperature_file,
                top_p_file,
                top_k_file,
                repetition_penalty_file,
                language_file,
            )
        else:
            return None, "Please provide an .epub or .txt file!"

    def process_mic_and_generate(
        self,
        input_text_mic,
        mic_ref_audio,
        speed_mic,
        enhance_speech_mic,
        temperature_mic,
        top_p_mic,
        top_k_mic,
        repetition_penalty_mic,
        language_mic,
    ):
        if mic_ref_audio:
            data: bytes = str(time.time()).encode("utf-8")
            hash: str = hashlib.sha1(data).hexdigest()[:10]
            output_path = self.temp_dir / (f"mic_{hash}.wav")

            torch_audio: torch.Tensor = torch.from_numpy(mic_ref_audio[1].astype(float))
            try:
                torchaudio.save(
                    str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0]
                )
                return self.process_text_and_generate(
                    input_text_mic,
                    [Path(output_path)],
                    speed_mic,
                    enhance_speech_mic,
                    temperature_mic,
                    top_p_mic,
                    top_k_mic,
                    repetition_penalty_mic,
                    language_mic,
                )
            except Exception as e:
                self.logger.error(f"Error saving audio file: {e}")
                return None, f"Error saving audio file: {e}"
        else:
            return None, "Please record an audio!"
