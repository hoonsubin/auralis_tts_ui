from typing import Optional
from auralis import TTS, TTSRequest, TTSOutput, setup_logger
from gradio import File, Files, Slider
import numpy as np
import torch
import torchaudio
from tts_ui.utils import (
    calculate_byte_size,
    split_text_into_chunks,
    extract_text_from_epub,
    text_from_file,
    convert_audio_to_int16,
    torchaudio_stretch,
)
import tempfile
from tts_ui.utils.doc_processor import DocumentProcessor
import hashlib
import time
from pathlib import Path
import os
import shutil
import soundfile as sf
import wave
import subprocess


class AuralisTTSEngine:
    def __init__(self):
        # create a unique temp file location for this inst
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_dir.mkdir(exist_ok=True)

        # Loading the TTS engine first and assign it to the class.
        logger = setup_logger(__file__)

        tts: TTS = TTS()
        model_path = "AstraMindAI/xttsv2"  # change this if you have a different model
        gpt_model = "AstraMindAI/xtts2-gpt"

        try:
            cuda_available: bool = torch.cuda.is_available()
            if not cuda_available:
                logger.warning("CUDA is not available for this platform")
                os.environ["VLLM_NO_GPU"] = "1"
                os.environ["TRITON_CPU_ONLY"] = "1"
            else:
                logger.info("CUDA is available for this platform")

            tts = tts.from_pretrained(
                model_name_or_path=model_path,
                gpt_model=gpt_model,
                enforce_eager=False,
                max_concurrency=6,  # need to adjust this based on the host machine or it can cause `AsyncEngineDeadError`
            )

            logger.info(f"Successfully loaded model {model_path}")
        except Exception as e:
            error_msg = f"Failed to load model: {e}."
            logger.error(error_msg)
            raise Exception(error_msg)

        self.logger = logger
        self.tts: TTS = tts
        self.model_path: str = model_path
        self.gpt_model: str = gpt_model
        self.tmp_dir: Path = tmp_dir
        self.doc_processor = DocumentProcessor()

        # Generate a random hex string to identify this process
        self._create_new_session_ui()

    def _create_new_session_ui(self):
        self.session_uid: str = os.urandom(4).hex()
        return self.session_uid

    def generate_audio_from_large_text(
        self,
        input_full_text: str,
        ref_audio: str | list[str] | bytes | list[bytes],
        speed: float,
        enhance_speech: bool,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
        language: str = "auto",
    ):
        """The main text processing function that can handle text of any size and convert them into a long audio file"""

        log_messages: str = ""

        if not ref_audio:
            log_messages += "Please provide at least one reference audio!\n"
            return None, log_messages

        self._create_new_session_ui()

        print(f"Using sample voice from {ref_audio}")

        # Note: This works without issue (but we could make some improvements)
        chunks_to_process: list[str] = split_text_into_chunks(
            text=input_full_text, chunk_size=2000, chunk_overlap=0
        )
        print(f"Created {len(chunks_to_process)} chunks")

        request = TTSRequest(
            text=input_full_text,
            speaker_files=ref_audio,
            stream=False,
            enhance_speech=enhance_speech,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            language=language,
        )

        # Note: This could be done in parallel, but it works in most cases without issues (albeit very slow)
        processed_chunk_paths, failed_chunks = self._process_text_in_chunks(
            chunks_to_process=chunks_to_process, tts_req=request
        )

        if failed_chunks:
            log_messages = (
                f"⚠️ Completed with {len(failed_chunks)} failed chunks "
                f"({len(processed_chunk_paths)} succeeded)\n"
            )
        else:
            self.logger.info(f"Converted {len(processed_chunk_paths)} chunks to audio")

        # Combine the saved audio chunks into one using pydub (good for WAV files)
        try:
            # Note: This mostly works, but audio format becomes an important factor. We can improve this
            # Note: .wav files cannot be larger than 4gb. Probably good to just make this into a mp3
            combined_audio_path: list = self._combine_audio(processed_chunk_paths)

            print(
                f"Reading the combined audio from {combined_audio_path} using Soundfile"
            )

            # Todo: This consumes a lot of memory.
            # Read the exported audio file again
            audio_data, sample_rate = sf.read(combined_audio_path[0])

            final_audio: np.ndarray = audio_data

            if speed != 1:
                final_audio = torchaudio_stretch(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    speed_factor=speed,
                )
                print(f"Adjusting the final audio speed value by {speed}")

            log_messages += "✅ Successfully Generated audio\n"
            self.logger.info(log_messages)

        except Exception as e:
            log_messages += f"❌ Failed to write chunks: {e}"
            self.logger.error(log_messages)

        finally:
            # Clean up all temporary files
            self._clean_temp_path()

            print("Returning the final audio file")
            # return the final audio
            return (
                sample_rate,
                convert_audio_to_int16(final_audio),
            ), log_messages

    def _process_text_in_chunks(
        self, chunks_to_process: list[str], tts_req: TTSRequest, max_retry=5
    ):
        # failed text chunks
        failed_chunks: list[(int, str)] = []
        # successful audio chunks
        processed_chunks: list[str] = []
        processed_count = 0

        # Todo: refactor this to be processed in parallel
        # Process the batch of chunks into audio
        for idx, chunk in enumerate(chunks_to_process):
            request = TTSRequest(
                text=chunk,
                speaker_files=tts_req.speaker_files,
                stream=tts_req.stream,
                enhance_speech=tts_req.enhance_speech,
                temperature=tts_req.temperature,
                top_p=tts_req.top_p,
                top_k=tts_req.top_k,
                repetition_penalty=tts_req.repetition_penalty,
                language=tts_req.language,
            )
            current_attempts = 0
            can_retry: bool = current_attempts <= max_retry

            while can_retry:
                try:
                    with torch.no_grad():
                        # self.logger.info(f"Processing {chunk}")
                        audio: TTSOutput = self.tts.generate_speech(request)

                    # Create a unique chunk name that can be sorted alphanumerically
                    temp_file_path: str = os.path.join(
                        self.tmp_dir, f"{self.session_uid}_chunk_{idx}.wav"
                    )

                    self.logger.info(f"Writing the converted chunk to {temp_file_path}")

                    # Todo: maybe consider saving as mp3?
                    # Write the converted audio chunk to the disk so we can process them later
                    sf.write(
                        file=temp_file_path,
                        # Convert float16 to int16
                        data=convert_audio_to_int16(audio.array),
                        samplerate=audio.sample_rate,
                    )
                    processed_chunks.append(temp_file_path)

                    processed_count += 1
                    self.logger.info(
                        f"Processed {idx + 1} chunks out of {len(chunks_to_process)}"
                    )

                    # Clean up GPU memory for every 10 chunks
                    if (idx + 1) % 10 == 0:
                        self.logger.info("Emptying GPU cache")
                        torch.cuda.empty_cache()  # If using GPU

                    # Break out of the while loop and continue with the next chunk
                    break

                except Exception as e:
                    # Retry the chunk process until the retry limit
                    if not can_retry:
                        # Add this chunk to the error list and move on to the next chunk
                        failed_chunks.append((idx, str(e)))
                        self.logger.warning(
                            f"⚠️ Exceeded max retries for chunk {idx + 1}"
                        )
                        break

                    # Retry the chunk after a waiting period
                    current_attempts += 1
                    wait_time = 2 ** (current_attempts - 1)  # Exponential backoff
                    self.logger.warning(
                        f"⚠️ Failed chunk {idx + 1}: {str(e)}.\nRetrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    # Better to be explicit
                    continue

        # If the entire process failed
        if processed_count == 0:
            raise Exception("❌ All chunk processing failed")

        return processed_chunks, failed_chunks

    def _clean_temp_path(self):
        self.logger.info("Performing clean up task")
        # remove and make an empty temp folder
        shutil.rmtree(self.tmp_dir)
        # self.tmp_dir.mkdir(exist_ok=True)

    def _combine_audio(self, chunk_paths: list[str], audio_chunk_size=8192) -> str:
        max_size_gb = 3.8

        max_size: float = max_size_gb * 1024**3  # Convert GB to bytes

        output_dir: str = os.path.abspath("./data/")

        combined_output_path: list[str] = []

        # Verify consistent audio format
        with sf.SoundFile(chunk_paths[0]) as ref_file:
            samplerate = ref_file.samplerate
            channels = ref_file.channels
            subtype = ref_file.subtype
            format_check = (samplerate, channels, subtype)

        for f in chunk_paths[1:]:
            with sf.SoundFile(f) as test_file:
                if (
                    test_file.samplerate,
                    test_file.channels,
                    test_file.subtype,
                ) != format_check:
                    raise ValueError(f"Format mismatch in {os.path.basename(f)}")

        # Initialize output management
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, "combined")
        output_index = 0
        current_out = None

        for file_path in chunk_paths:
            with sf.SoundFile(file_path) as infile:
                while True:  # Process in chunks
                    chunk = infile.read(1024 * channels)  # ~23ms chunks at 44.1kHz
                    if chunk.size == 0:
                        break

                    # Create new output file when needed
                    if not current_out or os.path.getsize(current_out.name) >= max_size:
                        if current_out:
                            current_out.close()
                        output_index += 1
                        output_path = (
                            f"{output_base}_{self.session_uid}_{output_index:03d}.wav"
                        )
                        current_out = sf.SoundFile(
                            output_path,
                            "w",
                            samplerate=samplerate,
                            channels=channels,
                            subtype=subtype,
                        )
                        combined_output_path.append(output_path)

                    current_out.write(chunk)

        if current_out:
            current_out.close()

        return combined_output_path

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

        input_size = calculate_byte_size(input_text)

        # use the chunking process if the text is too large
        if input_size > 45000:
            self.logger.info(
                f"Found {input_size} bytes of text. Switching to chunk mode."
            )
            # todo: this function has a couple of overlapping functions as normal processing. I need to optimize the code
            return self.generate_audio_from_large_text(
                input_full_text=input_text,
                ref_audio_files=ref_audio_files,
                speed=speed,
                enhance_speech=enhance_speech,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                language=language,
            )
        else:
            try:
                with torch.no_grad():
                    # clone voices from all file paths (shorten them)
                    base64_voices: str | list[str] | bytes | list[bytes] = (
                        ref_audio_files[:5]
                    )

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

                    output: TTSOutput = self.tts.generate_speech(request)

                    if output:
                        if speed != 1:
                            output.change_speed(speed)
                        log_messages += "✅ Successfully Generated audio\n"
                        # return the sample rate and the audio file as a byte array
                        return (
                            output.sample_rate,
                            convert_audio_to_int16(output.array),
                        ), log_messages

                    else:
                        log_messages += "❌ No output was generated. Check that the model was correctly loaded\n"
                        return None, log_messages
            except Exception as e:
                self.logger.error(f"Error: {e}")
                log_messages += f"❌ An Error occured: {e}\n"
                return None, log_messages

    def process_file_and_generate(
        self,
        file_input: File,
        ref_audio_files_file: Files,
        speed_file: Slider,
        enhance_speech_file,
        temperature_file,
        top_p_file,
        top_k_file,
        repetition_penalty_file,
        language_file,
    ):
        # todo: refactor this to use the document processor object
        if file_input:
            file_extension: str = Path(file_input.name).suffix

            match file_extension:
                case ".epub":
                    input_text: str = extract_text_from_epub(file_input.name)
                case ".txt" | ".md":
                    input_text = text_from_file(file_input.name)
                case _:
                    return (
                        None,
                        "Unsupported file format, it needs to be either .epub or .txt",
                    )

            return self.generate_audio_from_large_text(
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
            output_path = self.tmp_dir / (f"mic_{hash}.wav")

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
