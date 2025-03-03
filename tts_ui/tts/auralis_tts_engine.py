from auralis import TTS, TTSRequest, TTSOutput, setup_logger
from gradio import File, Files, Slider
import numpy as np
import torch
import torchaudio
from tts_ui.utils import (
    optimize_text_input,
    extract_text_from_epub,
    text_from_file,
    convert_audio_to_int16,
    torchaudio_stretch,
    get_hash_from_data,
)
import tempfile
from tts_ui.utils.doc_processor import DocumentProcessor
import hashlib
import time
from pathlib import Path
import os
import shutil
import soundfile as sf


class AuralisTTSEngine:
    def __init__(self):
        self.session_uid = None
        # Generate a random hex string to identify this process
        self._create_new_session_ui()

        # create a unique temp file location for this inst
        tmp_dir_base = Path(tempfile.mkdtemp()).resolve()
        tmp_dir_base.mkdir(exist_ok=True)
        self.tmp_dir_base = tmp_dir_base

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

        self.log_messages: str | None = None

        self.doc_processor = DocumentProcessor()

    def _create_new_session_ui(self):
        if self.session_uid is not None:
            return self.session_uid

        # Generate a random UID to identify this session
        _gen_uid: str = os.urandom(4).hex()
        self.session_uid: str = _gen_uid

        return _gen_uid

    def generate_audio_from_text(self, request: TTSRequest, speed: float = 1.0):
        """
        The main text processing function that can handle text of any size and convert them into a long audio file.
        This only converts and returns the path to the final audio file. It does not return the audio data
        """

        self.log_messages: str = ""
        combined_audio_path: list[str] = []

        if not request.speaker_files:
            self.log_messages += "Please provide at least one reference audio!\n"
            return combined_audio_path

        print(f"Using sample voice from {request.speaker_files}")

        full_text = str(request.text)

        # Note: This works without issue (but we could make some improvements)
        chunks_to_process: list[str] = optimize_text_input(
            text=full_text, chunk_size=1000, chunk_overlap=0
        )
        print(f"Created {len(chunks_to_process)} chunks")

        # Note: This could be done in parallel, but it works in most cases without issues (albeit very slow)
        processed_chunk_paths, failed_chunks = self._process_text_in_chunks(
            chunks_to_process=chunks_to_process, tts_req=request, speed=speed
        )

        if failed_chunks:
            self.log_messages = (
                f"⚠️ Completed with {len(failed_chunks)} failed chunks "
                f"({len(processed_chunk_paths)} succeeded)\n"
            )
        else:
            self.logger.info(f"Converted {len(processed_chunk_paths)} chunks to audio")

        # Combine the saved audio chunks into one using pydub (good for WAV files)
        try:
            # Note: This mostly works, but audio format becomes an important factor. We can improve this
            # Note: .wav files cannot be larger than 4gb. Probably good to just make this into a mp3
            combined_audio_path = (
                self._combine_audio(processed_chunk_paths)
                if len(processed_chunk_paths)
                > 1  # Only process if there are more than one audio chunks
                else processed_chunk_paths
            )

            print(
                f"Reading the combined audio from {combined_audio_path} using Soundfile"
            )

            # Todo: This consumes a lot of memory.
            # Todo: Properly handle multiple outputs when the audio is too large
            # Read the exported audio file again
            # audio_data, sample_rate = sf.read(combined_audio_path[0])

            self.log_messages += "✅ Successfully Generated audio\n"
            # self.logger.info(self.log_messages)

        except Exception as e:
            self.log_messages += f"❌ Failed to write chunks: {e}"
            self.logger.error(self.log_messages)

        finally:
            print("Returning the final audio file")
            # return the final audio
            return combined_audio_path

    def _process_text_in_chunks(
        self,
        chunks_to_process: list[str],
        tts_req: TTSRequest,
        speed: float = 1.0,
        max_retry=5,
    ):
        # failed text chunks
        failed_chunks: list[(int, str)] = []
        # successful audio chunks
        processed_chunks: list[str] = []
        processed_count = 0

        base_work_dir = self.tmp_dir_base.joinpath("_working/").resolve()
        base_work_dir.mkdir(exist_ok=True)

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

                        final_audio_data: np.ndarray = audio.array

                        # Might be slower to process her chunk, but better than having to load gb of audio at once
                        if speed != 1:
                            final_audio_data = torchaudio_stretch(
                                audio_data=audio.array,
                                sample_rate=audio.sample_rate,
                                speed_factor=speed,
                            )
                            self.logger.info(
                                f"Adjusting the final audio speed value by {speed}"
                            )

                    # Create a unique chunk name that can be sorted alphanumerically
                    audio_hash: str = get_hash_from_data(final_audio_data.tobytes())
                    chunk_save_path: str = (
                        base_work_dir.joinpath(f"{audio_hash}_chunk_{idx:03d}.wav")
                        .resolve()
                        .as_posix()
                    )

                    self.logger.info(
                        f"Writing the converted chunk to {chunk_save_path}"
                    )

                    # Todo: maybe consider saving as mp3?
                    # Write the converted audio chunk to the disk so we can process them later
                    sf.write(
                        file=chunk_save_path,
                        # Convert float16 to int16
                        data=convert_audio_to_int16(final_audio_data),
                        samplerate=audio.sample_rate,
                    )
                    processed_chunks.append(chunk_save_path)

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

    def _clean_temp_work_path(self):
        self.logger.info("Performing clean up task")
        # remove and make an empty temp folder
        shutil.rmtree(self.tmp_dir_base)
        self.tmp_dir_base.mkdir(exist_ok=True)

    def _combine_audio(self, chunk_paths: list[str]) -> str:
        max_size_gb = 3.8

        max_size: float = max_size_gb * 1024**3  # Convert GB to bytes

        combined_output_path: list[str] = []

        # Verify consistent audio format
        with sf.SoundFile(chunk_paths[0]) as ref_file:
            samplerate = ref_file.samplerate
            channels = ref_file.channels
            subtype = ref_file.subtype
            format_check = (samplerate, channels, subtype)

            head_file_hash = get_hash_from_data(ref_file.name + self.session_uid)

        # Initialize output management
        _tmp_dir_base: Path = self.tmp_dir_base.joinpath("_store/")
        _tmp_dir_base.mkdir(exist_ok=True)

        for f in chunk_paths[1:]:
            with sf.SoundFile(f) as test_file:
                if (
                    test_file.samplerate,
                    test_file.channels,
                    test_file.subtype,
                ) != format_check:
                    raise ValueError(f"Format mismatch in {os.path.basename(f)}")

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

                        output_path: str = (
                            _tmp_dir_base.joinpath(
                                f"combined_{head_file_hash}_{output_index:02d}.wav"
                            )
                            .resolve()
                            .as_posix()
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
            # Remove the chunk file after processing it
            os.remove(file_path)

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
        request = TTSRequest(
            text=input_text,
            speaker_files=ref_audio_files,
            stream=False,
            enhance_speech=enhance_speech,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            language=language,
        )

        converted_audio_list = self.generate_audio_from_text(request, speed)

        self.logger.info(self.log_messages)
        # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result

        return converted_audio_list[0], self.log_messages

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
            input_text: str = ""
            match file_extension:
                case ".epub":
                    input_text = extract_text_from_epub(file_input.name)
                case ".txt" | ".md":
                    input_text = text_from_file(file_input.name)
                case _:
                    return (
                        [],
                        "Unsupported file format, it needs to be either .epub or .txt",
                    )

            request = TTSRequest(
                text=input_text,
                speaker_files=ref_audio_files_file,
                stream=False,
                enhance_speech=enhance_speech_file,
                temperature=temperature_file,
                top_p=top_p_file,
                top_k=top_k_file,
                repetition_penalty=repetition_penalty_file,
                language=language_file,
            )

            converted_audio_list = self.generate_audio_from_text(request, speed_file)

            self.logger.info(self.log_messages)
            # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result
            return converted_audio_list[0], self.log_messages
        else:
            return [], "No document file was provided!"

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
            output_path = self.tmp_dir_base.joinpath(f"mic_{hash}.wav")

            torch_audio: torch.Tensor = torch.from_numpy(mic_ref_audio[1].astype(float))
            try:
                torchaudio.save(
                    str(output_path), torch_audio.unsqueeze(0), mic_ref_audio[0]
                )

                request = TTSRequest(
                    text=input_text_mic,
                    speaker_files=output_path,
                    stream=False,
                    enhance_speech=enhance_speech_mic,
                    temperature=temperature_mic,
                    top_p=top_p_mic,
                    top_k=top_k_mic,
                    repetition_penalty=repetition_penalty_mic,
                    language=language_mic,
                )

                converted_audio_list = self.generate_audio_from_text(request, speed_mic)

                self.logger.info(self.log_messages)
                # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result
                return converted_audio_list[0], self.log_messages

            except Exception as e:
                self.logger.error(f"Error saving audio file: {e}")
                return [], f"Error saving audio file: {e}"
        else:
            return [], "Please record an audio!"
