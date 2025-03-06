from auralis import TTS, TTSRequest, TTSOutput, setup_logger, AudioPreprocessingConfig
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
    print_memory_summary,
)
import tempfile
from tts_ui.utils.doc_processor import DocumentProcessor
import hashlib
import time
from pathlib import Path
import os
import shutil
import soundfile as sf
import gc
import asyncio


class AuralisTTSEngine:
    def __init__(self):
        """
        Initialize the TTS engine with necessary configurations.

        This function sets up:
        - A unique session identifier for tracking
        - Temporary directories for processing files
        - The text-to-speech model using specified parameters

        Parameters:
            None (all settings are hardcoded or generated internally)

        Returns:
            None
        """
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
                max_concurrency=3,
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

    def _create_new_session_ui(self) -> str:
        """
        Generate a unique session identifier for the current process.

        This function creates a random hexadecimal string to identify the session.
        If a session UID already exists, it will be returned instead of generating a new one.

        Returns:
            str: Unique session identifier
        """
        if self.session_uid is not None:
            return self.session_uid

        # Generate a random UID to identify this session
        _gen_uid: str = os.urandom(4).hex()
        self.session_uid: str = _gen_uid

        return _gen_uid

    @torch.no_grad()
    async def _process_text_in_chunks(
        self,
        chunks_to_process: list[str],
        tts_req: TTSRequest,
        speed: float = 1.0,
        max_retry=5,
    ) -> tuple[list[str], list[Exception]]:
        """
        Process text chunks asynchronously and convert them to audio files.

        This method splits the input text into manageable chunks, processes each chunk
        using the TTS engine, adjusts the audio speed if specified, and saves the output
        as separate audio files. Failed chunks are collected for potential retrying.

        Parameters:
            chunks_to_process (list[str]): List of text chunks to process
            tts_req (TTSRequest): Configuration parameters for TTS generation
            speed (float, optional): Speed adjustment factor for audio (default: 1.0)
            max_retry (int, optional): Number of retry attempts per chunk (not used yet)

        Returns:
            tuple[list[str], list[Exception]]: A tuple containing:
                - List of successfully processed audio file paths
                - List of exceptions encountered during processing
        """
        base_work_dir: Path = self.tmp_dir_base.joinpath("_working/").resolve()
        base_work_dir.mkdir(exist_ok=True)

        # Todo: might want to pull this out and make it an argument
        audio_config = AudioPreprocessingConfig(
            normalize=True,
            trim_silence=True,
            enhance_speech=tts_req.enhance_speech,
            enhance_amount=1.5 if tts_req.enhance_speech else 1,
        )

        self.processed_items: int = 0

        async def _process_and_save_chunk(req: TTSRequest, chunk_id: int) -> str:
            try:
                audio: TTSOutput = await self.tts.generate_speech_async(req)
                final_audio_data: np.ndarray = audio.array

                # Might be slower to process her chunk, but better than having to load gb of combined audio at once
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
                    base_work_dir.joinpath(
                        f"chunk_{chunk_id:03d}_{self.session_uid}_{audio_hash}.wav"
                    )
                    .resolve()
                    .as_posix()
                )

                self.logger.info(f"Writing the converted chunk to {chunk_save_path}")

                # Todo: maybe consider saving as mp3?
                # Write the converted audio chunk to the disk so we can process them later
                sf.write(
                    file=chunk_save_path,
                    # Convert float16 to int16
                    data=convert_audio_to_int16(final_audio_data),
                    samplerate=audio.sample_rate,
                )

                self.processed_items += 1

                # Clean up GPU memory for every 10 chunks
                if (self.processed_items) % 10 == 0:
                    self.logger.info("Emptying GPU cache")
                    torch.cuda.empty_cache()  # If using GPU
                    torch.cuda.reset_peak_memory_stats()

                print_memory_summary()
                return chunk_save_path
            except Exception as e:
                self.logger.warning(
                    f"Encountered an error while processing chunk {chunk_id}: {e}"
                )
                return e

        all_reqs = [
            TTSRequest(
                text=chunk,
                speaker_files=tts_req.speaker_files,
                stream=tts_req.stream,
                enhance_speech=tts_req.enhance_speech,
                audio_config=audio_config,
                temperature=tts_req.temperature,
                top_p=tts_req.top_p,
                top_k=tts_req.top_k,
                repetition_penalty=tts_req.repetition_penalty,
                language=tts_req.language,
            )
            for chunk in chunks_to_process
        ]

        coroutines = [
            _process_and_save_chunk(req, idx) for idx, req in enumerate(all_reqs)
        ]
        outputs = await asyncio.gather(*coroutines, return_exceptions=True)
        self.logger.info(
            f"Finished converting all {len(chunks_to_process)} text chunks to audio"
        )

        # failed text chunks
        failed_chunks: list[Exception] = []
        # successful audio chunks
        processed_chunks: list[str] = []

        for chunk_output in outputs:
            if isinstance(chunk_output, Exception):
                print(f"Found exception: {chunk_output}")
                # Todo: implement a retry mechanism for failed processes
                failed_chunks.append(chunk_output)
            else:
                processed_chunks.append(chunk_output)

        if torch.cuda.is_available():
            self.logger.info("Emptying GPU cache")
            torch.cuda.empty_cache()

        # If the entire process failed
        if len(processed_chunks) == 0:
            raise Exception("❌ All chunk processing failed")

        processed_chunks.sort()

        return processed_chunks, failed_chunks

    def _clean_temp_work_path(self):
        """
        Clean up temporary files created during the TTS process.

        This function removes all temporary directories and recreates them.
        Useful for preventing disk overflow from too many temporary files.

        Parameters:
            None

        Returns:
            None
        """
        self.logger.info("Performing clean up task")
        # remove and make an empty temp folder
        shutil.rmtree(self.tmp_dir_base)
        self.tmp_dir_base.mkdir(exist_ok=True)

    def _combine_audio(self, chunk_paths: list[str]) -> str:
        """
        Combine multiple audio chunks into a single output file.

        This method ensures all audio files have consistent format and sampling rate,
        then concatenates them into one or more output files based on maximum size
        constraints (default max size is 3.8 GB).

        Parameters:
            chunk_paths (list[str]): List of paths to individual audio chunks

        Returns:
            str: Path to the final combined audio file(s). If multiple files are created,
                returns a list with all output paths.

        Raises:
            ValueError: If input files have inconsistent formats
        """
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

    async def _generate_audio_from_text(
        self, request: TTSRequest, speed: float = 1.0, chunk_size=600
    ):
        """
        Main function for converting text to audio.

        This method handles large volumes of text by breaking it into chunks,
        processing each asynchronously, and combining the results into final
        output files.

        Parameters:
            request (TTSRequest): Configuration parameters for TTS generation
            speed (float, optional): Speed adjustment factor for audio (default: 1.0)
            chunk_size (int, optional): Size of text chunks in characters (default: 600)

        Returns:
            list[str]: List of paths to the generated audio files

        Raises:
            Exception: If all processing attempts fail
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
            text=full_text, chunk_size=chunk_size, chunk_overlap=0
        )
        print(f"Created {len(chunks_to_process)} chunks")

        # Note: This could be done in parallel, but it works in most cases without issues (albeit very slow)
        processed_chunk_paths, failed_chunks = await self._process_text_in_chunks(
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
            self.log_messages += "✅ Successfully Generated audio\n"
            # self.logger.info(self.log_messages)

        except Exception as e:
            self.log_messages += f"❌ Failed to write chunks: {e}"
            self.logger.error(self.log_messages)

        finally:
            print("Returning the final audio file")
            # Add manual garbage collection
            gc.collect()
            # return the final audio
            return combined_audio_path

    async def process_text_and_generate(
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
    ):
        """
        Main interface for converting text to audio using specified parameters.

        This function validates inputs and triggers the TTS generation process.

        Parameters:
            input_text (str): Text to be converted
            ref_audio_files (str | list[str] | bytes | list[bytes]): Reference audio files
                for speaker voice cloning
            speed (float): Speed multiplier for generated audio (0.5-2.0)
            enhance_speech (bool): Flag for speech enhancement
            temperature (float): Temperature parameter for text generation
            top_p (float): Top-p sampling parameter
            top_k (float): Top-k sampling parameter
            repetition_penalty (float): Repetition penalty for text generation
            language (str, optional): Language of the input text (default: "auto")

        Returns:
            tuple[str, str]: A tuple containing:
                - Path to the generated audio file(s)
                - Log messages indicating success or failure

        Raises:
            Exception: If processing fails completely
        """

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

        converted_audio_list = await self._generate_audio_from_text(request, speed)

        self.logger.info(self.log_messages)

        # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result
        return converted_audio_list[0], self.log_messages

    async def process_file_and_generate(
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
        """
        Process text from a file and generate audio.

        This function reads text content from supported file types (.epub, .txt, .md)
        and converts it into speech using the specified parameters.

        Parameters:
            file_input (File): Input document file
            ref_audio_files_file (Files): Reference audio files for speaker voice cloning
            speed_file (Slider): Speed multiplier for generated audio (0.5-2.0)
            enhance_speech_file (bool): Flag for speech enhancement
            temperature_file (float): Temperature parameter for text generation
            top_p_file (float): Top-p sampling parameter
            top_k_file (float): Top-k sampling parameter
            repetition_penalty_file (float): Repetition penalty for text generation
            language_file (str): Language of the input text

        Returns:
            tuple[str, str]: A tuple containing:
                - Path to the generated audio file(s)
                - Log messages indicating success or failure

        Raises:
            Exception: If processing fails completely
        """
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

            converted_audio_list = await self._generate_audio_from_text(
                request, speed_file
            )

            self.logger.info(self.log_messages)

            # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result
            return converted_audio_list[0], self.log_messages
        else:
            return [], "No document file was provided!"

    async def process_mic_and_generate(
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
        """
        Process text from microphone input and generate audio.

        This function converts spoken text into speech using the specified parameters.

        Parameters:
            input_text_mic (str): Text transcribed from microphone input
            mic_ref_audio (tuple[int, bytes]): Audio data captured from microphone
                (sample rate and audio bytes)
            speed_mic (float): Speed multiplier for generated audio (0.5-2.0)
            enhance_speech_mic (bool): Flag for speech enhancement
            temperature_mic (float): Temperature parameter for text generation
            top_p_mic (float): Top-p sampling parameter
            top_k_mic (float): Top-k sampling parameter
            repetition_penalty_mic (float): Repetition penalty for text generation
            language_mic (str): Language of the input text

        Returns:
            tuple[str, str]: A tuple containing:
                - Path to the generated audio file(s)
                - Log messages indicating success or failure

        Raises:
            Exception: If processing fails completely
        """
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

                converted_audio_list = await self._generate_audio_from_text(
                    request, speed_mic
                )

                self.logger.info(self.log_messages)

                # Todo: Refactor the Gradio UI code to allow multiple audio outputs. Right now, we only return the first result
                return converted_audio_list[0], self.log_messages

            except Exception as e:
                self.logger.error(f"Error saving audio file: {e}")
                return [], f"Error saving audio file: {e}"
        else:
            return [], "Please record an audio!"
