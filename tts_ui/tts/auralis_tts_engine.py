from auralis import TTS, TTSRequest, TTSOutput, setup_logger
from gradio import File, Files, Slider
import torch
import torchaudio
from tts_ui.utils import (
    calculate_byte_size,
    chunk_generator,
    extract_text_from_epub,
    text_from_file,
    convert_audio_to_int16,
)
import tempfile
from tts_ui.utils.doc_processor import DocumentProcessor
import hashlib
import time
from pathlib import Path
import os
import shutil
import soundfile as sf
from pydub import AudioSegment

# Loading the TTS engine first and assign it to the class.
# This looks ugly, but it works
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
        max_concurrency=4,
    )

    logger.info(f"Successfully loaded model {model_path}")
except Exception as e:
    error_msg = f"Failed to load model: {e}."
    logger.error(error_msg)
    raise Exception(error_msg)


class AuralisTTSEngine:
    def __init__(self):
        # create a unique temp file location for this inst
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_dir.mkdir(exist_ok=True)

        self.logger = logger
        self.tts: TTS = tts
        self.model_path: str = model_path
        self.gpt_model: str = gpt_model
        self.tmp_dir: Path = tmp_dir
        self.doc_processor = DocumentProcessor()

    def _process_large_text(
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
        """Process text in chunks and combine results"""
        log_messages: str = ""

        if not ref_audio:
            log_messages += "Please provide at least one reference audio!\n"
            return None, log_messages

        base64_voices: str | list[str] | bytes | list[bytes] = ref_audio[:5]

        # failed text chunks
        failed_chunks: list[(int, str)] = []
        # successful audio chunks
        temp_files: list[str] = []
        processed_count = 0

        # Todo: refactor this to be processed in parallel
        # Process the batch of chunks into audio
        for idx, chunk in enumerate(chunk_generator(input_full_text)):
            request = TTSRequest(
                text=chunk,
                speaker_files=base64_voices,
                stream=False,
                enhance_speech=enhance_speech,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                language=language,
            )

            max_retry = 5
            current_attempts = 0
            can_retry: bool = current_attempts <= max_retry

            while can_retry:
                try:
                    with torch.no_grad():
                        # self.logger.info(f"Processing {chunk}")
                        audio: TTSOutput = self.tts.generate_speech(request)

                        # Save the current chunk to disk for processing it later
                        temp_file_path: str = os.path.join(
                            self.tmp_dir, f"chunk_{idx:04d}.wav"
                        )

                        sf.write(
                            file=temp_file_path,
                            # Convert float16 to int16
                            data=convert_audio_to_int16(audio.array),
                            samplerate=audio.sample_rate,
                        )
                        temp_files.append(temp_file_path)

                        processed_count += 1
                        self.logger.info(f"Processed {idx + 1} chunks")

                        # Clean up GPU memory for every 10 chunks
                        if (idx + 1) % 10 == 0:
                            print("Emptying GPU cache")
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
            log_messages += "❌ All chunk processing failed"
            self._clean_temp_path()
            return None, log_messages

        # Report partial success if some chunks failed
        if failed_chunks:
            log_messages += (
                f"⚠️ Completed with {len(failed_chunks)} failed chunks "
                f"({processed_count} succeeded)\n"
            )

        # Combine the saved audio chunks into one using pydub (good for WAV files)
        try:
            self.logger.info(f"Combining {len(temp_files)} audio chunks...")
            combined_audio_path = os.path.join(self.tmp_dir, "combined_output.wav")

            # Create an empty audio segment
            combined: AudioSegment = AudioSegment.empty()

            # Add each chunk one by one (streaming from disk)
            for file_path in temp_files:
                chunk_audio = AudioSegment.from_wav(file_path)
                combined += chunk_audio

                # Immediately remove the file after adding to free disk space
                os.remove(file_path)

            # Export the combined audio to a temporary file
            combined.export(combined_audio_path, format="wav")

            # Todo: can we optimize this further? Loading what we just saved does't look nice
            # Read the exported audio file again
            audio_data, sample_rate = sf.read(combined_audio_path)

            final_audio: TTSOutput = TTSOutput(
                array=audio_data, sample_rate=sample_rate
            )

            if speed != 1:
                # final_audio = torchaudio_stretch(
                #     audio_data=final_audio.array,
                #     sample_rate=final_audio.sample_rate,
                #     speed_factor=speed,
                # )
                final_audio = final_audio.change_speed(speed)

            log_messages += "✅ Successfully Generated audio\n"
            self.logger.info(log_messages)

        except Exception as e:
            log_messages += f"❌ Failed to write chunks: {e}"
            self.logger.error(log_messages)

        finally:
            # Clean up all temporary files
            self._clean_temp_path()

            # return the final audio
            return (
                final_audio.sample_rate,
                convert_audio_to_int16(final_audio.array),
            ), log_messages

    def _clean_temp_path(self):
        # remove and make an empty temp folder
        shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)

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
            return self._process_large_text(
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

            return self._process_large_text(
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
