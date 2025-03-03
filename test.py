import gc
import time
import random
import soundfile as sf
import os
from pathlib import Path


def process_all_local_docs():
    from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine

    tts_engine = AuralisTTSEngine()

    base_path = Path("./data/")
    audio_save_path = base_path.joinpath("output/").resolve()

    doc_base_path = base_path.joinpath("docs/")

    all_texts = list(doc_base_path.rglob("*.txt")) + list(doc_base_path.rglob("*.md"))
    all_voices = list(base_path.joinpath("voices").glob("*.wav"))

    print(f"Found {len(all_texts)} doc files and {len(all_voices)} to process")

    all_texts.sort()
    for index, text_to_process in enumerate(all_texts):
        current_doc_path = text_to_process.resolve()

        random_voice_path = all_voices[
            random.randrange(0, len(all_voices) - 1)
        ].resolve()

        print(f"Using {random_voice_path} as sample")

        with open(current_doc_path, "r", encoding="utf-8") as f:
            text_content: str = f.read()

        doc_name: str = os.path.splitext(os.path.basename(str(current_doc_path)))[0]

        print(f"Converting {current_doc_path}")

        (sample_rate, audio_data), log = tts_engine.generate_audio_from_large_text(
            input_full_text=text_content,
            ref_audio=[str(random_voice_path)],
            speed=1.0,
            enhance_speech=True,
            temperature=0.6,
            top_p=0.65,
            top_k=30,
            repetition_penalty=4.5,
            language="auto",
        )
        print(log)

        audio_file_name = f"{doc_name}.mp3"

        # Note: The program halts here as it takes a lot of memory to perform this
        save_path: Path = audio_save_path.joinpath(audio_file_name)

        sf.write(
            file=str(save_path),
            data=audio_data,
            samplerate=sample_rate,
        )

        print(
            f"Finished processing {index + 1} files out of {len(all_texts)}.\nSaved to {save_path}."
        )
        print("Cleaning up task")
        # Add manual garbage collection
        del audio_data
        gc.collect()
        print("Waiting for 5 seconds to cool off the engine...")
        time.sleep(5)


def main():
    process_all_local_docs()


if __name__ == "__main__":
    # asyncio.run(main())
    main()
