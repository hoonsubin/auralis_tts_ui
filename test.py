import gc
import glob
from pathlib import Path
import random
import soundfile as sf
import os


def process_all_local_docs():
    from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine

    tts_engine = AuralisTTSEngine()
    base_path = os.path.abspath("./data")
    audio_save_path = os.path.join(base_path, "output/")

    all_texts = glob.glob(base_path + "/docs/*.txt") + glob.glob(
        base_path + "/docs/*.md"
    )
    all_voices = glob.glob("./data/voices/*.wav", recursive=False)

    print(f"Found {len(all_texts)} files to process")

    all_texts.sort()
    for index, text_to_process in enumerate(all_texts):
        random_voice_path: str = os.path.abspath(
            all_voices[random.randrange(0, len(all_voices) - 1)]
        )
        abs_text_path: str = os.path.abspath(text_to_process)

        print(f"Using {random_voice_path} as sample")

        with open(abs_text_path, "r", encoding="utf-8") as f:
            text_content: str = f.read()
            doc_name: str = os.path.splitext(os.path.basename(abs_text_path))[0]

            print(f"Converting {abs_text_path}")

            (sample_rate, audio_data), log = tts_engine.generate_audio_from_large_text(
                input_full_text=text_content,
                ref_audio=[random_voice_path],
                speed=1.0,
                enhance_speech=True,
                temperature=0.6,
                top_p=0.65,
                top_k=30,
                repetition_penalty=4.5,
                language="auto",
            )
            print(log)

        save_path: str = f"{audio_save_path}/{doc_name}.mp3"
        sf.write(
            file=save_path,
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


def main():
    process_all_local_docs()


if __name__ == "__main__":
    # asyncio.run(main())
    main()
