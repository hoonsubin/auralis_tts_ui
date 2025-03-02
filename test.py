import glob
import random
import soundfile as sf
import os
from tts_ui.utils import convert_audio_to_int16


def process_all_local_docs():
    from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine

    tts_engine = AuralisTTSEngine()
    audio_save_path = "./data/output"
    base_data_path = "./data"

    all_texts: list[str] = glob.glob(f"{base_data_path}/docs/*.txt", recursive=False)
    all_markdown: list[str] = glob.glob(f"{base_data_path}/docs/*.md", recursive=False)

    all_voices: list[str] = glob.glob(f"{base_data_path}/voices/*.wav", recursive=False)

    all_texts.extend(all_markdown)

    print(f"Found {len(all_texts)} files to process")

    for index, text_to_process in enumerate(all_texts):
        random_voice_path: str = os.path.abspath(random.choice(all_voices))
        abs_text_path: str = os.path.abspath(text_to_process)

        print(f"Using {random_voice_path} as sample")

        # ref_voice, sample_rate = sf.read(random_voice_path)

        with open(abs_text_path, "r", encoding="utf-8") as f:
            text_content: str = f.read()
            doc_name: str = os.path.splitext(os.path.basename(abs_text_path))[0]

            print(f"Converting {abs_text_path}")

            (sample_rate, audio_data), log = tts_engine._process_large_text(
                input_full_text=text_content,
                ref_audio=[random_voice_path],
                speed=1.1,
                enhance_speech=True,
                temperature=0.6,
                top_p=0.65,
                top_k=30,
                repetition_penalty=4.5,
                language="auto",
            )
            print(log)

        save_path: str = f"{audio_save_path}/{doc_name}.wav"
        sf.write(
            file=save_path,
            # Convert float16 to int16
            data=convert_audio_to_int16(audio_data),
            samplerate=sample_rate,
        )

        print(
            f"Finished processing {index + 1} files out of {len(all_texts)}.\nSaved to {save_path}."
        )


def main():
    # base_data_path = "./data"
    # all_texts = glob.glob(f"{base_data_path}/docs/" + "*.md", recursive=False)
    # all_voices: list[str] = glob.glob(f"{base_data_path}/voices/*.wav", recursive=False)

    # for doc_file in all_texts:
    #     print(doc_file)

    # for voice_file in all_voices:
    #     print(voice_file)

    process_all_local_docs()


if __name__ == "__main__":
    # asyncio.run(main())
    main()
