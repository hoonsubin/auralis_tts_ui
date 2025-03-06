import random
import shutil
import os
from pathlib import Path
from auralis import TTSRequest
import asyncio
import nest_asyncio

nest_asyncio.apply()


async def process_all_local_docs():
    from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine

    tts_engine = AuralisTTSEngine()

    base_path = Path("./data/")
    audio_save_path = base_path.joinpath("output/").resolve()
    audio_save_path.mkdir(exist_ok=True)

    doc_base_path: Path = base_path.joinpath("docs/")

    all_texts: list[Path] = list(doc_base_path.rglob("*.txt")) + list(
        doc_base_path.rglob("*.md")
    )
    all_voices = list(base_path.joinpath("voices").glob("*.wav"))

    print(f"Found {len(all_texts)} doc files and {len(all_voices)} voices to process")

    all_texts.sort()
    for index, text_to_process in enumerate(all_texts):
        current_doc_path: Path = text_to_process.resolve()

        random_voice_path: Path = all_voices[
            random.randrange(0, len(all_voices) - 1)
        ].resolve()

        print(f"Using {random_voice_path} as sample")

        with open(current_doc_path, "r", encoding="utf-8") as f:
            text_content: str = f.read()

        doc_name: str = os.path.splitext(os.path.basename(str(current_doc_path)))[0]

        print(f"Converting {current_doc_path}")

        request = TTSRequest(
            text=text_content,
            speaker_files=[str(random_voice_path)],
            stream=False,
            enhance_speech=True,
            temperature=0.6,
            top_p=0.65,
            top_k=20,
            repetition_penalty=4.5,
            language="auto",
        )

        converted_audio_list = await tts_engine._generate_audio_from_text(
            request=request, speed=1.1
        )

        print(tts_engine.log_messages)

        audio_file_name = f"{doc_name}.wav"

        # Note: The program halts here as it takes a lot of memory to perform this
        save_path: Path = audio_save_path.joinpath(audio_file_name)

        for audio_output in converted_audio_list:
            shutil.copyfile(audio_output, save_path)

        print(
            f"Finished processing {index + 1} files out of {len(all_texts)}.\nSaved to {save_path}."
        )

        print("Cleaning up task")


async def main():
    await process_all_local_docs()


if __name__ == "__main__":
    asyncio.run(main())
