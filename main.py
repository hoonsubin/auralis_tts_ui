import asyncio
from tts.auralis_tts_engine import AuralisTTSEngine
from ui import build_gradio_ui
import time
import asyncio
from utils import extract_text_from_epub
from auralis import TTS, TTSRequest, TTSOutput


def process_ebook(epub_path: str):
    text = extract_text_from_epub(epub_path)

    speaker_file = "data/sample-jp-1.wav"

    # Initialize the engine, you can experiment with the scheduler_max_concurrency parameter to optimize the performance
    tts: TTS = TTS(scheduler_max_concurrency=12).from_pretrained(
        model_name_or_path="AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt"
    )
    req = TTSRequest(
        text=text,
        language="auto",
        temperature=0.75,
        repetition_penalty=6.5,
        speaker_files=[speaker_file],
        stream=True,
    )

    start_time: float = time.time()

    # Execute requests in a generator to get audio instantly
    result_generator = tts.generate_speech(req)
    out_list = []

    for out in result_generator:
        out_list.append(out)
        # Play the audio
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

    # Save the audio to a file
    TTSOutput.combine_outputs(out_list).save("data/your_book.wav")


def main():
    tts_engine = AuralisTTSEngine()
    ui = build_gradio_ui(tts_engine)
    ui.launch(debug=True)


if __name__ == "__main__":
    # asyncio.run(main())
    main()
