import asyncio
from gradio import Blocks
import nest_asyncio

nest_asyncio.apply()


async def main():
    # from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine
    from tts_ui.ui import build_gradio_ui

    # tts_engine: AuralisTTSEngine = AuralisTTSEngine()

    try:
        # tts_engine = tts_engine.load_model()
        # tts_engine.loop = asyncio.get_running_loop()
        ui: Blocks = build_gradio_ui()
        queued_ui: Blocks = ui.queue(
            default_concurrency_limit=4, max_size=10, status_update_rate=15
        )
        await queued_ui.launch(
            debug=True,
            prevent_thread_lock=True,  # Critical parameter
            show_error=True,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
