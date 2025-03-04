from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine
from tts_ui.ui import build_gradio_ui
import asyncio


async def main():
    tts_engine = AuralisTTSEngine()
    ui = await build_gradio_ui(tts_engine)
    ui.launch(debug=True)


if __name__ == "__main__":
    asyncio.run(main())
