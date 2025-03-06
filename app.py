from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine
from tts_ui.ui import build_gradio_ui
import asyncio
import threading
import signal

# Create main event loop
loop = asyncio.new_event_loop()


def run_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()


# Start loop in dedicated thread
threading.Thread(target=run_loop, daemon=True).start()


async def main():
    tts_engine = AuralisTTSEngine()
    ui = build_gradio_ui(tts_engine)

    try:
        queued_ui = ui.queue(
            default_concurrency_limit=4, max_size=10, status_update_rate=15
        )
        await queued_ui.launch(
            debug=True,
            prevent_thread_lock=True,  # Critical parameter
            show_error=True,
        )
        await asyncio.Future()  # Keep server running
    finally:
        if hasattr(tts_engine, "destroy_process_group"):
            tts_engine.destroy_process_group()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Add proper signal handling
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, loop.stop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
