# Auralis TTS UI

This is a simple Gradio UI application for using the amazing [Autalis](https://github.com/astramind-ai/Auralis) by Astramind.

## What does this do?

This project started from the Gradio example included in the original project, but I took the liberty to restructure the project and add a better large text chunking system so the app can handle large files (including entire books) from the UI.
Just keep in mind that it might be slow based on our setup.

## Quick Start

This project uses uv for package management. You can technically use pip, but I highly recommend uv.

```bash
uv sync

uv pip install -e .

uv run app.py
```

I've tested this on Linux (Ubuntu) and WSL2.
You may need to install system dependencies like `make` and `gcc`, to name a few.

There are compatible issues with macOS (especially the M-chips) related to `triton` or `vllm`.
You may have to build them from source or find a different workaround.
I'll be working on a solution, but because the modern LLM development environment is quite primitive, there is a lot of workarounds required to make things work on other devices.

Also note that if you're uploading a Japanese text, you want to run `python -m unidic download` from the project root.
