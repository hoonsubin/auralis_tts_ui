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

uv run main.py
```
