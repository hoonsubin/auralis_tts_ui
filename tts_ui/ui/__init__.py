import gradio as gr
from tts_ui.tts.auralis_tts_engine import AuralisTTSEngine

supported_langs: list[str] = [
    "auto",
    "en",
    "ko",
    "ja",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "hi",
]

default_values: dict[str, float] = {
    "playback_speed": 1.0,
    "temperature": 0.6,
    "top_p": 0.65,
    "top_k": 30,
    "repetition_penalty": 4.5,
}


def build_gradio_ui() -> gr.Blocks:
    """Builds and launches the Gradio UI for Auralis."""

    with gr.Blocks(title="GPT-TTS UI - Clone any voice", theme="soft") as ui:
        gr.Markdown(
            """
          # Text-to-Speech Interface
          
          Convert text to speech with advanced voice cloning and enhancement.

          Powered by Auralis 🌌 made by Hoon
          """
        )

        tts_engine: AuralisTTSEngine = AuralisTTSEngine().load_model()

        async def _handle_tts_text_input(*args):
            return await tts_engine.process_text_and_generate(*args)

        async def _handle_tts_file_input(*args):
            return await tts_engine.process_file_and_generate(*args)

        async def _handle_tts_mic_input(*args):
            return await tts_engine.process_mic_and_generate(*args)

        with gr.Tab("Text to Speech"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Text(
                        label="Enter Text Here",
                        placeholder="Write the text you want to convert...",
                    )
                    ref_audio_files = gr.Files(
                        label="Reference Audio Files", file_types=["audio"]
                    )
                    with gr.Accordion("Advanced settings", open=True):
                        speed = gr.Slider(
                            label="Playback speed",
                            minimum=0.5,
                            maximum=2.0,
                            value=default_values["playback_speed"],
                            step=0.1,
                        )
                        enhance_speech = gr.Checkbox(
                            label="Enhance Reference Speech", value=True
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["temperature"],
                            step=0.05,
                        )
                        top_p = gr.Slider(
                            label="Top P",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["top_p"],
                            step=0.05,
                        )
                        top_k = gr.Slider(
                            label="Top K",
                            minimum=0,
                            maximum=100,
                            value=default_values["top_k"],
                            step=10,
                        )
                        repetition_penalty = gr.Slider(
                            label="Repetition penalty",
                            minimum=1.0,
                            maximum=10.0,
                            value=default_values["repetition_penalty"],
                            step=0.5,
                        )
                        language = gr.Dropdown(
                            label="Target Language",
                            choices=supported_langs,
                            value="auto",
                        )
                    generate_button = gr.Button("Generate Speech")
                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    log_output = gr.Text(label="Log Output")

            generate_button.click(
                fn=_handle_tts_text_input,
                concurrency_limit=3,
                max_batch_size=4,
                inputs=[
                    input_text,
                    ref_audio_files,
                    speed,
                    enhance_speech,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    language,
                ],
                outputs=[audio_output, log_output],
            )

        with gr.Tab("File to Speech"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Text / Ebook File", file_types=[".txt", ".md", ".epub"]
                    )
                    ref_audio_files_file = gr.Files(
                        label="Reference Audio Files", file_types=["audio"]
                    )
                    with gr.Accordion("Advanced settings", open=True):
                        speed_file = gr.Slider(
                            label="Playback speed",
                            minimum=0.5,
                            maximum=2.0,
                            value=default_values["playback_speed"],
                            step=0.1,
                        )
                        enhance_speech_file = gr.Checkbox(
                            label="Enhance Reference Speech", value=True
                        )
                        temperature_file = gr.Slider(
                            label="Temperature",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["temperature"],
                            step=0.05,
                        )
                        top_p_file = gr.Slider(
                            label="Top P",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["top_p"],
                            step=0.05,
                        )
                        top_k_file = gr.Slider(
                            label="Top K",
                            minimum=0,
                            maximum=100,
                            value=default_values["top_k"],
                            step=10,
                        )
                        repetition_penalty_file = gr.Slider(
                            label="Repetition penalty",
                            minimum=1.0,
                            maximum=10.0,
                            value=default_values["repetition_penalty"],
                            step=0.5,
                        )
                        language_file = gr.Dropdown(
                            label="Target Language",
                            choices=supported_langs,
                            value="auto",
                        )
                    generate_button_file = gr.Button("Generate Speech from File")
                with gr.Column():
                    audio_output_file = gr.Audio(
                        label="Generated Audio", type="filepath"
                    )
                    log_output_file = gr.Text(label="Log Output")

            generate_button_file.click(
                fn=_handle_tts_file_input,
                concurrency_limit=3,
                max_batch_size=4,
                inputs=[
                    file_input,
                    ref_audio_files_file,
                    speed_file,
                    enhance_speech_file,
                    temperature_file,
                    top_p_file,
                    top_k_file,
                    repetition_penalty_file,
                    language_file,
                ],
                outputs=[audio_output_file, log_output_file],
            )

        with gr.Tab("Clone With Microphone"):
            with gr.Row():
                with gr.Column():
                    input_text_mic = gr.Text(
                        label="Enter Text Here",
                        placeholder="Write the text you want to convert...",
                    )
                    mic_ref_audio = gr.Audio(
                        label="Record Reference Audio", sources=["microphone"]
                    )

                    with gr.Accordion("Advanced settings", open=True):
                        speed_mic = gr.Slider(
                            label="Playback speed",
                            minimum=0.5,
                            maximum=2.0,
                            value=default_values["playback_speed"],
                            step=0.1,
                        )
                        enhance_speech_mic = gr.Checkbox(
                            label="Enhance Reference Speech", value=True
                        )
                        temperature_mic = gr.Slider(
                            label="Temperature",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["temperature"],
                            step=0.05,
                        )
                        top_p_mic = gr.Slider(
                            label="Top P",
                            minimum=0.5,
                            maximum=1.0,
                            value=default_values["top_p"],
                            step=0.05,
                        )
                        top_k_mic = gr.Slider(
                            label="Top K",
                            minimum=0,
                            maximum=100,
                            value=default_values["top_k"],
                            step=10,
                        )
                        repetition_penalty_mic = gr.Slider(
                            label="Repetition penalty",
                            minimum=1.0,
                            maximum=10.0,
                            value=default_values["repetition_penalty"],
                            step=0.5,
                        )
                        language_mic = gr.Dropdown(
                            label="Target Language",
                            choices=supported_langs,
                            value="auto",
                        )
                    generate_button_mic = gr.Button("Generate Speech")
                with gr.Column():
                    audio_output_mic = gr.Audio(
                        label="Generated Audio", type="filepath"
                    )
                    log_output_mic = gr.Text(label="Log Output")

            generate_button_mic.click(
                fn=_handle_tts_mic_input,
                concurrency_limit=3,
                max_batch_size=4,
                inputs=[
                    input_text_mic,
                    mic_ref_audio,
                    speed_mic,
                    enhance_speech_mic,
                    temperature_mic,
                    top_p_mic,
                    top_k_mic,
                    repetition_penalty_mic,
                    language_mic,
                ],
                outputs=[audio_output_mic, log_output_mic],
            )

    return ui
