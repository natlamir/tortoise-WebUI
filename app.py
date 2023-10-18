import os
import torch
import gradio as gr
import torchaudio
import time
import numpy as np
from datetime import datetime
from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, load_voices

VOICE_OPTIONS = []

def inference(
    text,
    script,
    voice,
    voice_b,
    seed,
    split_by_newline,
):
    if text is None or text.strip() == "":
        with open(script.name) as f:
            text = f.read()
        if text.strip() == "":
            raise gr.Error("Please provide either text or script file with content.")

    if split_by_newline == "Yes":
        texts = list(filter(lambda x: x.strip() != "", text.split("\n")))
    else:
        texts = split_and_recombine_text(text)

    voices = [voice]
    if voice_b != "disabled":
        voices.append(voice_b)

    if len(voices) == 1:
        voice_samples, conditioning_latents = load_voice(voice)
    else:
        voice_samples, conditioning_latents = load_voices(voices)

    start_time = time.time()

    gen, dbg_state = tts.tts_with_preset(text=text, k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset='fast', use_deterministic_seed=None, return_deterministic_state=True, cvvp_amount=.0)

    sep_segment = gen.squeeze(0).squeeze(0).data.cpu().numpy()
    return 24000, np.round(sep_segment * 32767).astype(np.int16)

def main():
    title = "Tortoise TTS 🐢"
    description = """
    A text-to-speech system which powers lot of organizations in Speech synthesis domain.
    <br/>
    a model with strong multi-voice capabilities, highly realistic prosody and intonation.
    <br/>
    for faster inference, use the 'ultra_fast' preset and duplicate space if you don't want to wait in a queue.
    <br/>
    """
    
    for root, dirs, files in os.walk("tortoise/voices"):
        for folder in dirs:
            VOICE_OPTIONS.append(folder)
    
    text = gr.Textbox(
        lines=4,
        label="Text (Provide either text, or upload a newline separated text file below):",
    )
    script = gr.File(label="Upload a text file")

    voice = gr.Dropdown(
        VOICE_OPTIONS, value="jane_eyre", label="Select voice:", type="value"
    )
    voice_b = gr.Dropdown(
        VOICE_OPTIONS,
        value="disabled",
        label="(Optional) Select second voice:",
        type="value",
    )
    split_by_newline = gr.Radio(
        ["Yes", "No"],
        label="Split by newline (If [No], it will automatically try to find relevant splits):",
        type="value",
        value="No",
    )

    output_audio = gr.Audio(label="streaming audio:", scale=10)
    # download_audio = gr.Audio(label="dowanload audio:")
    interface = gr.Interface(
        fn=inference,
        inputs=[
            text,
            script,
            voice,
            voice_b,
            split_by_newline,
        ],
        title=title,
        description=description,
        outputs=[output_audio],
    )
    interface.queue().launch(inbrowser=True)

if __name__ == "__main__":
    #tts = TextToSpeech(kv_cache=True, use_deepspeed=False, half=True)
    tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)

    with open("Tortoise_TTS_Runs_Scripts.log", "a") as f:
        f.write(
            f"\n\n-------------------------Tortoise TTS Scripts Logs, {datetime.now()}-------------------------\n"
        )

    main()