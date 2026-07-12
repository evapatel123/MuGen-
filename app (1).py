import gradio as gr
import os
import spaces
import json
import re
import random
import numpy as np
from gradio_client import Client, handle_file
hf_token = os.environ.get("HF_TOKEN")

MAX_SEED = np.iinfo(np.int32).max

def check_api(model_name):
    if model_name == "MAGNet":
        try :
            client = Client("fffiloni/MAGNet")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "AudioLDM-2":
        try :
            client = Client("fffiloni/audioldm2-text2audio-text2music-API")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "Riffusion":
        try :
            client = Client("fffiloni/spectrogram-to-music")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "Mustango":
        try :
            client = Client("fffiloni/mustango-API-2")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "MusicGen":
        try :
            client = Client("https://facebook-musicgen.hf.space/")
            return "api ready"
        except : 
            return "api not ready yet"
    elif model_name == "Stable Audio Open":
        try:
            client = Client("fffiloni/Stable-Audio-Open-A10", hf_token=hf_token)
            return "api ready"
        except:
            return "api not ready yet"
    elif model_name == "ACE Step":
        try :
            client = Client("fffiloni/ACE-Step-API", hf_token=hf_token)
            return "api ready"
        except : 
            return "api not ready yet"
    
        
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioClip

def extract_audio(video_in):
    input_video = video_in
    output_audio = 'audio.wav'
    
    # Open the video file and extract the audio
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    
    # Save the audio as a .wav file
    audio_clip.write_audiofile(output_audio, fps=44100)  # Use 44100 Hz as the sample rate for .wav files  
    print("Audio extraction complete.")

    return 'audio.wav'



def get_caption(image_in):
    kosmos2_client = Client("fffiloni/Kosmos-2-API")
    kosmos2_result = kosmos2_client.predict(
		image_input=handle_file(image_in),
		text_input="Detailed",
		api_name="/generate_predictions"
    )
    print(f"KOSMOS2 RETURNS: {kosmos2_result}")

    data = kosmos2_result[1]

    # Extract and combine tokens starting from the second element
    sentence = ''.join(item['token'] for item in data[1:])

    # Find the last occurrence of "."
    #last_period_index = full_sentence.rfind('.')

    # Truncate the string up to the last period
    #truncated_caption = full_sentence[:last_period_index + 1]

    # print(truncated_caption)
    #print(f"\n—\nIMAGE CAPTION: {truncated_caption}")
    
    return sentence

def get_caption_from_MD(image_in):
    client = Client("https://vikhyatk-moondream1.hf.space/")
    result = client.predict(
		image_in,	# filepath  in 'image' Image component
		"Describe precisely the image.",	# str  in 'Question' Textbox component
		api_name="/answer_question"
    )
    print(result)
    return result

def get_magnet(prompt):

    client = Client("fffiloni/MAGNet")
    result = client.predict(
        model="facebook/magnet-small-10secs",	# Literal['facebook/magnet-small-10secs', 'facebook/magnet-medium-10secs', 'facebook/magnet-small-30secs', 'facebook/magnet-medium-30secs', 'facebook/audio-magnet-small', 'facebook/audio-magnet-medium']  in 'Model' Radio component
        model_path="",	# str  in 'Model Path (custom models)' Textbox component
        text=prompt,	# str  in 'Input Text' Textbox component
        temperature=3,	# float  in 'Temperature' Number component
        topp=0.9,	# float  in 'Top-p' Number component
        max_cfg_coef=10,	# float  in 'Max CFG coefficient' Number component
        min_cfg_coef=1,	# float  in 'Min CFG coefficient' Number component
        decoding_steps1=20,	# float  in 'Decoding Steps (stage 1)' Number component
        decoding_steps2=10,	# float  in 'Decoding Steps (stage 2)' Number component
        decoding_steps3=10,	# float  in 'Decoding Steps (stage 3)' Number component
        decoding_steps4=10,	# float  in 'Decoding Steps (stage 4)' Number component
        span_score="prod-stride1 (new!)",	# Literal['max-nonoverlap', 'prod-stride1 (new!)']  in 'Span Scoring' Radio component
        api_name="/predict_full"
    )
    print(result)
    return result[1]

def get_audioldm(prompt):
    client = Client("fffiloni/audioldm2-text2audio-text2music-API")
    seed = random.randint(0, MAX_SEED)
    result = client.predict(
        prompt=prompt,	# str in 'Input text' Textbox component
        negative_prompt="Low quality.",	# str in 'Negative prompt' Textbox component
        duration=10,	# int | float (numeric value between 5 and 15) in 'Duration (seconds)' Slider component
        guidance_scale=6.5,	# int | float (numeric value between 0 and 7) in 'Guidance scale' Slider component
        random_seed=seed,	# int | float in 'Seed' Number component
        n_candidates=3,	# int | float (numeric value between 1 and 5) in 'Number waveforms to generate' Slider component
        api_name="/text2audio"
    )
    print(result)
    
    return result

def get_riffusion(prompt):
    client = Client("fffiloni/spectrogram-to-music")
    result = client.predict(
		prompt=prompt,	# str  in 'Musical prompt' Textbox component
		negative_prompt="",	# str  in 'Negative prompt' Textbox component
		audio_input=None,	# filepath  in 'parameter_4' Audio component
		duration=10,	# float (numeric value between 5 and 10) in 'Duration in seconds' Slider component
		api_name="/predict"
    )
    print(result)
    return result[1]

def get_mustango(prompt):
    client = Client("fffiloni/mustango-API-2")
    result = client.predict(
		prompt=prompt,	# str  in 'Prompt' Textbox component
		steps=200,	# float (numeric value between 100 and 200) in 'Steps' Slider component
		guidance=6,	# float (numeric value between 1 and 10) in 'Guidance Scale' Slider component
		api_name="/predict"
    )
    print(result)
    return result

def get_musicgen(prompt):
    client = Client("https://facebook-musicgen.hf.space/")
    result = client.predict(
        prompt,	# str  in 'Describe your music' Textbox component
        None,	# str (filepath or URL to file) in 'File' Audio component
        fn_index=0
    )
    print(result)
    return result[1]

def get_stable_audio_open(prompt):
    client = Client("fffiloni/Stable-Audio-Open-A10", hf_token=hf_token)
    result = client.predict(
		prompt=prompt,
		seconds_total=10,
		steps=100,
		cfg_scale=7,
		api_name="/predict"
    )
    print(result)
    return result

def get_ace(prompt):
    from gradio_client import Client, handle_file

    client = Client("fffiloni/ACE-Step-API", hf_token=hf_token)
    result = client.predict(
		audio_duration=-1,
		prompt=prompt,
		lyrics="[inst]",
		infer_step=60,
		guidance_scale=15,
		scheduler_type="euler",
		cfg_type="apg",
		omega_scale=10,
		manual_seeds=None,
		guidance_interval=0.5,
		guidance_interval_decay=0,
		min_guidance_scale=3,
		use_erg_tag=True,
		use_erg_lyric=False,
		use_erg_diffusion=True,
		oss_steps=None,
		guidance_scale_text=0,
		guidance_scale_lyric=0,
		audio2audio_enable=False,
		ref_audio_strength=0.5,
		ref_audio_input=None,
		lora_name_or_path="none",
		api_name="/__call__"
    )
    print(result)
    return result[0]


import re
import torch
from transformers import pipeline

zephyr_model = "HuggingFaceH4/zephyr-7b-beta"
mixtral_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

pipe = pipeline("text-generation", model=zephyr_model, torch_dtype=torch.bfloat16, device_map="auto")

standard_sys = f"""
You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users.
In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model.
For example, if a user says, "a picture of a man in a black suit and tie riding a black dragon", provide immediately a musical prompt corresponding to the image description. 
Immediately STOP after that. It should be EXACTLY in this format:
"A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle"
"""

mustango_sys = f"""
You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users.
In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model, you MUST include chords progression.
For example, if a user says, "a painting of three old women having tea party", provide immediately a musical prompt corresponding to the image description. 
Immediately STOP after that. It should be EXACTLY in this format:
"The song is an instrumental. The song is in medium tempo with a classical guitar playing a lilting melody in accompaniment style. The song is emotional and romantic. The song is a romantic instrumental song. The chord sequence is Gm, F6, Ebm. The time signature is 4/4. This song is in Adagio. The key of this song is G minor."
"""

@spaces.GPU()
def get_musical_prompt(user_prompt, chosen_model):

    """
    if chosen_model == "Mustango" :
        agent_maker_sys = standard_sys
    else :
        agent_maker_sys = standard_sys
    """
    agent_maker_sys = standard_sys
    
    instruction = f"""
<|system|>
{agent_maker_sys}</s>
<|user|>
"""
    
    prompt = f"{instruction.strip()}\n{user_prompt}</s>"    
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    pattern = r'\<\|system\|\>(.*?)\<\|assistant\|\>'
    cleaned_text = re.sub(pattern, '', outputs[0]["generated_text"], flags=re.DOTALL)
    
    print(f"SUGGESTED Musical prompt: {cleaned_text}")
    return cleaned_text.lstrip("\n")

def infer(image_in, chosen_model):
    """
    Generate music from an input image and selected music generation model.
    This function performs the following steps:
    1. Checks that an image and a model have been provided.
    2. Verifies if the selected model's API is currently available.
    3. Uses an image captioning model (Kosmos-2) to describe the image.
    4. Generates a musical prompt from the image caption using a language model.
    5. Sends the musical prompt to the selected music generation model and retrieves the result.
    Args:
        image_in: The filepath to an input image. This image is used as inspiration to generate music.
        chosen_model: The name of the model to use for music generation. Supported values include: "Mustango", "AudioLDM-2", "Riffusion", "ACE Step", "Stable Audio Open".
    Returns:
        - A string containing the musical prompt generated from the image.
        - A flag to show the retry button in the UI (for user to edit and retry the generation).
        - The output of the selected model, typically an audio filepath or object depending on model.
    """
    if image_in == None :
        raise gr.Error("Please provide an image input")

    if chosen_model == [] :
        raise gr.Error("Please pick a model")

    api_status = check_api(chosen_model)

    if api_status == "api not ready yet" :
        raise gr.Error("This model is not ready yet, you can pick another one instead :)")
    
    gr.Info("Getting image caption with Kosmos-2...")
    user_prompt = get_caption(image_in)
    #user_prompt = get_caption_from_MD(image_in)
    
    gr.Info("Building a musical prompt according to the image caption ...")
    musical_prompt = get_musical_prompt(user_prompt, chosen_model)

    if chosen_model == "MAGNet" :
        gr.Info("Now calling MAGNet for music...")
        music_o = get_magnet(musical_prompt)
    elif chosen_model == "AudioLDM-2" :
        gr.Info("Now calling AudioLDM-2 for music...")
        music_o = get_audioldm(musical_prompt)
    elif chosen_model == "Riffusion" :
        gr.Info("Now calling Riffusion for music...")
        music_o = get_riffusion(musical_prompt)
    elif chosen_model == "Mustango" :
        gr.Info("Now calling Mustango for music...")
        music_o = get_mustango(musical_prompt)
    elif chosen_model == "MusicGen" :
        gr.Info("Now calling MusicGen for music...")
        music_o = get_musicgen(musical_prompt)
    elif chosen_model == "Stable Audio Open" :
        gr.Info("Now calling Stable Audio Open for music...")
        music_o = get_stable_audio_open(musical_prompt)
    elif chosen_model == "ACE Step" :
        gr.Info("Now calling ACE Step for music...")
        music_o = get_ace(musical_prompt)
    
    return gr.update(value=musical_prompt, interactive=True), gr.update(visible=True), music_o

def retry(chosen_model, caption):
    musical_prompt = caption
    music_o = None

    if chosen_model == "MAGNet" :
        gr.Info("Now calling evapatel123/MAGNet for music...")
        music_o = get_magnet(musical_prompt)
    elif chosen_model == "AudioLDM-2" :
        gr.Info("Now calling evapatel123/AudioLDM-2 for music...")
        music_o = get_audioldm(musical_prompt)
    elif chosen_model == "Riffusion" :
        gr.Info("Now calling evapatel123/Riffusion for music...")
        music_o = get_riffusion(musical_prompt)
    elif chosen_model == "Mustango" :
        gr.Info("Now calling evapatel123/Mustango for music...")
        music_o = get_mustango(musical_prompt)
    elif chosen_model == "MusicGen" :
        gr.Info("Now calling evapatel123/MusicGen for music...")
        music_o = get_musicgen(musical_prompt)
    elif chosen_model == "Stable Audio Open" :
        gr.Info("Now calling Stable Audio Open for music...")
        music_o = get_stable_audio_open(musical_prompt)
    elif chosen_model == "ACE Step" :
        gr.Info("Now calling ACE Step for music...")
        music_o = get_ace(musical_prompt)

    return music_o


# --- Cyberpunk Theme Configurations ---
demo_title = "MuGen - Image to Music "
description = "SYSTEM STATUS: ACTIVE // EXTRACTING AUDIO MATRIX FROM IMAGE SPECTRUM"

# Premium Neon Cyberpunk CSS Injection
css = """
/* Core Container Glow & Dark Mode Base */
#col-container {
    margin: 0 auto;
    max-width: 1050px;
    text-align: left;
    background: #0a0512;
    padding: 30px;
    border-radius: 12px;
    border: 2px solid #ff007f;
    box-shadow: 0 0 20px rgba(255, 0, 127, 0.2), inset 0 0 15px rgba(138, 43, 226, 0.2);
    font-family: 'Courier New', Courier, monospace;
}

/* Neon Typography Styles */
.cyber-title {
    text-transform: uppercase;
    background: linear-gradient(45deg, #ff007f, #8a2be2, #00ffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 10px rgba(255, 0, 127, 0.5);
    font-weight: 900;
    letter-spacing: 2px;
}

.cyber-subtitle {
    color: #00ffff;
    font-size: 14px;
    letter-spacing: 3px;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    margin-bottom: 20px;
}

/* Custom Textareas & Inputs Customizing */
#inspi-prompt textarea {
    font-size: 18px;
    line-height: 24px;
    font-weight: 600;
    color: #ff007f !important;
    background-color: #120a24 !important;
    border: 1px solid #8a2be2 !important;
    box-shadow: 0 0 8px rgba(138, 43, 226, 0.4) !important;
}

/* Cyberpunk Primary Button Glow Matrix */
.cyber-btn-primary {
    background: linear-gradient(135deg, #ff007f 0%, #8a2be2 100%) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: 0 0 15px rgba(255, 0, 127, 0.6) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.cyber-btn-primary:hover {
    box-shadow: 0 0 25px rgba(255, 0, 127, 0.9) !important;
    transform: scale(1.02);
}

/* Cyberpunk Secondary Button Glow Matrix */
.cyber-btn-secondary {
    background: transparent !important;
    color: #00ffff !important;
    border: 2px solid #00ffff !important;
    font-weight: bold !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}

.cyber-btn-secondary:hover {
    background: rgba(0, 255, 255, 0.1) !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.7) !important;

.cyber-copyright {
    text-align: center; 
    color: #00ffff !important; 
    font-family: 'Courier New', monospace; 
    font-size: 11px; 
    margin-top: 40px; 
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.3);
    letter-spacing: 2px;
}
}
"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    with gr.Column(elem_id="col-container"):
        
        # Cyberpunk Header Segment
        gr.HTML(f"""
        <h1 class="cyber-title" style="text-align: center; font-size: 3rem; margin-bottom: 5px;">{demo_title}</h1>
        <p class="cyber-subtitle" style="text-align: center;">{description}</p>
        <hr style="border: 0; height: 1px; background: linear-gradient(to right, transparent, #ff007f, #8a2be2, #00ffff, transparent); margin-bottom: 25px;"/>
        """)
        
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    label="📷 INPUT SPECTRUM (IMAGE REFERENCE)",
                    type="filepath",
                    elem_id="image-in"
                )
                
                with gr.Row():
                    chosen_model = gr.Dropdown(
                        label="🔮 CORE ENGINE MODEL",
                        choices=[
                            "AudioLDM-2",
                            "Riffusion",
                            "Mustango",
                        ],
                        value=None,
                        filterable=False
                    )
                    
                    check_status = gr.Textbox(
                        label="🌐 CORE API STATUS",
                        interactive=False
                    )
                
                # High visibility primary trigger action button
                submit_btn = gr.Button("⚡ COMPILE MUSIC MATRIX ⚡", elem_classes="cyber-btn-primary")
                """gr.Examples(
                    examples=[
                        ["examples/ocean_poet.jpeg"],
                        ["examples/jasper_horace.jpeg"],
                        ["examples/summer.jpeg"],
                        ["examples/mona_diner.png"],
                        ["examples/monalisa.png"],
                        ["examples/santa.png"],
                        ["examples/winter_hiking.png"],
                        ["examples/teatime.jpeg"],
                        ["examples/news_experts.jpeg"]
                    ],
                    fn=None, # Links back to your main architecture function mappings
                    inputs=[image_in, chosen_model],
                    examples_per_page=4
                )"""
                
                
            with gr.Column():
                caption = gr.Textbox(
                    label="📟 MATRIX INTERPRETATION (MUSICAL PROMPT)",
                    interactive=True,
                    elem_id="inspi-prompt"
                )
                
                # Interactive modification buttons
                retry_btn = gr.Button("🔄 RE-CALIBRATE PROMPT MATRIX", visible=False, elem_classes="cyber-btn-secondary")
                
                result = gr.Audio(
                    label="🎵 OUTPUT SONIC WAVEFORM"
                )
        gr.HTML('<p class="cyber-copyright">© 2026 MuGen V2 // By evapatel123, fffiloni</p>')

    # Note: Keep your existing structural endpoint handlers (.change and .click bindings) mapping right under here intact!
        
        

    chosen_model.change(
        fn = check_api,
        inputs = chosen_model,
        outputs = check_status,
        queue = False,
        api_visibility='undocumented'
    )

    retry_btn.click(
        fn = retry,
        inputs = [chosen_model, caption],
        outputs = [result],
        api_visibility='undocumented'
    )
    
    submit_btn.click(
        fn = infer,
        inputs = [
            image_in,
            chosen_model,
            #check_status
        ],
        outputs =[
            caption,
            retry_btn,
            result
        ]
    )

demo.queue(max_size=16).launch(show_error=True, ssr_mode=False, mcp_server=True)