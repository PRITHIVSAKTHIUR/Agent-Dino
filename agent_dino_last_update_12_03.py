import os
import random
import uuid
import json
import time
import asyncio
import tempfile
from threading import Thread
import base64
import shutil
import re

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import edge_tts
import trimesh
import soundfile as sf  # New import for audio file reading

import supervision as sv
from ultralytics import YOLO as YOLODetector
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.image_utils import load_image

from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers import ShapEImg2ImgPipeline, ShapEPipeline
from diffusers.utils import export_to_ply

os.system('pip install backoff')
# Global constants and helper functions

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def glb_to_data_url(glb_path: str) -> str:
    """
    Reads a GLB file from disk and returns a data URL with a base64 encoded representation.
    (Not used in this method.)
    """
    with open(glb_path, "rb") as f:
        data = f.read()
    b64_data = base64.b64encode(data).decode("utf-8")
    return f"data:model/gltf-binary;base64,{b64_data}"

def progress_bar_html(label: str) -> str:
    """
    Returns an HTML snippet for a thin progress bar with a label.
    The progress bar is styled as a dark red animated bar.
    """
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #AFEEEE; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #00FFFF; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

# Model class for Text-to-3D Generation (ShapE)

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)
        self.pipe.to(self.device)
        # Ensure the text encoder is in half precision to avoid dtype mismatches.
        if torch.cuda.is_available():
            try:
                self.pipe.text_encoder = self.pipe.text_encoder.half()
            except AttributeError:
                pass

        self.pipe_img = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16)
        self.pipe_img.to(self.device)
        # Use getattr with a default value to avoid AttributeError if text_encoder is missing.
        if torch.cuda.is_available():
            text_encoder_img = getattr(self.pipe_img, "text_encoder", None)
            if text_encoder_img is not None:
                self.pipe_img.text_encoder = text_encoder_img.half()

    def to_glb(self, ply_path: str) -> str:
        mesh = trimesh.load(ply_path)
        # Rotate the mesh for proper orientation
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        mesh.apply_transform(rot)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(mesh_path.name, file_type="glb")
        return mesh_path.name

    def run_text(self, prompt: str, seed: int = 0, guidance_scale: float = 15.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)

    def run_image(self, image: Image.Image, seed: int = 0, guidance_scale: float = 3.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe_img(
            image,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)

# New Tools for Web Functionality using DuckDuckGo and smolagents

from typing import Any, Optional
from smolagents.tools import Tool
import duckduckgo_search

class DuckDuckGoSearchTool(Tool):
    name = "web_search"
    description = "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."
    inputs = {'query': {'type': 'string', 'description': 'The search query to perform.'}}
    output_type = "string"

    def __init__(self, max_results=10, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [
            f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
        ]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    inputs = {'url': {'type': 'string', 'description': 'The url of the webpage to visit.'}}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def forward(self, url: str) -> str:
        try:
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException

            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, 10000)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
            
# rAgent Reasoning using Llama mode OpenAI

from openai import OpenAI

ACCESS_TOKEN = os.getenv("HF_TOKEN")
ragent_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=ACCESS_TOKEN,
)

SYSTEM_PROMPT = """

        "You are an expert assistant who solves tasks using Python code. Follow these steps:\n"
        "1. **Thought**: Explain your reasoning and plan for solving the task.\n"
        "2. **Code**: Write Python code to implement your solution.\n"
        "3. **Observation**: Analyze the output of the code and summarize the results.\n"
        "4. **Final Answer**: Provide a concise conclusion or final result.\n\n"
        f"Task: {{task}}"

"""

def ragent_reasoning(prompt: str, history: list[dict], max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.95):
    """
    Uses the Llama mode OpenAI model to perform a structured reasoning chain.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Incorporate conversation history (if any)
    for msg in history:
        if msg.get("role") == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg.get("role") == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    response = ""
    stream = ragent_client.chat.completions.create(
         model="meta-llama/Meta-Llama-3.1-8B-Instruct",
         max_tokens=max_tokens,
         stream=True,
         temperature=temperature,
         top_p=top_p,
         messages=messages,
    )
    for message in stream:
         token = message.choices[0].delta.content
         response += token
         yield response

# ------------------------------------------------------------------------------
# New Phi-4 Multimodal Feature (Image & Audio)
# ------------------------------------------------------------------------------
# Define prompt structure for Phi-4
phi4_user_prompt = '<|user|>'
phi4_assistant_prompt = '<|assistant|>'
phi4_prompt_suffix = '<|end|>'

# Load Phi-4 multimodal model and processor using unique variable names
phi4_model_path = "microsoft/Phi-4-multimodal-instruct"
phi4_processor = AutoProcessor.from_pretrained(phi4_model_path, trust_remote_code=True)
phi4_model = AutoModelForCausalLM.from_pretrained(
    phi4_model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
)

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Models and Pipelines for Chat, Image, and Multimodal Processing
# Load the text-only model and tokenizer (for pure text chat)

model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

# Voices for text-to-speech
TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

# Load multimodal processor and model (e.g. for OCR and image processing)
MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

# Asynchronous text-to-speech

async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    """Convert text to speech using Edge TTS and save as MP3"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

# Utility function to clean conversation history

def clean_chat_history(chat_history):
    """
    Filter out any chat entries whose "content" is not a string.
    This helps prevent errors when concatenating previous messages.
    """
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

# Stable Diffusion XL Pipeline for Image Generation 
# Model In Use : SG161222/RealVisXL_V5.0_Lightning

MODEL_ID_SD = os.getenv("MODEL_VAL_PATH")  # SDXL Model repository path via env variable 
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # For batched image generation

sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID_SD,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    add_watermarker=False,
).to(device)
sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

if torch.cuda.is_available():
    sd_pipe.text_encoder = sd_pipe.text_encoder.half()

if USE_TORCH_COMPILE:
    sd_pipe.compile()

if ENABLE_CPU_OFFLOAD:
    sd_pipe.enable_model_cpu_offload()

def save_image(img: Image.Image) -> str:
    """Save a PIL image with a unique filename and return the path."""
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

@spaces.GPU(duration=60, enable_queue=True)
# SG161222/RealVisXL_V5.0_Lightning
def generate_image_fn(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    num_inference_steps: int = 25,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    num_images: int = 1,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate images using the SDXL pipeline."""
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    options = {
        "prompt": [prompt] * num_images,
        "negative_prompt": [negative_prompt] * num_images if use_negative_prompt else None,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "output_type": "pil",
    }
    if use_resolution_binning:
        options["use_resolution_binning"] = True

    images = []
    # Process in batches
    for i in range(0, num_images, BATCH_SIZE):
        batch_options = options.copy()
        batch_options["prompt"] = options["prompt"][i:i+BATCH_SIZE]
        if "negative_prompt" in batch_options and batch_options["negative_prompt"] is not None:
            batch_options["negative_prompt"] = options["negative_prompt"][i:i+BATCH_SIZE]
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = sd_pipe(**batch_options)
        else:
            outputs = sd_pipe(**batch_options)
        images.extend(outputs.images)
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

# Text-to-3D Generation using the ShapE Pipeline

@spaces.GPU(duration=120, enable_queue=True)
def generate_3d_fn(
    prompt: str,
    seed: int = 1,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
    randomize_seed: bool = False,
):
    """
    Generate a 3D model from text using the ShapE pipeline.
    Returns a tuple of (glb_file_path, used_seed).
    """
    seed = int(randomize_seed_fn(seed, randomize_seed))
    model3d = Model()
    glb_path = model3d.run_text(prompt, seed=seed, guidance_scale=guidance_scale, num_steps=num_steps)
    return glb_path, seed

# YOLO Object Detection Setup
YOLO_MODEL_REPO = "strangerzonehf/Flux-Ultimate-LoRA-Collection"
YOLO_CHECKPOINT_NAME = "images/demo.pt"
yolo_model_path = hf_hub_download(repo_id=YOLO_MODEL_REPO, filename=YOLO_CHECKPOINT_NAME)
yolo_detector = YOLODetector(yolo_model_path)

def detect_objects(image: np.ndarray):
    """Runs object detection on the input image."""
    results = yolo_detector(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    return Image.fromarray(annotated_image)

# Chat Generation Function with support for @tts, @image, @3d, @web, @rAgent, @yolo, and now @phi4 commands

@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates chatbot responses with support for multimodal input and special commands:
      - "@tts1" or "@tts2": triggers text-to-speech.
      - "@image": triggers image generation using the SDXL pipeline.
      - "@3d": triggers 3D model generation using the ShapE pipeline.
      - "@web": triggers a web search or webpage visit.
      - "@rAgent": initiates a reasoning chain using Llama mode.
      - "@yolo": triggers object detection using YOLO.
      - **"@phi4": triggers multimodal (image/audio) processing using the Phi-4 model.**
    """
    text = input_dict["text"]
    files = input_dict.get("files", [])

    # --- 3D Generation branch ---
    if text.strip().lower().startswith("@3d"):
        prompt = text[len("@3d"):].strip()
        yield progress_bar_html("Processing 3D Mesh Generation")
        glb_path, used_seed = generate_3d_fn(
            prompt=prompt,
            seed=1,
            guidance_scale=15.0,
            num_steps=64,
            randomize_seed=True,
        )
        # Copy the GLB file to a static folder.
        yield progress_bar_html("Finalizing 3D Mesh Generation")
        static_folder = os.path.join(os.getcwd(), "static")
        if not os.path.exists(static_folder):
            os.makedirs(static_folder)
        new_filename = f"mesh_{uuid.uuid4()}.glb"
        new_filepath = os.path.join(static_folder, new_filename)
        shutil.copy(glb_path, new_filepath)
        
        yield gr.File(new_filepath)
        return

    # --- Image Generation branch ---
    if text.strip().lower().startswith("@image"):
        prompt = text[len("@image"):].strip()
        yield progress_bar_html("Generating Image")
        image_paths, used_seed = generate_image_fn(
            prompt=prompt,
            negative_prompt="",
            use_negative_prompt=False,
            seed=1,
            width=1024,
            height=1024,
            guidance_scale=3,
            num_inference_steps=25,
            randomize_seed=True,
            use_resolution_binning=True,
            num_images=1,
        )
        yield gr.Image(image_paths[0])
        return

    # --- Web Search/Visit branch ---
    if text.strip().lower().startswith("@web"):
        web_command = text[len("@web"):].strip()
        # If the command starts with "visit", then treat the rest as a URL
        if web_command.lower().startswith("visit"):
            url = web_command[len("visit"):].strip()
            yield progress_bar_html("Visiting Webpage")
            visitor = VisitWebpageTool()
            content = visitor.forward(url)
            yield content
        else:
            # Otherwise, treat the rest as a search query.
            query = web_command
            yield progress_bar_html("Performing Web Search")
            searcher = DuckDuckGoSearchTool()
            results = searcher.forward(query)
            yield results
        return

    # --- rAgent Reasoning branch ---
    if text.strip().lower().startswith("@ragent"):
        prompt = text[len("@ragent"):].strip()
        yield progress_bar_html("Processing Reasoning Chain")
        # Pass the current chat history (cleaned) to help inform the chain.
        for partial in ragent_reasoning(prompt, clean_chat_history(chat_history)):
            yield partial
        return

    # --- YOLO Object Detection branch ---
    if text.strip().lower().startswith("@yolo"):
        yield progress_bar_html("Performing Object Detection")
        if not files or len(files) == 0:
            yield "Error: Please attach an image for YOLO object detection."
            return
        # Use the first attached image
        input_file = files[0]
        try:
            if isinstance(input_file, str):
                pil_image = Image.open(input_file)
            else:
                pil_image = input_file
        except Exception as e:
            yield f"Error loading image: {str(e)}"
            return
        np_image = np.array(pil_image)
        result_img = detect_objects(np_image)
        yield gr.Image(result_img)
        return

    # --- Phi-4 Multimodal branch (Image/Audio) with Streaming ---
    if text.strip().lower().startswith("@phi4"):
        question = text[len("@phi4"):].strip()
        if not files:
            yield "Error: Please attach an image or audio file for @phi4 multimodal processing."
            return
        if not question:
            yield "Error: Please provide a question after @phi4."
            return
        # Determine input type (Image or Audio) from the first file
        input_file = files[0]
        try:
            # If file is already a PIL Image, treat as image
            if isinstance(input_file, Image.Image):
                input_type = "Image"
                file_for_phi4 = input_file
            else:
                # Try opening as image; if it fails, assume audio
                try:
                    file_for_phi4 = Image.open(input_file)
                    input_type = "Image"
                except Exception:
                    input_type = "Audio"
                    file_for_phi4 = input_file
        except Exception:
            input_type = "Audio"
            file_for_phi4 = input_file

        if input_type == "Image":
            phi4_prompt = f'{phi4_user_prompt}<|image_1|>{question}{phi4_prompt_suffix}{phi4_assistant_prompt}'
            inputs = phi4_processor(text=phi4_prompt, images=file_for_phi4, return_tensors='pt').to(phi4_model.device)
        elif input_type == "Audio":
            phi4_prompt = f'{phi4_user_prompt}<|audio_1|>{question}{phi4_prompt_suffix}{phi4_assistant_prompt}'
            audio, samplerate = sf.read(file_for_phi4)
            inputs = phi4_processor(text=phi4_prompt, audios=[(audio, samplerate)], return_tensors='pt').to(phi4_model.device)
        else:
            yield "Invalid file type for @phi4 multimodal processing."
            return

        # Initialize the streamer
        streamer = TextIteratorStreamer(phi4_processor, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare generation kwargs
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 200,
            "num_logits_to_keep": 0,
        }

        # Start generation in a separate thread
        thread = Thread(target=phi4_model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the response
        buffer = ""
        yield progress_bar_html("Processing Phi-4 Multimodal")
        for new_text in streamer:
            buffer += new_text
            time.sleep(0.01)  # Small delay to simulate real-time streaming
            yield buffer
        return

    # --- Text and TTS branch ---
    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)
    
    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        conversation = [{"role": "user", "content": text}]
    else:
        voice = None
        text = text.replace(tts_prefix, "").strip()
        conversation = clean_chat_history(chat_history)
        conversation.append({"role": "user", "content": text})

    if files:
        if len(files) > 1:
            images = [load_image(image) for image in files]
        elif len(files) == 1:
            images = [load_image(files[0])]
        else:
            images = []
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        yield progress_bar_html("Processing with Qwen2VL OCR")
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
    else:
        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        outputs = []
        yield progress_bar_html("Processing Chat Response")
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)

        final_response = "".join(outputs)
        yield final_response

        if is_tts and voice:
            output_file = asyncio.run(text_to_speech(final_response, voice))
            yield gr.Audio(output_file, autoplay=True)

# Gradio Chat Interface Setup and Launch

demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        [{"text": "@phi4 Transcribe the audio to text.", "files": ["examples/harvard.wav"]}],
        [{"text": "@phi4 Summarize the content", "files": ["examples/write.jpg"]}],
        [{"text": "Explain the Image", "files": ["examples/3.jpg"]}],
        [{"text": "Extract as JSON table from the table", "files": ["examples/4.jpg"]}],
        ["@image Chocolate dripping from a donut"],
        ["@3d A birthday cupcake with cherry"],
        ["@image A drawing of an man made out of hamburger, blue sky background, soft pastel colors"],
        ["@tts2 What causes rainbows to form?"],
        [{"text": "Summarize the letter", "files": ["examples/1.png"]}],
        [{"text": "@yolo", "files": ["examples/yolo.jpeg"]}],
        ["@rAgent Explain how a binary search algorithm works."],
        ["@web Is Grok-3 Beats DeepSeek-R1 at Reasoning ?"],
        ["@tts1 Explain Tower of Hanoi"],
        ["Python Program for Array Rotation"],
    ],
    cache_examples=False,
    type="messages",
    description="# **Agent Dino `@phi4 'prompt..', @image, etc..`**",
    fill_height=True,
    textbox=gr.MultimodalTextbox(
        label="Query Input", 
        file_types=["image", "audio"],
        file_count="multiple", 
        placeholder="‎ @tts1, @tts2, @image, @3d, @phi4 [image, audio], @rAgent, @web, @yolo, default [plain text]"
    ),
    stop_btn="Stop Generation",
    multimodal=True,
)

# Ensure the static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

from fastapi.staticfiles import StaticFiles
demo.app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    demo.queue(max_size=20).launch(mcp_server=True, share=True)
